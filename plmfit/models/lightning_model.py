import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
import time
import json
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning.pytorch.strategies import DeepSpeedStrategy
from plmfit.shared_utils import utils
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
import os
import torch.distributed as dist
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassMatthewsCorrCoef,
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryMatthewsCorrCoef,
    BinaryROC,
    MultilabelAccuracy,
    MultilabelMatthewsCorrCoef,
    MultilabelConfusionMatrix,
)
from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    SpearmanCorrCoef,
)
from torchmetrics.text import Perplexity
from lightning.pytorch.callbacks import BasePredictionWriter
from plmfit.shared_utils.custom_loss_functions import MaskedBCEWithLogitsLoss, MaskedFocalWithLogitsLoss
import numpy as np


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model,
        training_config=None,
        plmfit_logger=None,
        log_interval=-1,
        method="lora",
        experimenting=False,
        train=True,
    ):
        torch.set_float32_matmul_precision("medium")
        super().__init__()
        self.model = model
        self.save_hyperparameters(training_config)
        if train:
            self.loss_function = self.initialize_loss_function()
        self.plmfit_logger = plmfit_logger
        self.log_interval = log_interval
        self.method = method

        if train:
            if "no_classes" not in self.hparams:
                self.hparams.no_classes = 1
            if self.model.task == "classification":
                if self.hparams.no_classes < 2:
                    self.train_metric = (
                        BinaryAccuracy()
                    )  # Binary accuracy requires 1 dimension at output
                else:
                    self.train_metric = MulticlassAccuracy(
                        num_classes=self.hparams.no_classes
                    )
                self.metric_label = "accuracy"
            elif self.model.task == "regression":
                self.train_metric = MeanSquaredError(squared=False)
                self.metric_label = "rmse"
            elif self.model.task == "masked_lm":
                self.train_metric = Perplexity(ignore_index=-100)
                self.metric_label = "perplexity"
            elif self.model.task == "token_classification":
                self.train_metric = MulticlassAccuracy(
                    num_classes=self.hparams.no_classes, ignore_index=-100
                )
                self.metric_label = "accuracy"
            elif self.model.task == "multilabel_classification":
                self.train_metric = MultilabelAccuracy(
                    num_labels=self.hparams.no_labels, ignore_index=-100
                )
                self.metric_label = "accuracy"
            else:
                raise ValueError(f"Unsupported task: {self.model.task}")

            self.val_metric = self.train_metric.clone()

            self.metrics = Metrics(
                self.model.task,
                no_classes=(
                    1 if "no_classes" not in self.hparams else self.hparams.no_classes
                ),
                no_labels=(
                    1 if "no_labels" not in self.hparams else self.hparams.no_labels
                ),
            )

            self.track_validation_after = 0
            self.track_training_loss = False

        self.profiling_interval = 100

        self.experimenting = experimenting

    def forward(self, input, **args):
        output = self.model(input, **args)
        return output

    # If code hangs here, try https://github.com/microsoft/DeepSpeed/issues/2816
    def on_fit_start(self) -> None:
        self.start_time = time.time()
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.plmfit_logger.log(self.model)
        utils.get_parameters(self.model, self.plmfit_logger)
        self.plmfit_logger.current_global_rank = self.trainer.global_rank

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0

        # To avoid error when min scale is reached
        if torch.cuda.is_available() and isinstance(
            self.trainer.strategy, DeepSpeedStrategy
        ):
            self.trainer.strategy.model.optimizer.loss_scaler.raise_error_at_min_scale = (
                False
            )
        # print all available properties for strategy
        if torch.cuda.is_available() and isinstance(
            self.trainer.strategy, DeepSpeedStrategy
        ):
            self.plmfit_logger.log(self.trainer.strategy.model.wall_clock_breakdown())
        if torch.cuda.is_available() and isinstance(
            self.trainer.strategy, DeepSpeedStrategy
        ):
            self.profiler = FlopsProfiler(self, ds_engine=self.trainer.strategy.model)

    def on_fit_end(self) -> None:
        total_time = time.time() - self.start_time
        self.plmfit_logger.log(
            f"\nMean time per epoch: {total_time/(self.current_epoch):.4f}s"
        )
        self.plmfit_logger.log(f"Total training time: {total_time:.1f}s")
        loss_data = {
            "epoch_train_loss": self.epoch_train_loss,
            "epoch_val_loss": self.epoch_val_loss,
        }
        if self.trainer.global_rank == 0:
            with open(
                f"{self.plmfit_logger.base_dir}/{self.plmfit_logger.experiment_name}_loss.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(loss_data, f, indent=4)
            report = {
                "training_time": f"{total_time:.1f}",
                "avg_time_per_epoch": f"{total_time/(self.current_epoch):.4f}",
            }
            self.plmfit_logger.save_data(report, "report")

    ### TRAINING STEPS ###
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        self.plmfit_logger.log(
            "\nEpoch {}/{}".format(self.current_epoch + 1, self.trainer.max_epochs)
        )
        self.plmfit_logger.log("-" * 10)
        self.on_profiling = False

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx == 0 and self.experimenting:
            self.batches_start_time = (
                time.time()
            )  # To measure precisely the time of the first batch
        if batch_idx == 99 and self.experimenting:
            self.profiler.start_profile()
            print("FLOPS PROFILING INITIATED...", flush=True)
            self.plmfit_logger.log(
                f"Avg iter time: {(time.time() - self.batches_start_time)/100:.4f} s"
            )

    def training_step(self, batch, batch_idx):
        batch_start_time = time.time()
        if self.model.task == "masked_lm":
            input = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = self(input, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            outputs = outputs.logits.squeeze(dim=1)
            outputs = outputs.to(torch.float32)
        else:
            input, labels = batch
            outputs = self(input)
            # No squeezing, leave logits as is for CrossEntropyLoss
            if self.model.task == "classification" and self.hparams.no_classes > 1:
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                labels = torch.nn.functional.one_hot(
                    labels.long(), num_classes=self.hparams.no_classes
                )
                labels = labels.float()
            elif (
                self.model.task == "token_classification"
                and self.hparams.no_classes > 1
            ):
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                # swap 3rd dimension to 2nd dimension
                outputs = outputs.permute(0, 2, 1)
                # Convert labels to long
                labels = labels.long()
            elif self.model.task == "multilabel_classification":
                if self.hparams.no_classes < 2:
                    if hasattr(outputs, "logits"):
                        outputs = outputs.logits
            else:
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits.squeeze(dim=1)
                else:
                    outputs = outputs.squeeze(dim=1)
            loss = self.loss_function(outputs, labels)

        if self.model.task == "classification" and self.hparams.no_classes > 1:
            labels = torch.argmax(labels, dim=1)
            outputs = torch.argmax(outputs, dim=1)
        if self.model.task == "token_classification" and self.hparams.no_classes > 1:
            # Get the maximum value of the 3rd dimension
            outputs = torch.argmax(outputs, dim=1)
        if (
            self.model.task == "multilabel_classification"
            and self.hparams.no_classes == 1
        ):
            # Logits loss function must be being used so we have to convert to probabilities
            outputs = torch.sigmoid(outputs)
            labels = labels.int()

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.train_metric.update(outputs, labels)
        self.log(
            f"train_{self.metric_label}_step",
            self.train_metric,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )

        if self.log_interval != -1 and batch_idx % self.log_interval == 0:
            self.plmfit_logger.log(
                f"(train) batch : {batch_idx + 1}  / {len(self.trainer.train_dataloader)} | running_loss : {loss} (batch time : {time.time() - batch_start_time:.4f})"
            )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx == 99 and self.experimenting:
            # self.profiler.print_model_profile(profile_step=batch_idx, output_file=f'{self.plmfit_logger.base_dir}/flops.log')
            self.profiler.end_profile()

            with open(f"{self.plmfit_logger.base_dir}/flops.log", "w") as f:
                f.write(
                    f"Avg iter time: {(time.time() - self.batches_start_time)/100:.4f} s\n"
                )

            print("Successful test")
            raise SystemExit("Experiment over")

    def on_train_epoch_end(self):
        self.log(f"train_{self.metric_label}_epoch", self.train_metric, sync_dist=True)
        self.epoch_train_loss.append(
            self.trainer.logged_metrics["train_loss_epoch"].item()
        )
        if self.plmfit_logger:
            self.plmfit_logger.log(
                f'(train) loss: {self.trainer.logged_metrics["train_loss_epoch"]:.4f} {time.time() - self.epoch_start_time:.4f}s'
            )
            self.plmfit_logger.log(
                f'(train) {self.metric_label}: {self.trainer.logged_metrics[f"train_{self.metric_label}_epoch"]:.4f}'
            )
        if self.experimenting:
            print("Successful test")
            raise SystemExit("Experiment over")
        total_time = time.time() - self.start_time
        self.plmfit_logger.log(
            f"\nMean time per epoch: {total_time/(self.current_epoch+1):.4f}s"
        )
        self.plmfit_logger.log(f"Total training time: {total_time:.1f}s")
        loss_data = {
            "epoch_train_loss": self.epoch_train_loss,
            "epoch_val_loss": self.epoch_val_loss,
        }
        if self.trainer.global_rank == 0:
            with open(
                f"{self.plmfit_logger.base_dir}/{self.plmfit_logger.experiment_name}_loss.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(loss_data, f, indent=4)
            report = {
                "training_time": f"{total_time:.1f}",
                "avg_time_per_epoch": f"{total_time/(self.current_epoch+1):.4f}",
            }
            self.plmfit_logger.save_data(report, "report")

    ### VALIDATION STEPS ###
    def on_validation_epoch_start(self):
        if self.trainer.sanity_checking:
            if self.plmfit_logger:
                self.plmfit_logger.log(f"Sanity checking...")

    def validation_step(self, batch, batch_idx):
        batch_start_time = time.time()

        if self.model.task == "masked_lm":
            input = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = self(input, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            outputs = outputs.logits.squeeze(dim=1)
            outputs = outputs.to(torch.float32)
        else:
            input, labels = batch
            outputs = self(input)
            # No squeezing, leave logits as is for CrossEntropyLoss
            if self.model.task == "classification" and self.hparams.no_classes > 1:
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                labels = torch.nn.functional.one_hot(
                    labels.long(), num_classes=self.hparams.no_classes
                )
                labels = labels.float()
            elif (
                self.model.task == "token_classification"
                and self.hparams.no_classes > 1
            ):
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                outputs = outputs.permute(0, 2, 1)
                # Convert labels to long
                labels = labels.long()
            elif self.model.task == "multilabel_classification":
                if self.hparams.no_classes < 2:
                    if hasattr(outputs, "logits"):
                        outputs = outputs.logits
            else:
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits.squeeze(dim=1)
                else:
                    outputs = outputs.squeeze(dim=1)
            loss = self.loss_function(outputs, labels)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )

        if self.model.task == "classification" and self.hparams.no_classes > 1:
            labels = torch.argmax(labels, dim=1)
            outputs = torch.argmax(outputs, dim=1)
        if self.model.task == "token_classification" and self.hparams.no_classes > 1:
            # Get the maxium value of the 3rd dimension
            outputs = torch.argmax(outputs, dim=1)
        if (
            self.model.task == "multilabel_classification"
            and self.hparams.no_classes == 1
        ):
            # Logits loss function must be being used so we have to convert to probabilities
            outputs = torch.sigmoid(outputs)
            labels = labels.int()
        self.val_metric.update(outputs, labels)
        self.log(
            f"val_{self.metric_label}_step",
            self.val_metric,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )

        if self.log_interval != -1 and batch_idx % self.log_interval == 0:
            self.plmfit_logger.log(
                f"(val) batch : {batch_idx + 1}  / {len(self.trainer.val_dataloaders)} | running_loss : {loss} (batch time : {time.time() - batch_start_time:.4f})"
            )
        return loss

    def on_validation_epoch_end(self):
        self.log(f"val_{self.metric_label}_epoch", self.val_metric, sync_dist=True)
        if not self.trainer.sanity_checking:
            self.epoch_val_loss.append(
                self.trainer.logged_metrics["val_loss_epoch"].item()
            )
        if self.plmfit_logger:
            self.plmfit_logger.log(
                f'(val) loss: {self.trainer.logged_metrics["val_loss_epoch"]:.4f}'
            )
            self.plmfit_logger.log(
                f'(val) {self.metric_label}: {self.trainer.logged_metrics[f"val_{self.metric_label}_epoch"]:.4f}'
            )

        if (
            self.trainer.logged_metrics["val_loss_epoch"] < self.best_val_loss
            and self.current_epoch >= self.track_validation_after
            or self.track_validation_after == -1
        ):
            self.best_val_loss = self.trainer.logged_metrics["val_loss_epoch"]
            self.trainer.save_checkpoint(
                f"{self.plmfit_logger.base_dir}/lightning_logs/best_model.ckpt"
            )
            self.best_epoch = self.current_epoch
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        self.plmfit_logger.log(
            f"The best model was last saved at epoch {self.best_epoch + 1}."
        )

    ### TESTING STEPS ###
    def on_test_start(self) -> None:
        self.epoch_start_time = time.time()
        self.plmfit_logger.log("\n\nTESTING")
        self.plmfit_logger.log("-" * 10)

    def test_step(self, batch, batch_idx):
        batch_start_time = time.time()
        if self.model.task == "masked_lm":
            input = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = self(input, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            outputs = outputs.logits.squeeze(dim=1)
            outputs = outputs.to(torch.float32)
        else:
            input, labels, ids = batch
            outputs = self(input)

            # No squeezing, leave logits as is for CrossEntropyLoss
            if self.model.task == "classification" and self.hparams.no_classes > 1:
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                labels = torch.nn.functional.one_hot(
                    labels.long(), num_classes=self.hparams.no_classes
                )
                labels = labels.float()
            elif (
                self.model.task == "token_classification"
                and self.hparams.no_classes > 1
            ):
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                # swap 3rd dimension to 2nd dimension
                outputs = outputs.permute(0, 2, 1)
                # Convert labels to long
                labels = labels.long()
            elif self.model.task == "multilabel_classification":
                if self.hparams.no_classes < 2:
                    if hasattr(outputs, "logits"):
                        outputs = outputs.logits
            else:
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits.squeeze(dim=1)
                else:
                    outputs = outputs.squeeze(dim=1)
            loss = self.loss_function(outputs, labels)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=False
        )

        if self.model.task == "classification" and self.hparams.no_classes > 1:
            labels = torch.argmax(labels, dim=1)
            # outputs = torch.argmax(outputs, dim=1)
        if self.model.task == "token_classification" and self.hparams.no_classes > 1:
            # Get the maximum value of the 3rd dimension
            outputs = torch.argmax(outputs, dim=1)
        if (
            self.model.task == "multilabel_classification"
            and self.hparams.no_classes == 1
        ):
            # Logits loss function must be being used so we have to convert to probabilities
            outputs = torch.sigmoid(outputs)
            labels = labels.int()
        self.metrics.add(outputs, labels, ids)

        if self.log_interval != -1 and batch_idx % self.log_interval == 0:
            self.plmfit_logger.log(
                f"(test) batch : {batch_idx + 1}  / {len(self.trainer.test_dataloaders)} | running_loss : {loss} (batch time : {time.time() - batch_start_time:.4f})"
            )

        return loss

    def on_test_end(self) -> None:
        self.metrics.preds_list = self.merge_lists(self.metrics.preds_list)
        self.metrics.actual_list = self.merge_lists(self.metrics.actual_list)
        self.metrics.ids = self.merge_lists(self.metrics.ids)
        metrics = self.metrics.get_metrics(device=self.device)
        self.plmfit_logger.log(
            f'loss: {self.trainer.logged_metrics["test_loss_epoch"]:.4f} {time.time() - self.epoch_start_time:.4f}s'
        )
        for key, value in metrics["main"].items():
            self.plmfit_logger.log(f"{key}: {value}")
        if self.trainer.global_rank == 0:
            self.plmfit_logger.save_data(metrics["main"], "metrics")
            self.metrics.save_metrics(
                path=f"{self.plmfit_logger.base_dir}/{self.plmfit_logger.experiment_name}"
            )

    ### PREDICTION STEPS ###
    def on_predict_start(self) -> None:
        self.epoch_start_time = time.time()
        self.plmfit_logger.log("\nPREDICTING")
        self.plmfit_logger.log("-" * 10)

    def predict_step(self, batch, batch_idx):
        batch_start_time = time.time()
        (input,) = batch
        outputs = self(input)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        if self.log_interval != -1 and batch_idx % self.log_interval == 0:
            self.plmfit_logger.log(
                f"(predict) batch : {batch_idx + 1}  / {len(self.trainer.predict_dataloaders)} (batch time : {time.time() - batch_start_time:.4f})"
            )

        return outputs

    def on_predict_end(self) -> None:
        self.plmfit_logger.log(
            f"Prediction ended in {time.time() - self.epoch_start_time:.4f}s"
        )

    def configure_optimizers(self):
        optimizer = self.initialize_optimizer(self.trainer.model.parameters())
        lr_scheduler = self.initialize_lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]

    def initialize_optimizer(self, parameters):
        if self.hparams.optimizer is None:
            return None
        if self.hparams.optimizer == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adam":
            # if strategy is deepspeed, use DeepSpeedCPUAdam instead of torch.optim.Adam
            if isinstance(self.trainer.strategy, DeepSpeedStrategy):
                return DeepSpeedCPUAdam(
                    parameters,
                    lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay,
                )
            else:
                return torch.optim.Adam(
                    parameters,
                    lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay,
                )
        elif self.hparams.optimizer == "rmsprop":
            return torch.optim.RMSprop(
                parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

    def initialize_lr_scheduler(self, optimizer):
        if optimizer is None:
            return None
        return torch.optim.lr_scheduler.ConstantLR(optimizer)

    def initialize_loss_function(self):
        if self.hparams.loss_f == "bce":
            return torch.nn.BCELoss()
        elif self.hparams.loss_f == "bce_logits":
            return torch.nn.BCEWithLogitsLoss()
        elif self.hparams.loss_f == "mse":
            return torch.nn.MSELoss()
        elif (
            self.hparams.loss_f == "cross_entropy"
        ):  # Add cross-entropy loss for multiclass
            return torch.nn.CrossEntropyLoss(ignore_index=-100)
        elif self.hparams.loss_f == "masked_bce_logits":
            return MaskedBCEWithLogitsLoss(
                ignore_index=-100,
                pos_weight=(
                    self.hparams.pos_weight
                    if self.handle_hparam_exists("pos_weight")
                    else None
                )
            )
        elif self.hparams.loss_f == "masked_focal_logits":
            return MaskedFocalWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.hparams.loss_f}")

    def handle_bool_float_config_param(self, config_param, false_value=0, true_value=1):
        if isinstance(config_param, bool):
            if config_param:
                return true_value
            else:
                return false_value
        elif isinstance(config_param, (int, float)):
            return config_param  # Use the specified numeric value
        else:
            raise ValueError(
                "Invalid configuration. Expected boolean or numeric value."
            )

    def handle_hparam_exists(self, hparam_name):
        return hparam_name in self.hparams and self.hparams[hparam_name] is not None

    def early_stopping(self, patience=None):
        if patience is None:
            patience = self.handle_bool_float_config_param(
                self.hparams.early_stopping, false_value=-1, true_value=10
            )
        if patience == -1:
            return None
        return EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=patience,
            verbose=True,
            mode="min",
        )

    def gradient_accumulation_steps(self):
        return self.handle_bool_float_config_param(
            self.hparams.gradient_accumulation, false_value=1, true_value=8
        )

    def gradient_clipping(self):
        return self.handle_bool_float_config_param(
            self.hparams.gradient_clipping, false_value=0, true_value=0.5
        )

    def epoch_sizing(self):
        return self.handle_bool_float_config_param(
            self.hparams.epoch_sizing, false_value=1.0, true_value=0.2
        )

    def merge_lists(self, lists):
        if dist.is_initialized():
            all_rank_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_rank_list, lists)
            lists = [x for y in all_rank_list for x in y]
        return lists


class Metrics(torch.nn.Module):
    def __init__(self, task: str, no_classes=1, no_labels=1):
        super().__init__()
        self.task = task
        self.preds_list = []
        self.actual_list = []
        self.ids = []
        if task == "classification":
            self.no_classes = no_classes
            if self.no_classes < 2:
                self.acc = BinaryAccuracy()
                self.roc_auc = BinaryAUROC()
                self.mcc = BinaryMatthewsCorrCoef()
                self.cm = BinaryConfusionMatrix()
                self.roc = BinaryROC()
            else:
                self.acc = MulticlassAccuracy(num_classes=self.no_classes)
                self.micro_acc = MulticlassAccuracy(
                    num_classes=self.no_classes, average="micro"
                )
                self.mcc = MulticlassMatthewsCorrCoef(num_classes=self.no_classes)
                self.cm = MulticlassConfusionMatrix(num_classes=self.no_classes)
        elif task == "regression":
            self.mse = MeanSquaredError()
            self.rmse = MeanSquaredError(squared=False)
            self.mae = MeanAbsoluteError()
            self.r2 = R2Score()
            self.spearman = SpearmanCorrCoef()
        elif task == "masked_lm":
            self.perplexity = Perplexity(ignore_index=-100)
        elif task == "token_classification":
            self.no_classes = no_classes
            self.acc = MulticlassAccuracy(
                num_classes=self.no_classes, ignore_index=-100
            )
            self.micro_acc = MulticlassAccuracy(
                num_classes=self.no_classes, average="micro", ignore_index=-100
            )
            self.mcc = MulticlassMatthewsCorrCoef(
                num_classes=self.no_classes, ignore_index=-100
            )
            self.cm = MulticlassConfusionMatrix(
                num_classes=self.no_classes, ignore_index=-100
            )
        elif task == "multilabel_classification":
            self.no_labels = no_labels
            self.acc = MultilabelAccuracy(num_labels=self.no_labels, ignore_index=-100)
            self.mcc = MultilabelMatthewsCorrCoef(
                num_labels=self.no_labels, ignore_index=-100
            )
            self.cm = MultilabelConfusionMatrix(
                num_labels=self.no_labels, ignore_index=-100
            )

    def add(self, preds, actual, ids):
        if (
            self.task == "token_classification"
            or self.task == "multilabel_classification"
        ):
            self.preds_list.extend(preds.tolist())
            self.actual_list.extend(actual.tolist())
            (
                self.ids.extend(ids.tolist())
                if len(ids.tolist()) > 1
                else self.ids.append(ids.item())
            )
        else:
            (
                self.preds_list.extend(preds.tolist())
                if len(preds.tolist()) > 1
                else self.preds_list.append(preds.item())
            )
            (
                self.actual_list.extend(actual.tolist())
                if len(actual.tolist()) > 1
                else self.actual_list.append(actual.item())
            )
            (
                self.ids.extend(ids.tolist())
                if len(ids.tolist()) > 1
                else self.ids.append(ids.item())
            )

    def calculate(self, preds, actual):
        if self.task == "classification":
            self.calc_classification_metrics(preds, actual)
        elif self.task == "regression":
            self.calc_regression_metrics(preds, actual)
        elif self.task == "masked_lm":
            self.calc_masked_lm_metrics(preds, actual)
        elif self.task == "token_classification":
            self.calc_token_classification_metrics(preds, actual)
        elif self.task == "multilabel_classification":
            self.calc_multilabel_classification_metrics(preds, actual)

    def calc_classification_metrics(self, preds, actual):
        self.acc.update(preds, actual)
        if self.no_classes < 2:
            self.roc_auc.update(preds, actual)
        else:
            self.micro_acc.update(preds, actual)
        self.mcc.update(preds, actual)
        self.cm.update(preds, actual)
        if self.no_classes < 2:
            self.roc.update(preds, actual.int())

    def calc_regression_metrics(self, preds, actual):
        self.mse.update(preds, actual)
        self.rmse.update(preds, actual)
        self.mae.update(preds, actual)
        self.r2.update(preds, actual)
        self.spearman.update(preds, actual)

    def calc_masked_lm_metrics(self, preds, actual):
        self.perplexity.update(preds, actual)

    def calc_token_classification_metrics(self, preds, actual):
        self.acc.update(preds, actual)
        self.micro_acc.update(preds, actual)
        self.mcc.update(preds, actual)
        self.cm.update(preds, actual)

    def calc_multilabel_classification_metrics(self, preds, actual):
        self.acc.update(preds, actual)
        self.mcc.update(preds, actual)
        self.cm.update(preds, actual)

    def get_metrics(self, device="cpu"):
        self.calculate(
            torch.tensor(self.preds_list, device=device),
            torch.tensor(self.actual_list, device=device),
        )
        if self.task == "classification":
            return self.get_classification_metrics()
        elif self.task == "regression":
            return self.get_regression_metrics()
        elif self.task == "masked_lm":
            return self.get_masked_lm_metrics()
        elif self.task == "token_classification":
            return self.get_token_classification_metrics()
        elif self.task == "multilabel_classification":
            return self.get_multilabel_classification_metrics()

    def get_classification_metrics(self):
        if self.no_classes < 2:
            fpr, tpr, thresholds = list(self.roc.compute())
            self.report = {
                "main": {
                    "accuracy": self.acc.compute().item(),
                    "roc_auc": self.roc_auc.compute().item(),
                    "mcc": self.mcc.compute().item(),
                    "confusion_matrix": self.cm.compute().tolist(),
                },
                "roc_auc_data": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "roc_auc_val": self.roc_auc.compute().item(),
                },
                "pred_data": {
                    "preds": self.preds_list,
                    "actual": self.actual_list,
                    "ids": self.ids,
                },
            }
        else:
            self.report = {
                "main": {
                    "accuracy": self.acc.compute().item(),
                    "micro_accuracy": self.micro_acc.compute().item(),
                    "mcc": self.mcc.compute().item(),
                    "confusion_matrix": self.cm.compute().tolist(),
                },
                "pred_data": {
                    "preds": self.preds_list,
                    "actual": self.actual_list,
                    "ids": self.ids,
                },
            }
        return self.report

    def get_regression_metrics(self):
        metrics = {
            "mse": self.mse.compute().item(),
            "rmse": self.rmse.compute().item(),
            "mae": self.mae.compute().item(),
            "r_sq": self.r2.compute().item(),
            "spearman": self.spearman.compute().item(),
        }

        self.report = {
            "main": metrics,
            "pred_data": {
                "preds": self.preds_list,
                "actual": self.actual_list,
                "ids": self.ids,
                "eval_metrics": metrics,
            },
        }

        return self.report

    def get_masked_lm_metrics(self):
        metrics = {
            "perplexity": self.perplexity.compute().item(),
        }

        self.report = {"main": metrics}

        return self.report

    def get_token_classification_metrics(self):
        self.report = {
            "main": {
                "accuracy": self.acc.compute().item(),
                "micro_accuracy": self.micro_acc.compute().item(),
                "mcc": self.mcc.compute().item(),
                "confusion_matrix": self.cm.compute().tolist(),
            },
            "pred_data": {
                "preds": self.preds_list,
                "actual": self.actual_list,
                "ids": self.ids,
            },
        }
        return self.report

    def get_multilabel_classification_metrics(self):
        self.report = {
            "main": {
                "accuracy": self.acc.compute().item(),
                "mcc": self.mcc.compute().item(),
                "confusion_matrix": self.cm.compute().tolist(),
            },
            "pred_data": {
                "preds": self.preds_list,
                "actual": self.actual_list,
                "ids": self.ids,
            },
        }
        return self.report

    def save_metrics(self, path):
        metrics_path = f"{path}_metrics.json"
        if self.report is None:
            self.get_metrics()

        # Check if the metrics file already exists
        if os.path.exists(metrics_path):
            # Load the existing data
            with open(metrics_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            # Check if 'pred_data' field exists and update it
            if "pred_data" in existing_data:
                existing_data["pred_data"]["preds"].extend(
                    self.report["pred_data"]["preds"]
                )
                existing_data["pred_data"]["actual"].extend(
                    self.report["pred_data"]["actual"]
                )
                existing_data["pred_data"]["ids"].extend(
                    self.report["pred_data"]["ids"]
                )
                self.report = existing_data
            else:
                # If 'pred_data' does not exist, simply prepare to write the current report
                pass

        # Write the updated or original report to the file
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=4)


class PredictionWriter(BasePredictionWriter):

    def __init__(self, logger, write_interval, split_size=0, format="pt"):
        super().__init__(write_interval)
        self.output_dir = logger.base_dir
        self.file_name = logger.experiment_name
        self.logger = logger
        self.split_size = split_size
        self.format = format

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Make list into a single tensor
        predictions = torch.cat(predictions, dim=0)
        batch_indices = [
            item
            for sublist1 in batch_indices
            for sublist2 in sublist1
            for item in sublist2
        ]

        # Sort the predictions by the batch indices without doubling the memory usage
        sorted_predictions = torch.zeros_like(predictions)
        sorted_predictions[batch_indices] = predictions

        if self.split_size == 0:
            if self.format == "pt":
                torch.save(sorted_predictions, f"{self.output_dir}/{self.file_name}.pt")
            elif self.format == "csv":
                sorted_predictions = sorted_predictions.cpu().numpy()
                # Save the predictions to a CSV file
                # First row is index, second prediction with 4 decimal places

                # Create an array with indices
                indices = np.arange(sorted_predictions.shape[0]).reshape(-1, 1)
                # Concatenate indices and predictions
                predictions_with_indices = np.hstack((indices, sorted_predictions))

                # Format the predictions to 4 decimal places
                np.savetxt(
                    f"{self.output_dir}/{self.file_name}.csv",
                    predictions_with_indices,
                    delimiter=",",
                    fmt=["%d"] + ["%.4f"] * sorted_predictions.shape[1],
                    header="index,prediction",
                    comments="",
                )
        else:
            # Split the predictions into splits of size 'split_size' and the output file indicates the sample number in the batch (i.e. ..._1000-1999.pt)
            for i in range(0, len(sorted_predictions), self.split_size):
                split_size = (
                    self.split_size
                    if i + self.split_size < len(sorted_predictions)
                    else len(sorted_predictions) - i
                )
                chunk = sorted_predictions[
                    i : i + split_size
                ].clone()  # Use clone() to create a copy
                torch.save(
                    chunk,
                    f"{self.output_dir}/{self.file_name}_{i}-{i + split_size - 1}.pt",
                )

        self.logger.log(
            f"Predictions saved to {self.output_dir}/{self.file_name}.{self.format}"
        )
        self.logger.log(f"Predictions shape: {sorted_predictions.shape}")

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):

        torch.save(prediction, f"{self.output_dir}/{self.file_name}_{batch_idx}.pt")

        # Save a file called f"{self.output_dir}/{self.file_name}_{batch_idx}.json" to track the batch indices
        with open(
            f"{self.output_dir}/{self.file_name}_{batch_idx}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(batch_indices, f, indent=4)

        self.logger.log(f"Predictions saved to {self.output_dir}/{self.file_name}.pt")
        self.logger.log(f"Predictions shape: {prediction.shape}")
