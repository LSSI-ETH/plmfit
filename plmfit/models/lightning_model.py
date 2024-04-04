import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
from torchmetrics import classification, regression
import time
import json

class LightningModel(L.LightningModule):
    def __init__(self,  model, training_config, plmfit_logger = None, log_interval=-1):
        super().__init__()
        self.model = model
        self.save_hyperparameters(training_config)
        self.loss_function = self.initialize_loss_function()
        self.plmfit_logger = plmfit_logger
        self.log_interval = log_interval
        
        if self.model.task == 'classification':
            self.train_metric = classification.BinaryAccuracy()
            self.metric_label = 'accuracy'
        elif self.model.task == 'regression':
            self.train_metric = regression.MeanSquaredError(squared=False)
            self.metric_label = 'rmse'
        
        self.val_metric = self.train_metric.clone()

        self.metrics = Metrics(self.model.task)

    def forward(self, input):
        output = self.model(input)
        if hasattr(output, 'logits'):
            return output.logits
        else:
            return output
    
    def on_fit_start(self) -> None:
        self.start_time = time.time()
        self.epoch_train_loss = []
        self.epoch_val_loss = []

    def on_fit_end(self) -> None:
        total_time = time.time() - self.start_time
        self.plmfit_logger.log(f'\nMean time per epoch: {total_time/(self.current_epoch+1):.4f}s')
        self.plmfit_logger.log(f'Total training time: {total_time:.1f}s')
        loss_data = {
            "epoch_train_loss": self.epoch_train_loss,
            "epoch_val_loss": self.epoch_val_loss
        }
        with open(f'{self.plmfit_logger.base_dir}/{self.plmfit_logger.experiment_name}_loss.json', 'w', encoding='utf-8') as f:
            json.dump(loss_data, f, indent=4)
        report = {
            "training_time": f'{total_time:.1f}',
            "avg_time_per_epoch": f'{total_time/(self.current_epoch+1):.4f}'
        }
        self.plmfit_logger.save_data(report, 'report')

    ### TRAINING STEPS ###
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        self.plmfit_logger.log('\nEpoch {}/{}'.format(self.current_epoch + 1, self.trainer.max_epochs))
        self.plmfit_logger.log('-' * 10)

    def training_step(self, batch, batch_idx):
        batch_start_time = time.time()
        
        input, labels = batch
        outputs = self(input).squeeze(dim=1)

        loss = self.loss_function(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)

        self.train_metric.update(outputs, labels)
        self.log(f'train_{self.metric_label}_step', self.train_metric, on_step=False, on_epoch=True, logger=True, prog_bar=False)

        if self.log_interval != -1 and batch_idx % self.log_interval == 0:
            self.plmfit_logger.log(f'(train) batch : {batch_idx + 1}  / {len(self.trainer.train_dataloader)} | running_loss : {loss / (batch_idx + 1)} (batch time : {time.time() - batch_start_time:.4f})')

        return loss
    
    def on_train_epoch_end(self):
        self.log(f'train_{self.metric_label}_epoch', self.train_metric)
        self.epoch_train_loss.append(self.trainer.logged_metrics["train_loss_epoch"].item())
        if self.plmfit_logger: 
            self.plmfit_logger.log(f'(train) loss: {self.trainer.logged_metrics["train_loss_epoch"]:.4f} {time.time() - self.epoch_start_time:.4f}s')
            self.plmfit_logger.log(f'(train) {self.metric_label}: {self.trainer.logged_metrics[f"train_{self.metric_label}_epoch"]:.4f}')


    ### VALIDATION STEPS ###
    def on_validation_epoch_start(self):
        if self.trainer.sanity_checking:
            if self.plmfit_logger: self.plmfit_logger.log(f'Sanity checking...')
        self.epoch_start_time = time.time()

    def validation_step(self, batch, batch_idx):
        batch_start_time = time.time()

        input, labels = batch
        outputs = self(input).squeeze(dim=1)

        loss = self.loss_function(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)

        self.val_metric.update(outputs, labels)
        self.log(f'val_{self.metric_label}_step', self.val_metric, on_step=False, on_epoch=True, logger=True, prog_bar=False)

        if self.log_interval != -1 and batch_idx % self.log_interval == 0:
            self.plmfit_logger.log(f'(val) batch : {batch_idx + 1}  / {len(self.trainer.val_dataloaders)} | running_loss : {loss / (batch_idx + 1)} (batch time : {time.time() - batch_start_time:.4f})')
    
        return loss
    
    def on_validation_epoch_end(self):
        self.log(f'val_{self.metric_label}_epoch', self.val_metric)
        if not self.trainer.sanity_checking: self.epoch_val_loss.append(self.trainer.logged_metrics["val_loss_epoch"].item())
        if self.plmfit_logger: 
            self.plmfit_logger.log(f'(val) loss: {self.trainer.logged_metrics["val_loss_epoch"]:.4f} {time.time() - self.epoch_start_time:.4f}s')
            self.plmfit_logger.log(f'(val) {self.metric_label}: {self.trainer.logged_metrics[f"val_{self.metric_label}_epoch"]:.4f}')


    ### TESTING STEPS ###
    def on_test_start(self) -> None:
        self.epoch_start_time = time.time()
        self.plmfit_logger.log('\n\nTESTING')
        self.plmfit_logger.log('-' * 10)
    
    def test_step(self, batch, batch_idx):
        input, labels = batch
        outputs = self(input).squeeze(dim=1)

        loss = self.loss_function(outputs, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=False)

        self.metrics.calculate(outputs, labels)
        
        return loss
    
    def on_test_end(self) -> None:
        metrics = self.metrics.get_metrics()
        if self.plmfit_logger: 
            self.plmfit_logger.log(f'loss: {self.trainer.logged_metrics["test_loss"]:.4f} {time.time() - self.epoch_start_time:.4f}s')
            for key, value in metrics['main'].items():
                self.plmfit_logger.log(f'{key}: {value}')
            self.plmfit_logger.save_data(metrics['main'], 'metrics')
        self.metrics.save_metrics(path=f'{self.plmfit_logger.base_dir}/{self.plmfit_logger.experiment_name}')


    
    def configure_optimizers(self):
        optimizer = self.initialize_optimizer(self.model.parameters())
        lr_scheduler = self.initialize_lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]
    
    def initialize_optimizer(self, parameters):
        if self.hparams.optimizer == 'sgd':
            return torch.optim.SGD(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'adam':
            return torch.optim.Adam(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")
        
    def initialize_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ConstantLR(optimizer)

    def initialize_loss_function(self):
        if self.hparams.loss_f== 'bce':
            return torch.nn.BCELoss()
        elif self.hparams.loss_f == 'bce_logits':
            return torch.nn.BCEWithLogitsLoss()
        elif self.hparams.loss_f == 'mse':
            return torch.nn.MSELoss()
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
            raise ValueError("Invalid configuration. Expected boolean or numeric value.")
        
    def early_stopping(self):
        patience = self.handle_bool_float_config_param(self.hparams.early_stopping, false_value=-1, true_value=10)
        if patience == -1: return None
        return EarlyStopping(monitor="val_loss", min_delta=0.00, patience=patience, verbose=False, mode="min")
    
    def gradient_accumulation_steps(self):
        return self.handle_bool_float_config_param(self.hparams.gradient_accumulation, false_value=1, true_value=8)
    
    def gradient_clipping(self):
        return self.handle_bool_float_config_param(self.hparams.gradient_clipping, false_value=0, true_value=0.5)
    
    def epoch_sizing(self):
        return self.handle_bool_float_config_param(self.hparams.epoch_sizing, false_value=1.0, true_value=0.2)
        

class Metrics(torch.nn.Module):
    def __init__(self, task: str):
        super().__init__()
        self.task = task
        self.preds_list = []
        self.actual_list = []
        if task == 'classification':
            self.acc = classification.BinaryAccuracy()
            self.roc_auc = classification.BinaryAUROC()
            self.mcc = classification.BinaryMatthewsCorrCoef()
            self.cm = classification.BinaryConfusionMatrix()
            self.roc = classification.BinaryROC()
        elif task == 'regression':
            self.mse = regression.MeanSquaredError()
            self.rmse = regression.MeanSquaredError(squared=False)
            self.mae = regression.MeanAbsoluteError()
            self.r2 = regression.R2Score()
            self.spearman = regression.SpearmanCorrCoef()

    def calculate(self, preds, actual):
        self.preds_list.extend(preds.tolist()) if len(preds.tolist()) > 1 else self.preds_list.append(preds.tolist())
        self.actual_list.extend(actual.tolist()) if len(actual.tolist()) > 1 else self.actual_list.append(actual.tolist())
        if self.task == 'classification':
            self.calc_classification_metrics(preds, actual)
        elif self.task == 'regression':
            self.calc_regression_metrics(preds, actual)

    def calc_classification_metrics(self, preds, actual):
        self.acc.update(preds, actual)
        self.roc_auc.update(preds, actual)
        self.mcc.update(preds, actual)
        self.cm.update(preds, actual)
        self.roc.update(preds, actual.int())

    def calc_regression_metrics(self, preds, actual):
        self.mse.update(preds, actual)
        self.rmse.update(preds, actual)
        self.mae.update(preds, actual)
        self.r2.update(preds, actual)
        self.spearman.update(preds, actual)

    def get_metrics(self):
        if self.task == 'classification':
            return self.get_classification_metrics()
        elif self.task == 'regression':
            return self.get_regression_metrics()

    def get_classification_metrics(self):
        fpr, tpr, thresholds = list(self.roc.compute())
        self.report = {
            'main': {
                'accuracy': self.acc.compute().item(),
                'roc_auc': self.roc_auc.compute().item(),
                'mcc': self.mcc.compute().item(),
                'confusion_matrix': self.cm.compute().tolist()
            },
            'roc_auc_data': {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "roc_auc_val": self.roc_auc.compute().item()
            }
        }
        return self.report
    
    def get_regression_metrics(self):
        metrics = {
                'mse': self.mse.compute().item(),
                'rmse': self.rmse.compute().item(),
                'mae': self.mae.compute().item(),
                'r_sq': self.r2.compute().item(),
                'spearman': self.spearman.compute().item()
            }
        
        self.report = {
            'main': metrics,
            'pred_data': {
                "preds": self.preds_list,
                "actual": self.actual_list,
                "eval_metrics": metrics
            }
        }

        return self.report
    
    def save_metrics(self, path):
        if self.report is None: self.get_metrics()
        with open(f'{path}_metrics.json', 'w', encoding='utf-8') as f:
                    json.dump(self.report, f, indent=4)