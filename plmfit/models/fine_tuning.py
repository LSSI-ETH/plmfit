from abc import ABC, abstractmethod
import plmfit.shared_utils.utils as utils
import torch.nn as nn
import torch
import time
import numpy as np
import plmfit.shared_utils.data_explore as data_explore
from sklearn.metrics import accuracy_score, mean_squared_error
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import psutil
import plmfit.logger as l
import os
import json

class FineTuner(ABC):
    logger: l.Logger
    def __init__(self, training_config, logger = None):
        self.epochs = training_config['epochs']
        self.lr = training_config['learning_rate']
        self.weight_decay = self.handle_bool_float_config_param(training_config['weight_decay'], false_value=1, true_value=0.1)
        self.batch_size = training_config['batch_size']
        self.warmup_steps = training_config['warmup_steps']
        self.accumulation_steps = self.handle_bool_float_config_param(training_config['gradient_accumulation'], false_value=1, true_value=8)
        self.optimizer = training_config['optimizer']
        self.loss_function = training_config['loss_f']
        self.early_stopping = self.handle_bool_float_config_param(training_config['early_stopping'], false_value=-1, true_value=10)
        self.logger = logger

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


    @abstractmethod
    def set_trainable_parameters(self, model):
        """
        An abstract method to be implemented by subclasses for setting which parameters are trainable.
        This is particularly useful for fine-tuning specific layers of a model while freezing others.
        """
        pass

    @abstractmethod
    def train(self, model, dataloaders_dict, optimizer, loss_function, logger):
        """
        The core training logic to be implemented by subclasses. This method should handle the training
        and validation loop, including logging and any early stopping or checkpointing logic.

        Parameters:
        - model: The model to be fine-tuned.
        - dataloaders_dict: A dictionary containing 'train' and 'val' DataLoader objects.
        - optimizer: The optimizer to be used for training.
        - loss_function: The loss function to be used for training.
        - logger: A logger instance for logging training progress and metrics.
        """
        pass

    def initialize_optimizer(self, model_parameters):
        """
        Initializes the optimizer. Can be overridden by subclasses to use different optimizers.
        """
        if self.optimizer == 'sgd':
            return torch.optim.SGD(model_parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            return torch.optim.Adam(model_parameters, betas=(0.9, 0.99), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' not supported.")

    def initialize_loss_function(self):
        """
        Initializes the loss function. Can be overridden by subclasses to use different loss functions.
        """
        if self.loss_function == 'bce':
            return torch.nn.BCELoss()
        elif self.loss_function == 'bce_logits':
            return torch.nn.BCEWithLogitsLoss()
        elif self.loss_function == 'mse':
            return torch.nn.MSELoss()
        else:
            raise ValueError(f"Loss Function '{self.loss_function}' not supported.")


class FullRetrainFineTuner(FineTuner):
    def __init__(self, training_config, logger = None):
        super().__init__(training_config, logger)

    def set_trainable_parameters(self, model):
        utils.set_trainable_parameters(model.py_model)
        utils.get_parameters(model.py_model, True)
        utils.get_parameters(model.head, True)

    def train(self, model, dataloaders_dict, patience=10, log_interval = -1):
        memory_usage = psutil.virtual_memory()
        max_mem_usage = utils.print_gpu_utilization(memory_usage)
        self.task_type = model.task
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fp16 = False
        device_ids = list(range(torch.cuda.device_count()))

        if torch.cuda.is_available():
            self.logger.log(f'Available GPUs : {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                self.logger.log(
                    f'Running on {torch.cuda.get_device_properties(i).name}')
        else:
            self.logger.log(f'No GPU found, rolling device back to {device}')

        model = model.to(device)

        optimizer = self.initialize_optimizer(model.parameters())
        loss_function = self.initialize_loss_function()

        epoch_train_loss = []
        epoch_val_loss = []
        best_val_loss = float('inf')
        best_model_state = None  # To store the state of the best model
        best_epoch = 0
        epochs_no_improve = 0  # Counter for epochs with no improvement
        start_time = time.time()

        for epoch in range(self.epochs):

            epoch_start_time = time.time()
            self.logger.log('\nEpoch {}/{}'.format(epoch + 1, self.epochs))
            self.logger.log('-' * 10)
            
            # TODO torch_no_grad om val and autocast
            for phase in ['train', 'val']:
                if phase == 'train':
                    # Set model to training mode
                    model.train()
                else:
                    # Set model to evaluate mode
                    model.eval()

                batch_loss = 0
                all_preds = []
                all_labels = []

                for itr, training_data in enumerate(dataloaders_dict[phase], 0):
                    batch_start_time = time.time()
                    optimizer.zero_grad()
                    input, labels = training_data
                    input = input.to(device)
                    labels = labels.to(device)
                    outputs = model(input).squeeze()
                    loss = loss_function(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    batch_loss += loss.item()
                    mem_usage = utils.print_gpu_utilization(memory_usage)
                    if mem_usage > max_mem_usage: max_mem_usage = mem_usage
                    if self.task_type == 'classification':
                        preds = torch.round(outputs)
                    elif self.task_type == 'regression':
                        preds = outputs.squeeze()

                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

                    if log_interval != -1 and itr % log_interval == 0:
                        self.logger.log(
                            f'({phase}) batch : {itr + 1}  / {len(dataloaders_dict[phase])} | running_loss : {batch_loss / (itr + 1)} (batch time : {time.time() - batch_start_time:.4f})')

                epoch_loss = batch_loss / itr

                self.logger.log('({}) Loss: {:.4f} {:.4f}s'.format(
                    phase, epoch_loss, time.time() - epoch_start_time))
                if self.task_type == 'classification':
                    epoch_acc = accuracy_score(all_labels, all_preds)
                    self.logger.log(f'{phase} Accuracy: {epoch_acc:.4f}')
                elif self.task_type == 'regression':
                    epoch_rmse = np.sqrt(
                        mean_squared_error(all_labels, all_preds))
                    self.logger.log(f'{phase} RMSE: {epoch_rmse:.4f}')

                if phase == 'train':
                    epoch_train_loss.append(epoch_loss)
                else:
                    epoch_val_loss.append(epoch_loss)
                    # Early stopping check
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_state = model.state_dict()
                        best_epoch = epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

             # Check early stopping condition
            if self.early_stopping != -1 and epochs_no_improve >= self.early_stopping:
                self.logger.log('Early stopping triggered after {} epochs with no improvement'.format(patience))
                break  # Break the loop if model hasn't improved for 'patience' epochs

        # After the training loop, restore the best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
            self.logger.log(f'Restored model to epoch {best_epoch+1} with best validation loss.')

        
        total_time = time.time() - start_time
        self.logger.log(f'Mean time per epoch: {total_time/itr:.4f}s')
        self.logger.log(f'Total training time: {total_time:.1f}s')
        
        # After training, generate and save a plot of the training and validation loss
        loss_data = {
            "epoch_train_loss": epoch_train_loss,
            "epoch_val_loss": epoch_val_loss
        }
        with open(f'{self.logger.base_dir}/{self.logger.experiment_name}_loss.json', 'w', encoding='utf-8') as f:
            json.dump(loss_data, f, indent=4)

        loss_plot = data_explore.create_loss_plot(epoch_train_loss, epoch_val_loss)
        self.logger.save_plot(loss_plot, "training_validation_loss")
        self.logger.save_model(model, self.task_type)
        self.logger.log(f'Saved best model at epoch {best_epoch+1} with validation loss {best_val_loss:.4f}')
        file_size_bytes = os.path.getsize(f'{self.logger.base_dir}/{self.logger.experiment_name}.pt')
        file_size_mb = file_size_bytes / (1024 * 1024) # Convert bytes to megabytes
        report = {
            "training_time": f'{total_time:.1f}',
            "avg_time_per_epoch": f'{total_time/itr:.4f}',
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "model_file_size_mb": f'{file_size_mb:.2f}',
            "max_vram_usage_mb": max_mem_usage
        }
        self.logger.save_data(report, 'report')
        if self.task_type == 'classification':
            metrics, roc_auc_fig, cm_fig, roc_auc_data = data_explore.evaluate_classification(model, dataloaders_dict, device)
            self.logger.save_data(metrics, 'metrics')
            self.logger.save_plot(roc_auc_fig, 'roc_curve')
            self.logger.save_plot(cm_fig, 'confusion_matrix')
            with open(f'{self.logger.base_dir}/{self.logger.experiment_name}_roc_auc.json', 'w', encoding='utf-8') as f:
                json.dump(roc_auc_data, f, indent=4)
        elif self.task_type == 'regression':
            metrics, actual_vs_pred_fig, testing_data = data_explore.evaluate_regression(model, dataloaders_dict, device)
            self.logger.save_data(metrics, 'metrics')
            self.logger.save_plot(actual_vs_pred_fig, 'actual_vs_predicted')
            with open(f'{self.logger.base_dir}/{self.logger.experiment_name}_pred_vs_true.json', 'w', encoding='utf-8') as f:
                json.dump(testing_data, f, indent=4)
        


class LowRankAdaptationFineTuner(FineTuner):
    def __init__(self, epochs, lr, weight_decay, batch_size, val_split, log_interval, optimizer, loss_function, task_type='classification', rank=8, accumulation_steps = 16, epoch_size = 16384):
        super().__init__(epochs, lr, weight_decay, batch_size, val_split, optimizer, loss_function, log_interval, accumulation_steps, epoch_size)
        self.task_type = task_type  # New attribute to specify the task type
        self.rank = rank  # Rank for the low-rank adaptation
        self.peft_config = LoraConfig(task_type=TaskType.SEQ_CLS if task_type=='classification' else TaskType.FEATURE_EXTRACTION, inference_mode=False, r=rank, lora_alpha=32, lora_dropout=0.1, target_modules = ["qkv_proj"])

    def set_trainable_parameters(self, model):
        peft_model = get_peft_model(model, self.peft_config)
        peft_model.print_trainable_parameters()
        # utils.unset_trainable_parameters_after_layer(model)
        return peft_model

    def train(self, model, dataloaders_dict, logger, patience=10):
        print(model)
        utils.trainable_parameters_summary(model, logger)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fp16 = False
        device_ids = list(range(torch.cuda.device_count()))

        if torch.cuda.is_available():
            logger.log(f'Available GPUs : {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                logger.log(
                    f'Running on {torch.cuda.get_device_properties(i).name}')
        else:
            logger.log(f'No GPU found, rolling device back to {device}')

        model = model.to(device)

        optimizer = self.initialize_optimizer(model.parameters())
        loss_function = self.initialize_loss_function()

        epoch_train_loss = []
        epoch_val_loss = []
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_no_improve = 0  # Counter for epochs with no improvement

        memory_usage = psutil.virtual_memory()
        start_time = time.time()
        for epoch in range(self.epochs):

            epoch_start_time = time.time()
            logger.log('\nEpoch {}/{}'.format(epoch + 1, self.epochs))
            logger.log('-' * 10)

            # TODO torch_no_grad om val and autocast
            for phase in ['train', 'val']:
                if phase == 'train':
                    # Set model to training mode
                    model.train()
                else:
                    # Set model to evaluate mode
                    model.eval()

                batch_loss = 0
                all_preds = []
                all_labels = []
                epoch_dataloader_dict = utils.get_epoch_dataloaders(dataloaders_dict)
                for itr, training_data in enumerate(epoch_dataloader_dict[phase], 0):
                    batch_start_time = time.time()
                    input, labels = training_data
                    input = input.to(device)
                    labels = labels.to(device)
                    outputs = model(input).squeeze()
                    loss = loss_function(outputs, labels)
                    batch_loss += loss.item()
                    loss = loss / self.accumulation_steps
                    if phase == 'train':
                        loss.backward()
                        if (itr + 1) % self.accumulation_steps == 0 or (itr + 1) == len(epoch_dataloader_dict[phase]):
                            optimizer.step()
                            optimizer.zero_grad()

                    if self.task_type == 'classification':
                        preds = torch.round(outputs)
                    elif self.task_type == 'regression':
                        preds = outputs.squeeze()

                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

                    if self.log_interval != -1 and itr % self.log_interval == 0:
                        logger.log(
                            f'({phase}) batch : {itr + 1}  / {len(epoch_dataloader_dict[phase])} | running_loss : {batch_loss / (itr + 1)} (batch time : {time.time() - batch_start_time:.4f})')
                        logger.log(f"GPU memory occupied: {utils.print_gpu_utilization(memory_usage)} MB.")

                epoch_loss = batch_loss / (itr + 1)

                logger.log('({}) Loss: {:.4f} {:.4f}s'.format(
                    phase, epoch_loss, time.time() - epoch_start_time))
                if self.task_type == 'classification':
                    epoch_acc = accuracy_score(all_labels, all_preds)
                    logger.log(f'{phase} Accuracy: {epoch_acc:.4f}')
                elif self.task_type == 'regression':
                    epoch_rmse = np.sqrt(
                        mean_squared_error(all_labels, all_preds))
                    logger.log(f'{phase} RMSE: {epoch_rmse:.4f}')

                if phase == 'train':
                    epoch_train_loss.append(epoch_loss)
                else:
                    epoch_val_loss.append(epoch_loss)
                    # Early stopping check
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_epoch = epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

             # Check early stopping condition
            if epochs_no_improve >= patience:
                logger.log('Early stopping triggered after {} epochs with no improvement'.format(patience))
                break  # Break the loop if model hasn't improved for 'patience' epochs
        
        logger.log(f'Mean time per epoch: {(time.time() - start_time)/itr:.4f}s')
        logger.log(f'Total training time: {(time.time() - start_time):.4f}s')
        # After training, generate and save a plot of the training and validation loss
        loss_plot = data_explore.create_loss_plot(epoch_train_loss, epoch_val_loss)
        logger.save_plot(loss_plot, "training_validation_loss")
        logger.save_model(model, self.task_type)
        if self.task_type == 'classification':
            metrics, roc_auc_fig, cm_fig = data_explore.evaluate_classification(model, dataloaders_dict, device)
            logger.save_data(metrics, 'Metrics')
            logger.save_plot(roc_auc_fig, 'ROC_curve')
            logger.save_plot(cm_fig, 'confusion_matrix')
        elif self.task_type == 'regression':
            metrics, actual_vs_pred_fig = data_explore.evaluate_regression(model, dataloaders_dict, device)
            logger.save_data(metrics, 'Metrics')
            logger.save_plot(actual_vs_pred_fig, 'actual_vs_predicted')


class PEFTLora(FineTuner):
    def __init__(self, epochs, lr, weight_decay, batch_size, val_split, log_interval, optimizer, loss_function, task_type='classification', rank=8, accumulation_steps = 16, epoch_size = 16384):
        super().__init__(epochs, lr, weight_decay, batch_size, val_split, optimizer, loss_function, log_interval, accumulation_steps, epoch_size)
        self.task_type = task_type  # New attribute to specify the task type
        self.rank = rank  # Rank for the low-rank adaptation