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

class FineTuner(ABC):

    def __init__(self, epochs, lr, weight_decay, batch_size, val_split, optimizer, loss_function, log_interval):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_split = val_split
        self.log_interval = log_interval
        self.optimizer = optimizer
        self.loss_function = loss_function

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
        else:
            return torch.optim.Adam(model_parameters, betas=(0.9, 0.99), lr=self.lr, weight_decay=self.weight_decay)

    def initialize_loss_function(self):
        """
        Initializes the loss function. Can be overridden by subclasses to use different loss functions.
        """
        if self.loss_function == 'bce':
            return torch.nn.BCELoss()
        else:
            return torch.nn.MSELoss()


class FullRetrainFineTuner(FineTuner):
    def __init__(self, epochs, lr, weight_decay, batch_size, val_split, log_interval, optimizer, loss_function, task_type='classification'):
        super().__init__(epochs, lr, weight_decay, batch_size, val_split, optimizer, loss_function, log_interval)
        self.task_type = task_type  # New attribute to specify the task type

    def set_trainable_parameters(self, model):
        utils.set_trainable_parameters(model.py_model)
        utils.get_parameters(model.py_model, True)
        utils.get_parameters(model.head, True)

    def train(self, model, dataloaders_dict, logger, patience=10):
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

        for epoch in range(self.epochs):

            epoch_start_time = time.time()
            logger.log('\nEpoch {}/{}'.format(epoch + 1, self.epochs))
            logger.log('-' * 10)

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

                    if self.task_type == 'classification':
                        preds = torch.round(outputs)
                    elif self.task_type == 'regression':
                        preds = outputs.squeeze()

                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

                    if self.log_interval != -1 and itr % self.log_interval == 0:
                        logger.log(
                            f'({phase}) batch : {itr + 1}  / {len(dataloaders_dict[phase])} | running_loss : {batch_loss / (itr + 1)} (batch time : {time.time() - batch_start_time:.4f})')

                epoch_loss = batch_loss / itr

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


class LowRankAdaptationFineTuner(FineTuner):
    def __init__(self, epochs, lr, weight_decay, batch_size, val_split, log_interval, optimizer, loss_function, task_type='classification', rank=10):
        super().__init__(epochs, lr, weight_decay, batch_size, val_split, optimizer, loss_function, log_interval)
        self.task_type = task_type  # New attribute to specify the task type
        self.rank = rank  # Rank for the low-rank adaptation
        self.peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=rank, lora_alpha=32, lora_dropout=0.1, target_modules = ["qkv_proj", "out_proj"])

    def set_trainable_parameters(self, model):
        peft_model = get_peft_model(model.py_model, self.peft_config)
        peft_model.print_trainable_parameters()
        model.py_model = peft_model.base_model.model
        utils.set_trainable_parameters(model.head)

    def train(self, model, dataloaders_dict, logger, patience=10):
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

        for epoch in range(self.epochs):

            epoch_start_time = time.time()
            logger.log('\nEpoch {}/{}'.format(epoch + 1, self.epochs))
            logger.log('-' * 10)

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

                    if self.task_type == 'classification':
                        preds = torch.round(outputs)
                    elif self.task_type == 'regression':
                        preds = outputs.squeeze()

                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

                    if self.log_interval != -1 and itr % self.log_interval == 0:
                        logger.log(
                            f'({phase}) batch : {itr + 1}  / {len(dataloaders_dict[phase])} | running_loss : {batch_loss / (itr + 1)} (batch time : {time.time() - batch_start_time:.4f})')

                epoch_loss = batch_loss / itr

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
