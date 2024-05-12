from abc import ABC, abstractmethod
import plmfit.shared_utils.utils as utils
import torch
import time
import numpy as np
import plmfit.shared_utils.data_explore as data_explore
from sklearn.metrics import accuracy_score, mean_squared_error
from plmfit.models.peft import get_peft_model
from peft import LoraConfig
from plmfit.models.peft.tuners.bottleneck_adapters import BottleneckConfig
import psutil
import os
import json
import plmfit.shared_utils.custom_loss_functions as custom_loss_functions

class FineTuner(ABC):
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
        self.epoch_sizing = self.handle_bool_float_config_param(training_config['epoch_sizing'], false_value=0, true_value=0.2)
        self.model_output = training_config.get('model_output', 'default')
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
        
    def initialize_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    def initialize_loss_function(self, class_weights = None):
        """
        Initializes the loss function. Can be overridden by subclasses to use different loss functions.
        """
        if self.loss_function == 'bce':
            return torch.nn.BCELoss()
        elif self.loss_function == 'bce_logits':
            return torch.nn.BCEWithLogitsLoss()
        elif self.loss_function == 'mse':
            return torch.nn.MSELoss()
        elif self.loss_function == "weighted_bce":
            return custom_loss_functions.WeightedBCELoss(class_weights)
        else:
            raise ValueError(f"Loss Function '{self.loss_function}' not supported.")


class FullRetrainFineTuner(FineTuner):
    def __init__(self, training_config, logger = None):
        super().__init__(training_config, logger)

    def set_trainable_parameters(self, model):
        utils.set_trainable_parameters(model.py_model)
        utils.get_parameters(model.py_model, True)
        utils.get_parameters(model.head, True)

    def train(self, model, dataloaders_dict, log_interval = -1, on_ray_tuning=False):
        if on_ray_tuning: self.logger.mute = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        memory_usage = psutil.virtual_memory()
        max_mem_usage = utils.print_gpu_utilization(memory_usage, device)
        self.task_type = model.task
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
        class_weights = None
        if self.loss_function == "weighted_bce":
            train_labels = dataloaders_dict["train"].dataset.tensors[1]
            class_weights = utils.get_loss_weights(train_labels)
            self.logger.log(f'{class_weights}')

        loss_function = self.initialize_loss_function(class_weights)

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
                with torch.set_grad_enabled(phase == 'train'):
                    for itr, training_data in enumerate(dataloaders_dict[phase], 0):
                        batch_start_time = time.time()
                        optimizer.zero_grad()
                        input, labels = training_data
                        input = input.to(device)
                        labels = labels.to(device)
                        if self.model_output == 'default':
                            outputs = model(input).squeeze(dim=1)
                        elif self.model_output == 'logits':
                            outputs = model(input).logits.squeeze(dim=1)
                        else:
                            raise f'Model output "{self.model_output}" not defined'
                        loss = loss_function(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        batch_loss += loss.item()
                        
                        if self.task_type == 'classification':
                            preds = torch.round(outputs)
                        elif self.task_type == 'regression':
                            preds = outputs.squeeze()
                        
                        all_preds.extend(np.atleast_1d(preds.detach().cpu().numpy()))
                        all_labels.extend(np.atleast_1d(labels.detach().cpu().numpy()))

                        if log_interval != -1 and itr % log_interval == 0:
                            self.logger.log(
                                f'({phase}) batch : {itr + 1}  / {len(dataloaders_dict[phase])} | running_loss : {batch_loss / (itr + 1)} (batch time : {time.time() - batch_start_time:.4f})')

                mem_usage = utils.print_gpu_utilization(memory_usage, device)
                if mem_usage > max_mem_usage: max_mem_usage = mem_usage

                epoch_loss = batch_loss / (itr + 1)

                self.logger.log('({}) Loss: {:.4f} {:.4f}s'.format(
                    phase, epoch_loss, time.time() - epoch_start_time))
                if self.task_type == 'classification':
                    epoch_metric = accuracy_score(all_labels, all_preds)
                    self.logger.log(f'{phase} Accuracy: {epoch_metric:.4f}')
                elif self.task_type == 'regression':
                    epoch_metric = np.sqrt(
                        mean_squared_error(all_labels, all_preds))
                    self.logger.log(f'{phase} RMSE: {epoch_metric:.4f}')

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
                self.logger.log('Early stopping triggered after {} epochs with no improvement'.format(self.early_stopping))
                break  # Break the loop if model hasn't improved for 'patience' epochs
        
        if on_ray_tuning: 
            self.logger.mute = False
            return best_val_loss

        # After the training loop, restore the best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
            self.logger.log(f'Restored model to epoch {best_epoch+1} with best validation loss.')

        
        total_time = time.time() - start_time
        self.logger.log(f'Mean time per epoch: {total_time/(itr+1):.4f}s')
        self.logger.log(f'Total training time: {total_time:.1f}s')
        
        # After training, generate and save a plot of the training and validation loss
        loss_data = {
            "epoch_train_loss": epoch_train_loss,
            "epoch_val_loss": epoch_val_loss
        }
        with open(f'{self.logger.base_dir}/{self.logger.experiment_name}_loss.json', 'w', encoding='utf-8') as f:
            json.dump(loss_data, f, indent=4)
        
        if not on_ray_tuning:
            loss_plot = data_explore.create_loss_plot(epoch_train_loss, epoch_val_loss)
            self.logger.save_plot(loss_plot, "training_validation_loss")

        self.logger.save_model(model, self.task_type)
        self.logger.log(f'Saved best model at epoch {best_epoch+1} with validation loss {best_val_loss:.4f}')
        file_size_bytes = os.path.getsize(f'{self.logger.base_dir}/{self.logger.experiment_name}.pt')
        file_size_mb = file_size_bytes / (1024 * 1024) # Convert bytes to megabytes
        report = {
            "training_time": f'{total_time:.1f}',
            "avg_time_per_epoch": f'{total_time/(itr+1):.4f}',
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "model_file_size_mb": f'{file_size_mb:.2f}',
            "max_vram_usage_mb": max_mem_usage
        }
        self.logger.save_data(report, 'report')

        if not on_ray_tuning:
            if self.task_type == 'classification':
                metrics, roc_auc_fig, cm_fig, roc_auc_data = data_explore.evaluate_classification(model, dataloaders_dict, device, model_output=self.model_output)
                self.logger.save_data(metrics, 'metrics')
                self.logger.save_plot(roc_auc_fig, 'roc_curve')
                self.logger.save_plot(cm_fig, 'confusion_matrix')
                with open(f'{self.logger.base_dir}/{self.logger.experiment_name}_roc_auc.json', 'w', encoding='utf-8') as f:
                    json.dump(roc_auc_data, f, indent=4)
            elif self.task_type == 'regression':
                metrics, actual_vs_pred_fig, testing_data = data_explore.evaluate_regression(model, dataloaders_dict, device, model_output=self.model_output)
                self.logger.save_data(metrics, 'metrics')
                self.logger.save_plot(actual_vs_pred_fig, 'actual_vs_predicted')
                with open(f'{self.logger.base_dir}/{self.logger.experiment_name}_pred_vs_true.json', 'w', encoding='utf-8') as f:
                    json.dump(testing_data, f, indent=4)
        
    def prepare_model(self, model, target_layers="all"):
        # TODO: Train only last layer
        # if target_layers == "last":
        #     layers_to_train = model.layer_to_use
        # else:
        #     layers_to_train = None # Which will equal to all

        # Trim for test purposes 
        if model.experimenting: model.py_model.trim_model(0)
        
        utils.set_trainable_parameters(model.py_model)
        utils.set_modules_to_train_mode(model.py_model)
        utils.get_parameters(model.py_model, True)
        return model

class LowRankAdaptationFineTuner(FineTuner):
    def __init__(self, training_config, logger = None):
        super().__init__(training_config, logger)
        peft_config = utils.load_config('lora_config.json')
        self.logger.save_data(peft_config, 'lora_config')
            
        self.peft_config = LoraConfig(
            r = peft_config['r'],
            lora_alpha = peft_config['lora_alpha'],
            lora_dropout= peft_config['lora_dropout'],
            modules_to_save = peft_config['modules_to_save'],
            bias = peft_config['bias']
        )

    def prepare_model(self, model, target_layers="all"):
        if target_layers == "last":
            layers_to_train = model.layer_to_use
        else:
            layers_to_train = None # Which will equal to all
        utils.disable_dropout(model.py_model)
        self.peft_config.layers_to_transform = layers_to_train
        model.py_model = get_peft_model(model.py_model, self.peft_config)
        model.py_model.print_trainable_parameters()

        utils.set_modules_to_train_mode(model.py_model, self.peft_config.peft_type.lower())
        return model

class BottleneckAdaptersFineTuner(FineTuner):
    def __init__(self, training_config, logger = None):
        super().__init__(training_config, logger)
        peft_config = utils.load_config('bottleneck_config.json')
        self.logger.save_data(peft_config, 'bottleneck_config')
            
        self.peft_config = BottleneckConfig(
            bottleneck_size = peft_config['bottleneck_size'],
            non_linearity = peft_config['non_linearity'],
            adapter_dropout = peft_config['adapter_dropout'],
            scaling = peft_config['scaling'],
            modules_to_save = peft_config['modules_to_save'],
        )

    def prepare_model(self, model, target_layers="all"):
        # TODO: Choose layers to use adapter only, currently does all
        # if target_layers == "last":
        #     layers_to_train = model.layer_to_use
        # else:
        #     layers_to_train = None # Which will equal to all
        # self.peft_config.layers_to_transform = layers_to_train
        utils.disable_dropout(model.py_model)
        model.py_model = get_peft_model(model.py_model, self.peft_config)
        model.py_model.print_trainable_parameters()

        utils.set_modules_to_train_mode(model.py_model, self.peft_config.peft_type.lower())
        return model