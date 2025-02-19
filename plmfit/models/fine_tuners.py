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
import plmfit.shared_utils.utils as utils

class FineTuner(ABC):
    def __init__(self, logger = None):
        self.logger = logger

    def set_trainable_parameters(self, model):
        pass

    def prepare_model(self, model, target_layers="all"):
        pass


class FullRetrainFineTuner(FineTuner):
    def __init__(self, logger = None):
        super().__init__(logger)

    def set_trainable_parameters(self, model):
        utils.set_trainable_parameters(model.py_model)
        utils.get_parameters(model.py_model, True)
        utils.get_parameters(model.head, True)
   
    def prepare_model(self, model, target_layers="all"):
        if target_layers == "last":
            layers_to_train = model.layer_to_use
        elif type(target_layers) == list:
            layers_to_train = target_layers
        else:
            layers_to_train = None # Which will equal to all

        # Trim for test purposes 
        if model.experimenting: model.py_model.trim_model(0)
        
        utils.set_trainable_parameters(model.py_model)
        utils.set_modules_to_train_mode(model.py_model)
        if layers_to_train is not None: 
            utils.freeze_parameters(model.py_model)
            utils.set_trainable_layers(model.py_model, [layers_to_train])
            utils.set_trainable_head(model.py_model)
            utils.set_head_to_train_mode(model.py_model)
        utils.get_parameters(model.py_model, True)
        # TODO: set head to trainable
        return model

class LowRankAdaptationFineTuner(FineTuner):
    def __init__(self, logger = None):
        super().__init__(logger)
        peft_config = utils.load_config('peft/lora_config.json')
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
    def __init__(self, logger = None):
        super().__init__(logger)
        peft_config = utils.load_config('peft/bottleneck_adapters_config.json')
        self.logger.save_data(peft_config, "bottleneck_adapters_config")

        self.peft_config = BottleneckConfig(
            bottleneck_size = peft_config['bottleneck_size'],
            non_linearity = peft_config['non_linearity'],
            adapter_dropout = peft_config['adapter_dropout'],
            scaling = peft_config['scaling'],
            modules_to_save = peft_config['modules_to_save'],
        )

    def prepare_model(self, model, target_layers="all"):
        if target_layers == "last":
            layers_to_train = model.layer_to_use
        else:
            layers_to_train = None # Which will equal to all
        self.peft_config.layers_to_transform = layers_to_train
        utils.disable_dropout(model.py_model)
        model.py_model = get_peft_model(model.py_model, self.peft_config)
        model.py_model.print_trainable_parameters()

        utils.set_modules_to_train_mode(model.py_model, self.peft_config.peft_type.lower())
        return model
