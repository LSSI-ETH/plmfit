from abc import abstractmethod
import ctypes
import models.pretrained_models 
import utils

class Tuner():

    method = ''
    lr = 0.0006
    optimizer = 'adam'
    batch_size = 8
    train_split_name = 'two_vs_many_split'
    val_split = 0.2 
    loss_f = 'mse'
    log_interval = 1
    epochs = 0
    
    def __init__(self, method, epochs, lr , optimizer, batch_size, train_split_name , val_split, loss_f, log_interval):
        print('init')
        self.method = method
        self.lr = lr
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.train_split_name = train_split_name
        self.val_split = val_split
        self.loss_f = loss_f
        self.log_interval = log_interval
        self.epochs = epochs
        
    @abstractmethod    
    def set_trainable_parameters(self, model):
        pass
    @abstractmethod
    def train(self, model , dataloader):
        pass

class FullRetrainTuner(Tuner):
    
    def __init__(self, epochs,  lr , optimizer, batch_size, train_split_name , val_split, loss_f, log_interval):
        method = 'full_retrain'
        super().__init__(method, epochs, lr , optimizer, batch_size, train_split_name , val_split, loss_f, log_interval)
        pass
        
    def set_trainable_parameters(self, model):
        utils.set_trainable_parameters(model.py_model)
        utils.get_parameters(model.py_model , True)
        utils.get_parameters(model.head , True)

    def train(self, model , dataloader):
        pass
