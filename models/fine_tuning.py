from abc import abstractmethod
import ctypes
import models.pretrained_models 
import utils
import torch.nn as nn
import torch
import time
import os

class Tuner():

    method = ''
    lr = 0.0006
    batch_size = 8
    val_split = 0.2 

    log_interval = 1
    epochs = 0
    
    def __init__(self, method, epochs, lr , batch_size,  val_split, log_interval):
        self.method = method
        self.lr = lr
        self.batch_size = batch_size
        self.val_split = val_split
        self.log_interval = log_interval
        self.epochs = epochs
                    
    @abstractmethod    
    def set_trainable_parameters(self, model):
        pass
    @abstractmethod
    def train(self, model , dataloader):
        pass

class FullRetrainTuner(Tuner):  
    def __init__(self, epochs,  lr , batch_size, val_split, log_interval):
        method = 'full_retrain'
        super().__init__(method, epochs, lr ,  batch_size, val_split,  log_interval)
        pass
        
    def set_trainable_parameters(self, model):
        utils.set_trainable_parameters(model.py_model)
        utils.get_parameters(model.py_model , True)
        utils.get_parameters(model.head , True)

    def train(self, model , dataloaders_dict, optimizer, loss_f , logger):      
        device = 'cpu'
        fp16 = False
        device_ids = []
        ##TODO: Run on multiple GPUs data and model parallelization
        if torch.cuda.is_available():
            device = "cuda:0"
            fp16 = True
            logger.log(f'Available GPUs : {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                logger.log(f' Running on {torch.cuda.get_device_properties(i).name}')
                device_ids.append(i)
        else:
            logger.log(f' No gpu found rolling device back to {device}')
            
        ##model.py_model = model.py_model.to(device)
       ## model.head = model.head.to(device)
        model = model.to(device)
        
        if optimizer == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr= self.lr , betas=(0.9, 0.99), weight_decay= 0.1)
            #torch.nn.utils.clip_grad_norm_(self.py_model.parameters(), 1 , norm_type=2.0, error_if_nonfinite=False)
        else:
            raise 'Unsupported optimizer'
            
        if  loss_f == 'mse':
            loss_f = nn.MSELoss(reduction = 'mean')
        else:
            raise 'Unsupported criterion'
            
        epoch_train_loss = []
        epoch_val_loss = []
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(self.epochs):
            
            epoch_start_time = time.time()
            logger.log('Epoch {}/{}'.format(epoch + 1, self.epochs))
            logger.log('-' * 10)
            
            for phase in ['train', 'val']:  
                if phase == 'train':
                #model.py_model.train()  # Set model to training mode
                #model.head.train()
                    model.train()
                else:
                    model.eval()
                #model.py_model.eval()   # Set model to evaluate mode
                #model.head.eval()
                    
                batch_loss = 0
                    
                for itr , trainig_data in enumerate(dataloaders_dict[phase] , 0):
                    batch_start_time = time.time()
                    optim.zero_grad()
                    training_batch , training_labels = trainig_data
                    training_batch = training_batch.to(device)
                    training_labels = training_labels.to(device)
                    outputs = model(training_batch)
                    loss = loss_f(torch.squeeze(outputs).float(), training_labels.float())  
                    if phase == 'train':
                           loss.backward()
                           optim.step()
                    batch_loss += loss.item()       
                    
                    if itr%self.log_interval == 0:
                        logger.log(f'({phase}) batch : {itr + 1}  / {len(dataloaders_dict[phase])} | running_loss : {batch_loss / (itr + 1)} (batch time : {time.time() - batch_start_time:.4f})')
                
                epoch_loss = batch_loss / itr
                if phase == 'train':
                    epoch_train_loss.append(epoch_loss)
                else:
                    epoch_val_loss.append(epoch_loss)
                
                logger.log('({}) Loss: {:.4f} {:.4f}s'.format(phase, epoch_loss, time.now() - epoch_start_time))
                
            ##TODO: Implement early stopping
            if epoch_val_loss[-1] < best_val_loss:
                best_val_loss = epoch_val_loss[-1] 
                torch.save(model.state_dict(), f'./models/saved_models/model:{model.name}_head:{model.head_name}_ft:{self.method}.pt')
                best_epoch  = epoch
          
