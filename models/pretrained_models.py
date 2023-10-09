from language_models.progen2.models.progen.modeling_progen import ProGenForCausalLM
import utils
import torch.nn as nn
import logger as l
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import time

class PretrainedProteinLanguageModel():

    py_model = None
       
    tokenizer = None
    name = ""
    fine_tuned = 'none'
    data_type = 'none'
    head = 'none'
    no_parameters = 0
    no_layers = 0
    
    def __init__(self, name, py_model , tokenizer):
        self.name = name
        self.py_model = py_model
        self.tokenizer = tokenizer

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        
    def get_py_model(self):
        return self.py_model

    def set_py_model(self, py_model):
        self.py_model = py_model
                
    def get_tokenizer(self):
        return self.tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def fine_tune(self, data_type , method):
        #set all parameters require grad
        # create all the necessary parameters
        #do the training loop
        pass
    
    def evaluate(self, data_type ):
        pass
        
    def concat_task_specific_head(self):
        pass
    
    def extract_embeddings(self, layer):
        pass
    
    
        
class ProgenSmall151M(PretrainedProteinLanguageModel):
    def __init__(self):
        PretrainedProteinLanguageModel.__init__(self, 'progen2-small', ProGenForCausalLM.from_pretrained('./language_models/progen2/checkpoints/progen2-small') , utils.load_tokenizer('progen2-small'))
        
        
    def concat_task_specific_head(self , head):
    
       # assert head.in_.in_features == embs.shape[1], f'Embeddings dimension ({embs.shape[1]}) is not compatible with the input size of the task specific head ({head.in_.in_features}) . Change "input_len" to {embs.shape[1]} in config file : {args.head_config}'
     
        self.py_model = nn.Sequential(
          self.py_model,
          head
         )
        self.head = head.__class__.__name__ ## parse name from variable
        return 
        
    def extract_embeddings(self , data_type , layer = 11, reduction = 'mean'):
        logger = l.Logger(f'extract_embeddings_{self.name}_{self.head}_{data_type}_layer:{layer}_reduction:{reduction}.txt')
        logger.log('Extracting embeddings...')
        pass
    
    def fine_tune(self, data_type , method , epochs , lr , optimizer , batch_size , train_split_name , val_split , loss_f , lr_scheduler = None , log_interval = 1000 ):
        
        logger = l.Logger(f'fine_tune_{self.name}_{self.head}_{method}_{data_type}.txt')
        
        data = utils.load_dataset(data_type)   
        
        logger.log(f' Encoding {data.shape[0]} sequences....')
        encs = utils.categorical_encode(data['aa_seq'].values, self.tokenizer , max(data['len'].values))   
        data_train = data[data[train_split_name] == 'train']
        data_test = data[data[train_split_name] == 'test']
        encs_train = encs[data_train.index]
        encs_test = encs[data_test.index]
        train_dataset = data_utils.TensorDataset( encs_train , torch.tensor(data_train['score'].values))  
        n_val_samples = int(val_split * len(train_dataset))
        n_train_samples = len(train_dataset) - n_val_samples 
        train_set, val_set = torch.utils.data.random_split(train_dataset , [n_train_samples, n_val_samples]) 
        test_dataset = data_utils.TensorDataset( encs_test  , torch.tensor(data_test['score'].values))     
        
        train_dataloader = DataLoader(train_set, batch_size = batch_size , shuffle=True)
        valid_dataloader = DataLoader(val_set, batch_size = batch_size , shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.py_model.parameters(), lr=lr , betas=(0.9, 0.95))
        else:
            raise 'Unsupported optimizer'
            
        if  loss_f == 'mse':
            loss_f = nn.MSELoss(reduction = 'mean')
        else:
            raise 'Unsupported criterion'
            
        
        if method == 'full_retrain':
            utils.set_trainable_parameters(self.py_model)
            utils.get_parameters(self.py_model , True)
        else:
            raise 'Unsupported fine tuning method'
            
        epoch_train_loss = []
        epoch_val_loss = []
        
        trainig_start_time = time.time()
        
        for epoch in range(epochs):
            
            epoch_start_time = time.time()
            logger.log('Epoch {}/{}'.format(epoch + 1, epochs))
            logger.log('-' * 10)
            for phase in ['train', 'val']:
                
                if phase == 'train':
                    self.py_model.train()  # Set model to training mode
                    dataloader = train_dataloader
                else:
                    self.py_model.eval()   # Set model to evaluate mode
                    dataloader = valid_dataloader
                    
                batch_loss = 0
                    
                for itr , trainig_data in enumerate(dataloader , 0):
                    optimizer.zero_grad()
                    training_batch , training_labels = trainig_data
                    logger.log(f' {training_batch[0]=}')
                    logger.log(f' {training_labels=}')
                    outputs = self.py_model(training_batch)  
                    loss = loss_f(torch.squeeze(outputs).float(), training_labels.float())  
                    if phase == 'train':
                           loss.backward()
                           optimizer.step()
                    batch_loss += loss.item()       
                    
                    if itr%log_interval == 0:
                        logger.log(f'({phase}) minibatch :{itr + 1}  / {len(dataloader)} | running_loss : {batch_loss / (itr + 1)}')
                
                epoch_loss = batch_loss /itr
                logger.log('({}) Loss: {:.4f}'.format(phase, epoch_loss))
        
            


    
