from language_models.progen2.models.progen.modeling_progen import ProGenForCausalLM
import utils
import torch.nn as nn
import logger as l
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import time
from abc import abstractmethod
from models.fine_tuning import *
from tokenizers import Tokenizer

class IPretrainedProteinLanguageModel(nn.Module):
    
    name : str
    py_model : nn.Module
    head : nn.Module
    head_name : str
    no_parameters : int
    emb_layers_dim : int
    output_dim : int
    
    
    def __init__(self):
        super().__init__()
        self.head = None
        self.head_name = 'none'
        pass
        
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
        
    @abstractmethod    
    def concat_task_specific_head(self , head):
        pass
    @abstractmethod
    def extract_embeddings(self , data_type , batch_size , layer = 11, reduction = 'mean'):
        pass
    @abstractmethod
    def fine_tune(self, data_type, fine_tuner , train_split_name, optimizer , loss_f ):
        pass 
    @abstractmethod
    def evaluate(self, data_type ):
        pass
    
    @abstractmethod
    def forward(self, src):
        pass

#TODO: infere based on aa_seq list
    @abstractmethod
    def infere(self, aa_seq_list):
        pass
        
    
    
  #Implement class for every supported Portein Language Model family  
        
class ProGenFamily(IPretrainedProteinLanguageModel): ##

    tokenizer : Tokenizer
    
    def __init__(self , progen_model_name : str):
        #IPretrainedProteinLanguageModel.__init__(self)
        super().__init__()
        self.name = progen_model_name 
        self.py_model = ProGenForCausalLM.from_pretrained(f'./language_models/progen2/checkpoints/{progen_model_name}')    
        self.no_parameters = utils.get_parameters(self.py_model)
        self.no_layers = len(self.py_model.transformer.h)
        self.output_dim = self.py_model.lm_head.out_features
        self.emb_layers_dim = self.py_model.transformer.h[0].attn.out_proj.out_features
        self.tokenizer = utils.load_tokenizer(progen_model_name)
      
    def concat_task_specific_head(self , head):  
        assert head.in_.in_features == self.output_dim, f' Head\'s input dimension ({head.in_.in_features}) is not compatible with {self.name}\'s output dimension ({self.output_dim}). To concat modules these must be equal.'
        #TODO: Add concat option with lm head or final transformer layer.
        self.head = head 
        self.head_name = head.__class__.__name__ ## parse name from variable
        self.no_parameters += utils.get_parameters(self.head)
        return 
        
    def extract_embeddings(self , data_type , batch_size , layer = 11, reduction = 'mean'):
        logger = l.Logger(f'logger_extract_embeddings_{data_type}_{self.name}_layer{layer}_{reduction}.txt')
        device = 'cpu'
        fp16 = False
        device_ids = []
        if torch.cuda.is_available():
            device = "cuda:0"
            fp16 = True
            logger.log(f'Available GPUs : {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                logger.log(f' Running on {torch.cuda.get_device_properties(i).name}')
                device_ids.append(i)

        else:
            logger.log(f' No gpu found rolling device back to {device}')
        data = utils.load_dataset(data_type)  
        start_enc_time = time.time()
        logger.log(f' Encoding {data.shape[0]} sequences....')
        encs = utils.categorical_encode(data['aa_seq'].values, self.tokenizer , max(data['len'].values))   
        logger.log(f' Encoding completed! {time.time() -  start_enc_time:.4f}s')
        encs = encs.to(device)
        seq_dataset = data_utils.TensorDataset(encs)
        seq_loader =  data_utils.DataLoader(seq_dataset, batch_size= batch_size, shuffle=False)
        logger.log(f'Extracting embeddings for {len(seq_dataset)} sequences...')
        
        embs = torch.zeros((len(seq_dataset), self.emb_layers_dim)).to(device) ### FIX: Find embeddings dimension either hard coded for model or real the pytorch model of ProGen. Maybe add reduction dimension as well
        self.py_model = self.py_model.to(device)
        i = 0
        self.py_model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled= fp16):
                for batch in seq_loader:
                    start = time.time()
                    if layer == 13:
                        out = self.py_model(batch[0]).logits
                    else:
                        out = self.py_model(batch[0]).hidden_states
                        out = out[layer - 1] 
                   
                    if reduction == 'mean':
                        embs[i : i+ batch_size, : ] = torch.mean(out , dim = 1)
                    elif reduction == 'sum':
                        embs[i : i+ batch_size, : ] = torch.sum(out , dim = 1)
                    else:
                        raise 'Unsupported reduction option'
                    del out
                    i = i + batch_size
                    logger.log(f' {i} / {len(seq_dataset)} | {time.time() - start:.2f}s ') # | memory usage : {100 - memory_usage.percent:.2f}%
           


        torch.save(embs,f'./data/{data_type}/embeddings/{data_type}_{self.name}_embs_layer{layer}_{reduction}.pt')
        t = torch.load(f'./data/{data_type}/embeddings/{data_type}_{self.name}_embs_layer{layer}_{reduction}.pt')
        logger.log(f'Saved embeddings ({t.shape[1]}-d) as "{data_type}_{self.name}_embs_layer{layer}_{reduction}.pt" ({time.time() - start_enc_time:.2f}s)')
        return
    
    def fine_tune(self, data_type, fine_tuner , train_split_name, optimizer , loss_f ):
        
        assert self.head != None , 'Task specific head haven\'t specified.'
        logger = l.Logger(f'logger_fine_tune_{self.name}_{self.head_name}_{fine_tuner.method}_{data_type}.txt')
        data = utils.load_dataset(data_type)           
        logger.log(f' Encoding {data.shape[0]} sequences....')
        start_enc_time = time.time()
        encs = utils.categorical_encode(data['aa_seq'].values, self.tokenizer , max(data['len'].values))   
        logger.log(f' Encoding completed! {time.time() -  start_enc_time:.4f}s')
        data_train = data[data[train_split_name] == 'train']
        data_test = data[data[train_split_name] == 'test']
        encs_train = encs[data_train.index]
        encs_test = encs[data_test.index]
        train_dataset = data_utils.TensorDataset( encs_train , torch.tensor(data_train['score'].values))  
        n_val_samples = int(fine_tuner.val_split * len(train_dataset))
        n_train_samples = len(train_dataset) - n_val_samples 
        train_set, val_set = torch.utils.data.random_split(train_dataset , [n_train_samples, n_val_samples]) 
        test_dataset = data_utils.TensorDataset( encs_test  , torch.tensor(data_test['score'].values))             
        train_dataloader = DataLoader(train_set, batch_size = fine_tuner.batch_size , shuffle=True)
        valid_dataloader = DataLoader(val_set, batch_size = fine_tuner.batch_size , shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size = fine_tuner.batch_size, shuffle=True)
        
        dataloader_dict = { 'train' : train_dataloader , 'val' : valid_dataloader}
    
        fine_tuner .set_trainable_parameters(self)
        ## Check if parameters of self model are affected just by calling them as argument
            ##TODO: move the whole training loop in tuner method train
        training_start_time = time.time()
        fine_tuner.train(self, dataloader_dict , optimizer , loss_f , logger)
        logger.log(' Finetuning  ({}) on {} data completed after {:.4f}s '.format(fine_tuner.method , data_type , time.time() - training_start_time))
        self.fine_tuned = fine_tuner.method
        return

    def evaluate(self):
        return 0
    
    def forward(self, src):
        src = self.py_model(src).logits ## 
        src = torch.mean(src , dim = 1)
        if self.head != None:
            src = self.head(src)
        return src

###TODO: Implement handler classes for different PLM families

class ESMFamily():
    pass

class ProtBERTFamilyM():
    pass

class AnkahFamily():
    pass

class SapiesFamily():
    pass

