import os
from plmfit.language_models.progen2.models.progen.modeling_progen import ProGenForSequenceClassification
from plmfit.language_models.proteinbert import load_pretrained_model
from plmfit.language_models.proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs


import plmfit.shared_utils.utils as utils
import torch.nn as nn
import plmfit.logger as l
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import time
from abc import abstractmethod
from plmfit.models.fine_tuning import *
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModel, EsmForMaskedLM,  EsmForSequenceClassification
from antiberty import AntiBERTyRunner
from numpy import array
import psutil
import traceback

class IPretrainedProteinLanguageModel(nn.Module):

    name: str
    py_model: nn.Module
    head: nn.Module
    head_name: str
    no_parameters: int
    emb_layers_dim: int
    output_dim: int
    logger: l.Logger

    def __init__(self, logger = None):
        super().__init__()
        self.head_name = 'none'
        self.logger = logger

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
    def concat_task_specific_head(self, head):
        pass

    @abstractmethod
    def extract_embeddings(self, data_type, batch_size, layer=11, reduction='mean'):
        pass

    @abstractmethod
    def fine_tune(self, data_type, fine_tuner, train_split_name, optimizer, loss_f):
        pass

    @abstractmethod
    def evaluate(self, data_type):
        pass

    @abstractmethod
    def forward(self, src):
        pass

# TODO: infere based on aa_seq list
    @abstractmethod
    def infere(self, aa_seq_list):
        pass

  # Implement class for every supported Portein Language Model family


class Antiberty(IPretrainedProteinLanguageModel):
    def __init__(self):
        super().__init__()
        self.name = 'antiberty'
        self.logger = l.Logger(
            f'{self.name}')
        self.model = AntiBERTyRunner()

    def extract_embeddings(self, data_type, layer, reduction, output_dir = 'default'):
        logger = self.logger
        device = 'cpu'
        fp16 = False
        device_ids = []
        if torch.cuda.is_available():
            device = "cuda:0"
            fp16 = True
            self.logger.log(f'Available GPUs : {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                self.logger.log(
                    f'Running on {torch.cuda.get_device_properties(i).name}')
                device_ids.append(i)

        else:
            self.logger.log(f'No gpu found rolling device back to {device}')

        batch_size = 1
        if output_dir == 'default':
            output_path = f'./plmfit/data/{data_type}/embeddings/'
        else:
            output_path = f'{output_dir}/{data_type}/embeddings/'
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        start_emb_time = time.time()
        data = utils.load_dataset(data_type)
            
        self.logger = l.Logger(
            f'extract_embeddings_{data_type}_{self.name}_{reduction}')
        
        embs = torch.zeros((len(data), 1024)).to(device)

        for vh, vl in zip(data['V_h'], data['V_l']):
            start = time.time()
            out = self.model.embed([vh, vl])
            if reduction == 'mean':
                embs[i, 0:512] = torch.mean(out[0], dim=0)
                embs[i, 512:1024] = torch.mean(out[1], dim=0)
            elif reduction == 'sum':
                embs[i, 0:512] = torch.sum(out[0], dim=0)
                embs[i, 512:1024] = torch.sum(out[1], dim=0)
            elif reduction == 'bos':
                # Select the embeddings for the first token of each sequence in the batch
                embs[i, 0:512] = out[0][0, :]
                embs[i, 512:1024] = out[1][0, :]
            elif reduction == 'eos':
                # Select the embeddings for the last token of each sequence in the batch
                embs[i, 0:512] = out[0][-1, :]
                embs[i, 512:1024] = out[1][-1, :]
            else:
                raise ValueError('Unsupported reduction option')
            del out
            i = i + batch_size
            logger.log(
                f' {i} / {len(data)} | {time.time() - start:.2f}s ')
        
        torch.save(
            embs, os.path.join(output_path, f'{data_type}_{self.name}_embs_layer{layer}_{reduction}.pt'))
        t = torch.load(
            os.path.join(output_path, f'{data_type}_{self.name}_embs_layer{layer}_{reduction}.pt'))
        logger.log(
            f'Saved embeddings ({t.shape[1]}-d) as "{data_type}_{self.name}_embs_layer{layer}_{reduction}.pt" ({time.time() - start_emb_time:.2f}s)')
        return
                
        
class ProGenFamily(IPretrainedProteinLanguageModel):

    tokenizer: Tokenizer

    def __init__(self, progen_model_name: str, logger : l.Logger):
        # IPretrainedProteinLanguageModel.__init__(self)
        super().__init__(logger)
        self.name = progen_model_name
        self.py_model = ProGenForSequenceClassification.from_pretrained(
            f'./plmfit/language_models/progen2/checkpoints/{progen_model_name}')
        self.no_parameters = utils.get_parameters(self.py_model)
        self.no_layers = len(self.py_model.transformer.h)
        self.output_dim = self.py_model.classifier.out_features
        self.emb_layers_dim = self.py_model.transformer.h[0].attn.out_proj.out_features
        self.tokenizer = utils.load_tokenizer(progen_model_name)
        self.layer_to_use = self.no_layers - 1
        self.config = self.py_model.config
  

    def concat_task_specific_head(self, head):
        # assert head.in_.in_features == self.output_dim, f'Head\'s input dimension ({head.in_.in_features}) is not compatible with {self.name}\'s output dimension ({self.output_dim}). To concat modules these must be equal.'
        # TODO: Add concat option with lm head or final transformer layer.
        self.head = head
        self.head_name = head.__class__.__name__  # parse name from variable
        self.no_parameters += utils.get_parameters(head)
        return
    
    def categorical_encode(self, data):
        encs = utils.categorical_encode(
            data['aa_seq'].values, self.tokenizer, max(data['len'].values), add_bos=True, add_eos=True, logger=self.logger)
        return encs

    def extract_embeddings(self, data_type, batch_size = 1, layer=11, reduction='mean', log_interval=1000):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            memory_usage = psutil.virtual_memory()
            max_mem_usage = utils.print_gpu_utilization(memory_usage, device)
            fp16 = False
            device_ids = []
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(1/1)
                torch.cuda.memory._record_memory_history(enabled='all', max_entries=100000)
                device = "cuda:0"
                fp16 = True
                self.logger.log(f'Available GPUs : {torch.cuda.device_count()}')
                for i in range(torch.cuda.device_count()):
                    self.logger.log(
                        f'Running on {torch.cuda.get_device_properties(i).name}')
                    device_ids.append(i)

            else:
                self.logger.log(f'No gpu found rolling device back to {device}')
                
            data = utils.load_dataset(data_type)

            start_enc_time = time.time()
            self.logger.log(f'Encoding {data.shape[0]} sequences....')
            encs = self.categorical_encode(data)
            mem_usage = utils.print_gpu_utilization(memory_usage, device)
            if mem_usage > max_mem_usage: max_mem_usage = mem_usage
            self.logger.log(
                f'Encoding completed! {time.time() -  start_enc_time:.4f}s')
            enc_time = time.time() -  start_enc_time
            encs = encs.to(device)
            seq_dataset = data_utils.TensorDataset(encs)
            seq_loader = data_utils.DataLoader(
                seq_dataset, batch_size=batch_size, shuffle=False)
            self.logger.log(
                f'Extracting embeddings for {len(seq_dataset)} sequences...')

            # FIX: Find embeddings dimension either hard coded for model or real the pytorch model of ProGen. Maybe add reduction dimension as well
            embs = torch.zeros((len(seq_dataset), self.emb_layers_dim)).to(device)
            self.py_model = self.py_model.to(device)
            i = 0
            self.py_model.eval()
            start_extraction_time = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=fp16, cache_enabled=False):
                    for batch in seq_loader:
                        start = time.time()
                        current_batch_size = batch[0].shape[0]
                        if layer == 'logits':
                            out = self.py_model(batch[0]).logits
                            mem_usage = utils.print_gpu_utilization(memory_usage, device)
                            if mem_usage > max_mem_usage: max_mem_usage = mem_usage
                        else:
                            model_output = self.py_model(batch[0])
                            mem_usage = utils.print_gpu_utilization(memory_usage, device)
                            if mem_usage > max_mem_usage: max_mem_usage = mem_usage
                            hidden_states = model_output.hidden_states  # Get all hidden states

                            # Log the shape of each layer's embeddings for the first batch if we are unsure what our model outputs
                            # if i == 0:
                            #     for layer_index, layer_output in enumerate(hidden_states):
                            #         self.logger.log(f'Layer {layer_index} shape: {layer_output.shape}')

                            # Determine the layer index based on the 'layer' description
                            if layer == 'last':
                                # The last hidden layer (not counting the logits layer)
                                selected_layer_index = len(hidden_states) - 1
                            elif layer == 'middle':
                                # Adjusted to consider the first transformer block as the "first" layer
                                selected_layer_index = 1 + (len(hidden_states) - 1) // 2
                            elif layer == 'first':
                                # The first transformer block after the input embeddings
                                selected_layer_index = 1  # Adjusted to 1 to skip the input embeddings
                            else:
                                # Fallback for numeric layer specification or unexpected strings
                                selected_layer_index = int(layer) if layer.isdigit() else 0

                            # Now select the specific layer's output
                            out = hidden_states[selected_layer_index]
                        if reduction == 'mean':
                            embs[i: i + current_batch_size, :] = torch.mean(out, dim=1)
                            if i == 0:
                                self.logger.log(f'{(torch.mean(out, dim=1)).size()}')
                        elif reduction == 'sum':
                            embs[i: i + current_batch_size, :] = torch.sum(out, dim=1)
                        elif reduction == 'bos':
                            # Select the embeddings for the first token of each sequence in the batch
                            embs[i: i + current_batch_size, :] = out[:, 0, :]
                        elif reduction == 'eos':
                            # Initialize a tensor to store the selected embeddings
                            selected_embs = torch.zeros(current_batch_size, out.shape[2], device=out.device)
                            
                            # Iterate over each sequence in the batch
                            for seq_idx in range(current_batch_size):
                                # Find the positions of the token with ID equal to 2 in the current sequence
                                token_positions = (batch[0][seq_idx] == 2).nonzero(as_tuple=True)[0]
                                
                                # Check if the token ID is present in the sequence
                                if len(token_positions) > 0:
                                    # Select the position of the last occurrence of the token
                                    last_position = token_positions[-1].item()
                                    
                                    # Select the embeddings for the last occurrence of the token
                                    selected_embs[seq_idx, :] = out[seq_idx, last_position, :]
                                else:
                                    # If the token ID is not found, you might want to handle this case.
                                    # For example, use the embeddings of the last token of the sequence
                                    # or set to zeros, depending on your application's requirements.
                                    # Here, we use the embeddings of the last token as a fallback.
                                    selected_embs[seq_idx, :] = out[seq_idx, -1, :]
                                    self.logger.log(f'EOS token not found for sequence {i}')
                            
                            # Update the embeddings tensor with the selected embeddings
                            embs[i: i + current_batch_size, :] = selected_embs
                        elif utils.convert_to_number(reduction) is not None:
                            # Select the embeddings for the i token of each sequence in the batch
                            embs[i: i + current_batch_size, :] = out[:, utils.convert_to_number(reduction), :]
                        else:
                            raise ValueError('Unsupported reduction option')
                        del out
                        i = i + current_batch_size
                        if log_interval != -1 and i % log_interval == 0:
                            self.logger.log(
                                f' {i} / {len(seq_dataset)} | {time.time() - start:.2f}s ')
            extraction_time = time.time() - start_extraction_time
            torch.save(embs, f'{self.logger.base_dir}/{self.logger.experiment_name}.pt')
            file_size_bytes = os.path.getsize(f'{self.logger.base_dir}/{self.logger.experiment_name}.pt')
            file_size_mb = file_size_bytes / (1024 * 1024) # Convert bytes to megabytes
            report = {
                "encoding_time_needed": f'{enc_time:.4f}',
                "extraction_time_needed": f'{extraction_time:.4f}',
                "avg_time_per_seq": f'{extraction_time / len(seq_dataset):.4f}',
                "dataset_len": len(seq_dataset),
                "embeddings_dim": embs.shape,
                "data_type": data_type,
                "layer": layer,
                "reduction": reduction,
                "batch_size": batch_size,
                "embeddings_file_size_mb": f'{file_size_mb:.2f}',
                "max_vram_usage_mb": max_mem_usage
            }
            self.logger.save_data(report, 'report')
            t = torch.load(f'{self.logger.base_dir}/{self.logger.experiment_name}.pt')
            self.logger.log(
                f'Saved embeddings ({t.shape[1]}-d) as "{self.logger.experiment_name}.pt" ({time.time() - start_enc_time:.2f}s)')
            
            torch.cuda.memory._dump_snapshot(f'{self.logger.base_dir}/memory_profiler.pickle')
            torch.cuda.memory._record_memory_history(enabled=None)
            return
        except:
            torch.cuda.memory._dump_snapshot(f'{self.logger.base_dir}/memory_profiler.pickle')
            torch.cuda.memory._record_memory_history(enabled=None)
            stack_trace = traceback.format_exc()
            self.logger.log(stack_trace)

    def fine_tune(self, data_type, fine_tuner, train_split_name, optimizer, loss_f):

        assert self.head != None, 'Task specific head haven\'t specified.'
        
        data = utils.load_dataset(data_type)
        self.logger.log(f' Encoding {data.shape[0]} sequences....')
        start_enc_time = time.time()
        encs = utils.categorical_encode(
            data['aa_seq'].values, self.tokenizer, max(data['len'].values), add_bos=True, add_eos=True, logger=self.logger)
        self.logger.log(
            f' Encoding completed! {time.time() -  start_enc_time:.4f}s')
        data_train = data[data[train_split_name] == 'train']
        data_test = data[data[train_split_name] == 'test']
        encs_train = encs[data_train.index]
        encs_test = encs[data_test.index]
        train_dataset = data_utils.TensorDataset(
            encs_train, torch.tensor(data_train['score'].values))
        n_val_samples = int(fine_tuner.val_split * len(train_dataset))
        n_train_samples = len(train_dataset) - n_val_samples
        train_set, val_set = torch.utils.data.random_split(
            train_dataset, [n_train_samples, n_val_samples])
        test_dataset = data_utils.TensorDataset(
            encs_test, torch.tensor(data_test['score'].values))
        train_dataloader = DataLoader(
            train_set, batch_size=fine_tuner.batch_size, shuffle=True)
        valid_dataloader = DataLoader(
            val_set, batch_size=fine_tuner.batch_size, shuffle=True)
        test_dataloader = DataLoader(
            test_dataset, batch_size=fine_tuner.batch_size, shuffle=True)

        dataloader_dict = {'train': train_dataloader, 'val': valid_dataloader}

        fine_tuner .set_trainable_parameters(self)
        # Check if parameters of self model are affected just by calling them as argument
        # TODO: move the whole training loop in tuner method train
        training_start_time = time.time()
        fine_tuner.train(self, dataloader_dict, optimizer, loss_f, self.logger)
        self.logger.log(' Finetuning  ({}) on {} data completed after {:.4f}s '.format(
            fine_tuner.method, data_type, time.time() - training_start_time))
        self.fine_tuned = fine_tuner.method
        return
    
    def set_layer_to_use(self, layer):
        if layer == 'last':
            # The last hidden layer (not counting the logits layer)
            self.layer_to_use = self.no_layers - 1
        elif layer == 'middle':
            # Adjusted to consider the first transformer block as the "first" layer
            self.layer_to_use = 1 + (self.no_layers - 1) // 2
        elif layer == 'first':
            # The first transformer block after the input embeddings
            self.layer_to_use = 1  # Adjusted to 1 to skip the input embeddings
        else:
            # Fallback for numeric layer specification or unexpected strings
            self.layer_to_use = int(layer) if layer.isdigit() else self.no_layers - 1

    def evaluate(self):
        return 0

    def forward(self, src):
        src = self.py_model(src).hidden_states[self.layer_to_use]
        src = torch.mean(src, dim=1)
        if self.head != None:
            src = self.head(src)
        return src
    
class ProGenClassifier(ProGenFamily):
    def __init__(self, progen_model_name: str, head):
        # IPretrainedProteinLanguageModel.__init__(self)
        super().__init__(progen_model_name)
        self.classifier = head
        self.no_parameters += utils.get_parameters(head)

    def forward(self, input_ids, *args, **kwargs):
        src = self.py_model(input_ids).hidden_states[self.layer_to_use]
        src = torch.mean(src, dim=1)
        src = self.classifier(src)
        return src

# TODO: Implement handler classes for different PLM families
class ESMFamily(IPretrainedProteinLanguageModel):
    tokenizer : AutoTokenizer
    
    def __init__(self , esm_version : str, logger : l.Logger, task_type = "extract_embeddings", ):
        super().__init__()
        self.version = esm_version
        model_type = EsmForMaskedLM
        if task_type != "extract_embeddings":
            model_type = EsmForSequenceClassification
        self.py_model = model_type.from_pretrained(f'facebook/{esm_version}' , output_hidden_states = True)
        self.no_parameters = utils.get_parameters(self.py_model)
        self.no_layers = len(self.py_model.esm.encoder.layer)
        self.output_dim = self.py_model.config.vocab_size
        self.emb_layers_dim =  self.py_model.esm.encoder.layer[0].attention.self.query.in_features
        self.tokenizer = AutoTokenizer.from_pretrained(f'facebook/{esm_version}')
        self.adapters = nn.ModuleList()
        self.logger = logger
        
    def extract_embeddings(self , data_type , batch_size = 4 , layer = 48, reduction = 'mean',mut_pos = None):
        
        device = 'cpu'
        fp16 = False
        device_ids = []
        if torch.cuda.is_available():
            device = "cuda:0"
            fp16 = True
            self.logger.log(f'Available GPUs : {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                self.logger.log(f' Running on {torch.cuda.get_device_properties(i).name}')
                device_ids.append(i)

        else:
            self.logger.log(f' No gpu found rolling device back to {device}')
        data = utils.load_dataset(data_type)
        start_enc_time = time.time()
        self.logger.log(f' Encoding {data.shape[0]} sequences....')
        encs = self.categorical_encode(data['aa_seq'].values, self.tokenizer,max(data['len'].values)) 
        self.logger.log(f' Encoding completed! {time.time() -  start_enc_time:.4f}s')
        encs = encs.to(device)
        
        seq_dataset = data_utils.TensorDataset(encs)
        seq_loader =  data_utils.DataLoader(seq_dataset, batch_size= batch_size, shuffle=False)
        self.logger.log(f'Extracting embeddings for {len(seq_dataset)} sequences...')
        
        # Layers and reduction received from parser will be in string format
        layer = np.array(layer.split("_"))
        layer[layer == "last"] = self.no_layers
        layer[layer == "first"] = 1
        layer[layer == "middle"] = self.no_layers//2 + 1
        layer = layer.astype(int)
        reduction = reduction.split("_")
        
        embs = torch.zeros(len(layer),len(reduction),len(seq_dataset), self.emb_layers_dim).to(device) ### FIX: Find embeddings dimension either hard coded for model or real the pytorch model of ProGen. Maybe add reduction dimension as well
        
        self.py_model = self.py_model.to(device)
        self.py_model.eval()
        
        i = 0
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled= fp16):
                for batch in seq_loader:
                    start = time.time()
                    out = self.py_model(batch[0]).hidden_states
                    for j in range(len(layer)):
                        lay = layer[j]
                        for k in range(len(reduction)):
                            if reduction[k] == 'mean':
                                embs[j,k,i : i+ batch_size, : ] = torch.mean(out[lay] , dim = 1)
                            elif reduction[k] == 'sum':
                                embs[j,k,i : i+ batch_size, : ] = torch.sum(out[lay] , dim = 1)
                            elif reduction[k] == 'eos':
                                embs[j,k,i : i+ batch_size, : ] = out[lay][:,-1]
                            elif reduction[k] == 'bos':
                                embs[j,k,i : i+ batch_size, : ] = out[lay][:,0]
                            elif reduction[k].startswith('pos'):
                                embs[j,k,i : i+ batch_size, : ] = out[lay][:,int(reduction[k][3:])]
                            elif reduction[k] == 'mut_mean':
                                n_pos = len(mut_pos)
                                f_pos = mut_pos[0]
                                for pos in mut_pos[1:]:
                                    out[lay][:,f_pos] = torch.add(out[lay][:,f_pos],out[lay][:,pos])
                                embs[j,k,i : i+ batch_size, : ] = torch.div(out[lay][:,f_pos],n_pos)
                            else:
                                raise 'Unsupported reduction option'
                    del out
                    i = i + batch_size
                    self.logger.log(f' {i} / {len(seq_dataset)} | {time.time() - start:.2f}s ') # | memory usage : {100 - memory_usage.percent:.2f}%

           
        os.makedirs(f'./plmfit/data/{data_type}/embeddings', exist_ok = True)
        for j in range(len(layer)):
            lay = layer[j]
            for k in range(len(reduction)):
                tmp = embs[j,k].detach().clone()
                torch.save(tmp,f'{self.logger.base_dir}/{data_type}_{self.version}_embs_layer{layer[j]}_{reduction[k]}.pt')
                self.logger.log(f'Saved embeddings ({tmp.shape[1]}-d) as "{data_type}_{self.version}_embs_layer{layer[j]}_{reduction[k]}.pt" ({time.time() - start_enc_time:.2f}s)')
                del tmp
        return

    
    def fine_tune(self, data_type, fine_tuner , train_split_name, optimizer , loss_f ):
        assert self.head != None , 'Task specific head haven\'t specified.'
        logger = l.Logger(f'logger_fine_tune_{self.version}_{self.head_type}_{fine_tuner.method}_{data_type}.txt')
        data = utils.load_dataset(data_type) 
        #data = data[data[train_split_name] == 'train'].head(50)
        #data.reset_index(inplace = True) #### Remove after testing          
        logger.log(f' Encoding {data.shape[0]} sequences....')
        start_enc_time = time.time()
        encs = self.categorical_encode(data['aa_seq'].values, self.tokenizer , max(data['len'].values))
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
        #test_dataloader = DataLoader(test_dataset, batch_size = fine_tuner.batch_size, shuffle=True)
        
        dataloader_dict = { 'train' : train_dataloader , 'val' : valid_dataloader}
    
        fine_tuner .set_trainable_parameters(self)
        ## Check if parameters of self model are affected just by calling them as argument
            ##TODO: move the whole training loop in tuner method train
        training_start_time = time.time()
        fine_tuner.train(self, dataloader_dict , optimizer , loss_f , logger)
        logger.log(' Finetuning  ({}) on {} data completed after {:.4f}s '.format(fine_tuner.method , data_type , time.time() - training_start_time))
        self.fine_tuned = fine_tuner.method
        return
    
    def evaluate(self, data_type ):
        pass
    
    
    def categorical_encode(self, seqs, tokenizer , max_len):
        seq_tokens =  tokenizer.get_vocab()['<pad>'] * torch.ones((len(seqs) , max_len + 2) , dtype = int) ### Adding  to max_len because ESMTokenizer adds cls and eos tokens in the begging and the neding of aa_seq
        for itr , seq in enumerate(seqs):
            tok_seq = torch.tensor(tokenizer.encode(seq))
            seq_tokens[itr][:tok_seq.shape[0]] = tok_seq
        return seq_tokens

class ProteinBERTFamily(IPretrainedProteinLanguageModel):
    def __init__(self, bert_model_name: str):
        super().__init__()
        self.name = bert_model_name
        # Load the pre-trained BERT model. Replace with the appropriate method for BERT.
        pretrained_model_generator, input_encoder = load_pretrained_model()
        model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(1024))
        print(model)

    # Implement the abstract methods for BERT
    def concat_task_specific_head(self, head):
        # Implementation for BERT...
        pass

    def extract_embeddings(self, data_type, batch_size, layer, reduction):
        # Implementation for BERT...
        pass

    def fine_tune(self, data_type, fine_tuner, train_split_name, optimizer, loss_f):
        # Implementation for BERT...
        pass

class AnkhFamily(IPretrainedProteinLanguageModel):
    
    tokenizer : AutoTokenizer
    
    def __init__(self , ankh_version : str):
        super().__init__()
        self.version = ankh_version 
        self.py_model = AutoModel.from_pretrained(f'ElnaggarLab/{ankh_version}' , output_hidden_states = True)
        self.no_parameters = utils.get_parameters(self.py_model.encoder)
        self.no_layers = self.py_model.config.num_layers
        self.output_dim = self.py_model.config.vocab_size
        self.emb_layers_dim =  self.py_model.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(f'ElnaggarLab/{ankh_version}')
        self.adapters = nn.ModuleList()
        
    def extract_embeddings(self , data_type , batch_size = 4 , layer = 48, reduction = 'mean', mut_pos= None):
        logger = l.Logger(f'logger_extract_embeddings_{data_type}_{self.version}_layer{layer}_{reduction}')
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
        encs = self.categorical_encode(data['aa_seq'].values, self.tokenizer,max(data['len'].values)) 
        logger.log(f' Encoding completed! {time.time() -  start_enc_time:.4f}s')
        encs = encs.to(device)
        
        seq_dataset = data_utils.TensorDataset(encs)
        seq_loader =  data_utils.DataLoader(seq_dataset, batch_size= batch_size, shuffle=False)
        logger.log(f'Extracting embeddings for {len(seq_dataset)} sequences...')
        
        if type(layer) == int:
            layer = [layer]
        
        if type(reduction) == str:
            reduction = [reduction]
        
        embs = torch.zeros(len(layer),len(reduction),len(seq_dataset), self.emb_layers_dim).to(device) ### FIX: Find embeddings dimension either hard coded for model or real the pytorch model of ProGen. Maybe add reduction dimension as well
        
        self.py_model = self.py_model.encoder.to(device)
        self.py_model.eval()
        
        i = 0
        with torch.no_grad():
            #with torch.cuda.amp.autocast(enabled= fp16):
            for batch in seq_loader:
                start = time.time()
                out = self.py_model(batch[0]).hidden_states
                for j in range(len(layer)):
                    lay = layer[j]
                    for k in range(len(reduction)):
                        if reduction[k] == 'mean':
                            embs[j,k,i : i+ batch_size, : ] = torch.mean(out[lay] , dim = 1)
                        elif reduction[k] == 'sum':
                            embs[j,k,i : i+ batch_size, : ] = torch.sum(out[lay] , dim = 1)
                        elif reduction[k] == 'eos':
                            embs[j,k,i : i+ batch_size, : ] = out[lay][:,-1]
                        elif reduction[k].startswith('pos'):
                            embs[j,k,i : i+ batch_size, : ] = out[lay][:,int(reduction[k][3:])]
                        elif reduction[k] == 'mut_mean':
                            n_pos = len(mut_pos)
                            f_pos = mut_pos[0]
                            for pos in mut_pos[1:]:
                                out[lay][:,f_pos] = torch.add(out[lay][:,f_pos],out[lay][:,pos])
                            embs[j,k,i : i+ batch_size, : ] = torch.div(out[lay][:,f_pos],n_pos)
                        else:
                            raise 'Unsupported reduction option'
                del out
                i = i + batch_size
                logger.log(f' {i} / {len(seq_dataset)} | {time.time() - start:.2f}s ') # | memory usage : {100 - memory_usage.percent:.2f}%

           
        os.makedirs(f'./plmfit/data/{data_type}/embeddings', exist_ok = True)
        for j in range(len(layer)):
            lay = layer[j]
            for k in range(len(reduction)):
                tmp = embs[j,k].detach().clone()
                torch.save(tmp,f'./plmfit/data/{data_type}/embeddings/{data_type}_{self.version}_embs_layer{layer[j]}_{reduction[k]}.pt')
                logger.log(f'Saved embeddings ({tmp.shape[1]}-d) as "{data_type}_{self.version}_embs_layer{layer[j]}_{reduction[k]}.pt" ({time.time() - start_enc_time:.2f}s)')
                del tmp
        return
    
    def categorical_encode(self, seqs, tokenizer, max_len):
        seq_tokens =  tokenizer.get_vocab()['<pad>'] * torch.ones((len(seqs) , max_len + 1) , dtype = int) ### Adding  to max_len because ESMTokenizer adds cls and eos tokens in the begging and the neding of aa_seq
        for itr , seq in enumerate(seqs):
            tok_seq = torch.tensor(tokenizer.encode(seq))
            seq_tokens[itr][:tok_seq.shape[0]] = tok_seq
        return seq_tokens

class SapiesFamily():
    pass
