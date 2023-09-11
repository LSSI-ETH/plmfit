from language_models.progen2.models.progen.modeling_progen import ProGenForCausalLM
import torch
import json
import pandas as pd
from tokenizers import Tokenizer
from transformers import AutoTokenizer

def load_model(model_name):
    return ProGenForCausalLM.from_pretrained(f'./language_models/progen2/checkpoints/{model_name}')

def load_embeddings( data_type , embs ):
    embs_file = f'./data/{data_type}/embeddings/{embs}'
    return torch.load(f'{embs_file}.pt', map_location=torch.device('cpu'))

def load_dataset(data_type , data):  
    return pd.read_csv(f'./data/{data_type}/{data}.csv')

    
def get_wild_type(data_type):
    file = f'./data/{data_type}'
    wild_type_f = open(f'{file}/wild_type.json')
    wt = json.load(wild_type_f)['wild_type']       
    return wt

def load_tokenizer(model_name):
    model_file = ''
    if 'progen2' in model_name:
        model_file = 'progen2'
    file = f'./language_models/{model_file}/tokenizer.json'
   
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())
    
def load_head_config(config_file):
    file = f'./models/{config_file}'
    config_f = open(f'{file}.json')
    config = json.load(config_f)
    print(config)
    return config

def one_hot_encode(seqs):
    return torch.tensor([0])


def categorical_encode(seqs):
    return torch.tensor([0])

def get_parameters(model, print_w_mat = False):
    s =  0 
    c = 0
    for name, p in model.named_parameters():
        c += 1
        
        if print_w_mat:
            print(f' {name} size : {p.shape}')
        s += p.numel()
    return s


def read_fasta(file_path):
    sequences = {}
    current_sequence_id = None
    current_sequence = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # This line contains the sequence identifier
                if current_sequence_id is not None:
                    sequences[current_sequence_id] = ''.join(current_sequence)
                current_sequence_id = line[1:]
                current_sequence = []
            else:
                # This line contains sequence data
                if current_sequence_id is not None:
                    current_sequence.append(line)

    # Add the last sequence to the dictionary
    if current_sequence_id is not None:
        sequences[current_sequence_id] = ''.join(current_sequence)

    return sequences

