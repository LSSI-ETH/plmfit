from language_models.progen2.models.progen.modeling_progen import ProGenForCausalLM
import torch
import json
import pandas as pd
from tokenizers import Tokenizer

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
