import os
import math
import pandas as pd
from plmfit.language_models.progen2.models.progen.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer
import torch
import time
from scipy.stats.stats import pearsonr
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, EsmForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.utils.data as data_utils
from torch.nn import init
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import plmfit.logger as l
import argparse
import plmfit.shared_utils.utils as utils
from plmfit.models.pretrained_models import *

import torchmetrics


parser = argparse.ArgumentParser(description='plmfit_args')
# options ['progen2-small', 'progen2-xlarge', 'progen2-oas', 'progen2-medium', 'progen2-base', 'progen2-BFD90' , 'progen2-large']
parser.add_argument('--plm', type=str, default='progen2-small')
parser.add_argument('--ft_method', type=str, default='feature_extraction')
parser.add_argument('--data_type', type=str, default='aav')
# here you specifcy the different splits
parser.add_argument('--data_file_name', type=str, default='data_train')
parser.add_argument('--embs', type=str, default='aav_progen2-small_embs_layer12_mean')

# option ['mlp' , 'cnn' , 'inception', '{a custome head}' , 'attention']
parser.add_argument('--head', type=str, default='linear')
parser.add_argument('--head_config', type=str, default='config_mlp')
parser.add_argument('--task', type=str, default='cls')

parser.add_argument('--gpus', type=int, default=0)
parser.add_argument('--gres', type=str, default='gpumem:24g')
parser.add_argument('--mem-per-cpu', type=int, default=0)
parser.add_argument('--nodes', type=int, default=1)


parser.add_argument('--training_split', type=str, default='two_vs_many_split')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--val_split', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--function', type=str, default='extract_embeddings')
parser.add_argument('--reduction', type=str, default='mean')
parser.add_argument('--layer', type=int, default=0)

args = parser.parse_args()

logger = l.Logger(
    f'logger_{args.ft_method}_{args.head}_{args.data_type}_SPECS:_filename:{args.data_file_name}_gpus:{args.gpus}_gres:{args.gres}_nodes:{args.nodes}.txt')

if __name__ == '__main__':

    # Preparing input token (embeddings, one hot encoded or categorical encoded)
    model = None

    if 'progen' in args.plm:
        assert args.plm in ['progen2-small'], 'Progen version is not supported'
        model = ProGenFamily(args.plm)

    elif 'esm' in args.plm:
        assert args.plm in [''], 'ESM version is not supported'
        model = ESMFamily(args.plm)
    else: 
        raise 'PLM not supported'

    assert model != None , 'Model is not initialized'

    if args.function == 'extract_embeddings':

        model.extract_embeddings(data_type = args.data_type , layer= args.layer , reduction= args.reduction)
    
    else:

        raise 'Function is not supported'
