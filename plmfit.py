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

from plmfit.models.pretrained_models import MLP, LogisticRegression, AdapterLayer, LinearRegression

import torchmetrics


parser = argparse.ArgumentParser(description='plmfit_args')
# options ['progen2-small', 'progen2-xlarge', 'progen2-oas', 'progen2-medium', 'progen2-base', 'progen2-BFD90' , 'progen2-large']
parser.add_argument('--model_name', type=str, default='progen2-small')
parser.add_argument('--ft_method', type=str, default='feature_extraction')
parser.add_argument('--data_type', type=str, default='aav')
# here you specifcy the different splits
parser.add_argument('--data_file_name', type=str, default='data_train')
parser.add_argument('--embs', type=str,
                    default='aav_progen2-small_embs_layer12_mean')
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


args = parser.parse_args()

logger = l.Logger(
    f'logger_{args.model_name}_{args.ft_method}_{args.head}_{args.data_type}_SPECS:_filename:{args.data_file_name}_gpus:{args.gpus}_gres:{args.gres}_nodes:{args.nodes}.txt')

model = utils.load_model(args.model_name)
tokenizer = utils.load_tokenizer(args.model_name)
data = utils.load_dataset(args.data_type)
wild_type = utils.get_wild_type(args.data_type)

if __name__ == '__main__':

    # Preparing input token (embeddings, one hot encoded or categorical encoded)
    embs = None

    if args.ft_method == 'feature_extraction':  # added False here to not load embedding during development

        embs = utils.load_embeddings(args.data_type, args.embs)

    elif args.ft_method == 'feature_extraction' and args.embs == 'one-hot-encode':

        embs = utils.one_hot_encode(data['aa_seq'].values)

    else:

        embs = utils.categorical_encode(data['aa_seq'].values, tokenizer)

    assert embs != None, ' embeddings did not intialize'

    data_train = data[data[args.training_split] == 'train']
    data_test = data[data[args.training_split] == 'test']
    embs_train = embs[data_train.index]
    embs_test = embs[data_test.index]

# Initializing the task specific head

    head = None

    if args.head == 'mlp':

        config = utils.load_head_config(args.head_config)
        assert args.head == config["network_type"], f'The loaded configuration ("{config["network_type"]}") does not match the specified architecture "{args.head}". Make sure to upload a configuration with "network_type":"{args.head}"'
        head = MLP(config['input_len'], config['hidden_len'],
                   config['output_len'], config['activation_function'], config['dropout'])

    elif args.head == 'linear':
        # config = utils.load_head_config(args.head_config)
       # assert args.head == config["network_type"], f'The loaded configuration ("{config["network_type"]}") does not match the specified architecture "{args.head}". Make sure to upload a configuration with "network_type":"{args.head}"'
        # MLP(config['input_len'], config['hidden_len'], config['output_len'], config['activation_function'], config['dropout'])
        head = LinearRegression(embs.shape[1], 1)

    assert head != None, f' {args.task} head did not initialize'

# Prepare data loaders

    train_dataset = data_utils.TensorDataset(
        embs_train, torch.tensor(data_train['score'].values))
    n_val_samples = int(args.val_split * len(train_dataset))
    n_train_samples = len(train_dataset) - n_val_samples

    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [n_train_samples, n_val_samples])
    test_dataset = data_utils.TensorDataset(
        embs_test, torch.tensor(data_test['score'].values))

    train_dataloader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

# Concat pretraind model with task specific head (need to validate that dimenison match)

    # TODO : Check combatibility
    assert head.in_.in_features == embs.shape[1], f'Embeddings dimension ({embs.shape[1]}) is not compatible with the input size of the task specific head ({head.in_.in_features}) . Change "input_len" to {embs.shape[1]} in config file : {args.head_config}'

    ft_model = None

    if args.ft_method == 'feature_extraction':

        print('feature_extraction')
        ft_model = head
        # ft_model = ft_frameworks.feature_extraction(train_dataset , head)

    elif args.ft_method == 'retrain':

        ft_model = nn.Sequential(
            model,
            head
        )

    elif args.ft_method == 'ulmfit':

        print('ULMFIT')

    elif args.ft_method == 'lightweight':

        print('LIGHTWEIGHT')

    assert ft_model != None, f' {args.task} head did not initialize'

    logger.log(
        f' Number of trainable parameters ({args.model_name}/{args.ft_method}): {utils.get_parameters(ft_model)}')


# Start fine tuning / training

# -----------------------   Metrics definition  -------------------------------#

    specificity = torchmetrics.classification.BinarySpecificity(threshold=0.5)
    f1_score = torchmetrics.classification.BinaryF1Score(threshold=0.5)
    recall = torchmetrics.classification.BinaryRecall(threshold=0.5)
    accuracy = torchmetrics.classification.BinaryAccuracy(threshold=0.5)
    precision = torchmetrics.classification.BinaryPrecision(threshold=0.5)


# ----------------------- Logging parameters ---------------------------------#

    log_interval = 1000
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            ft_model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    else:
        raise 'Unsupported optimizer'

    criterion = nn.MSELoss(reduction='mean')
    lr_scheduler = False
    epoch_train_loss = []
    epoch_val_loss = []

    trainig_start_time = time.time()

    for epoch in range(args.epochs):

        epoch_start_time = time.time()
        logger.log('Epoch {}/{}'.format(epoch + 1, args.epochs))
        logger.log('-' * 10)
        for phase in ['train', 'val']:

            if phase == 'train':
                ft_model.train()  # Set model to training mode
                dataloader = train_dataloader
            else:
                ft_model.eval()   # Set model to evaluate mode
                dataloader = valid_dataloader

            batch_loss = 0

            for itr, trainig_data in enumerate(dataloader, 0):
                optimizer.zero_grad()
                training_batch, training_labels = trainig_data
                outputs = ft_model(training_batch)
                loss = criterion(torch.squeeze(
                    outputs).float(), training_labels.float())
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                batch_loss += loss.item()

                if itr % log_interval == 0:
                    logger.log(
                        f'({phase}) minibatch :{itr + 1}  / {len(dataloader)} | running_loss : {batch_loss / (itr + 1)}')

            epoch_loss = batch_loss / itr
            logger.log('({}) Loss: {:.4f}'.format(phase, epoch_loss))

            # TODO : Save ft_model after finetuning
