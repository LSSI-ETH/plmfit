from sklearn.preprocessing import StandardScaler
import torch
import json
import pandas as pd
from tokenizers import Tokenizer
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from pynvml import *

# def load_model(model_name):
#   return ProGenForCausalLM.from_pretrained(f'./plmfit/language_models/progen2/checkpoints/{model_name}')


def load_embeddings(data_type, embs):
    embs_file = f'./plmfit/data/{data_type}/embeddings/{embs}'
    return torch.load(f'{embs_file}.pt', map_location=torch.device('cpu'))


def load_dataset(data_type):
    return pd.read_csv(f'./plmfit/data/{data_type}/{data_type}_data_full.csv')


def load_embeddings(emb_path=None, data_type='aav', layer='last', model='progen2-small', reduction='mean', device='cpu'):
    """
    Process data based on either a provided data path or specified data type, layer, model, and reduction method.

    Parameters:
        emb_path (str): Path to the embeddings file. If provided, other arguments are ignored (except for storage device).
        data_type (str): Type of data (default is 'aav').
        layer (str): Layer information (default is 'last').
        model (str): Model information (default is 'progen2-small').
        reduction (str): Reduction method (default is 'mean').
    """
    if emb_path is None:
        # Process data using the provided data path
        emb_path = f'./plmfit/data/{data_type}/embeddings/{data_type}_{model}_embs_layer{layer}_{reduction}.pt'
    else:
        emb_path = f'{emb_path}/{data_type}/embeddings/{data_type}_{model}_embs_layer{layer}_{reduction}.pt'

    embeddings = torch.load(emb_path, map_location=torch.device(device))
    embeddings = embeddings.numpy() if embeddings.is_cuda else embeddings
    return torch.tensor(embeddings, dtype=torch.float32)


def create_data_loaders(dataset, scores, split=None, test_size=0.2, validation_size=0.1, batch_size=64, scaler=None, dtype=torch.float32):
    """
    Create DataLoader objects for training, validation, and testing.

    Parameters:
        dataset (numpy.ndarray): Input dataset.
        scores (numpy.ndarray): Scores aligned with dataset.
        split (numpy.ndarray): Array indicating the split for each sample (train, test, validation).
                                If provided, test_size and validation_size are ignored.
        test_size (float): Fraction of the data to be used as the test set (default is 0.2).
        validation_size (float): Fraction of the training data to be used as the validation set (default is 0.1).
        batch_size (int): Batch size for DataLoader (default is 64).
        scaler (str): Scaler name for feature scaling (default is None). Supported scalers: 'standard'.

    Returns:
        dict: Dictionary containing DataLoader objects for train, validation, and test.
    """
    if split is None:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            dataset, scores, test_size=test_size, random_state=42)

        # Further split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size/(1-test_size), random_state=42)

    # TODO: see if this works for provided sets
    else:
        # Use the provided split
        X_train = dataset[split == 'train']
        X_test = dataset[split == 'test']
        X_val = dataset[split == 'validation']
        y_train = scores[split == 'train']
        y_test = scores[split == 'test']
        y_val = scores[split == 'validation']

    # Scale the features if scaler is provided
    if scaler is not None:
        if scaler == 'standard':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        else:
            raise "Unsupported scaler. Use 'standard' or None."

    # Convert splits to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=dtype)
    X_val = torch.tensor(X_val, dtype=dtype)
    X_test = torch.tensor(X_test, dtype=dtype)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training, validation, and testing
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def get_epoch_dataloaders(dataloader, epoch_size=0):
    if epoch_size == 0:
        return dataloader

    # Access the original dataset from the DataLoader
    train = dataloader['train'].dataset
    val = dataloader['val'].dataset
    
    # Randomly sample indices from the dataset
    train_inds = np.random.choice(len(train), epoch_size, replace=False)
    val_inds = np.random.choice(len(val), int(len(val) * epoch_size / len(train)), replace=False)
    
    # Create a Subset for the sampled indices
    train_set = Subset(train, train_inds)
    val_set = Subset(val, val_inds)
    
    # Create a new DataLoader for the subset
    # Preserve the original DataLoader's batch size, shuffle, and other parameters as needed
    train_dataloader = DataLoader(train_set, batch_size=dataloader['train'].batch_size, shuffle=True, 
                                   num_workers=dataloader['train'].num_workers, pin_memory=dataloader['train'].pin_memory)
    val_dataloader = DataLoader(val_set, batch_size=dataloader['val'].batch_size, shuffle=False, 
                                   num_workers=dataloader['val'].num_workers, pin_memory=dataloader['val'].pin_memory)
    
    return {'train': train_dataloader, 'val': val_dataloader, 'test': dataloader['test']}

def load_head_config(config_file_name):
    """
    Load a head configuration file from the plmfit/models/configurations directory.

    Parameters:
        config_file_name (str): Name of the head configuration file.

    Returns:
        dict: Loaded configuration.
    """
    # Construct the full path to the configuration file
    config_file_path = f'./plmfit/models/configurations/{config_file_name}'

    # Load the configuration from the file
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    return config


def get_wild_type(data_type):
    file = f'./plmfit/data/{data_type}'
    wild_type_f = open(f'{file}/wild_type.json')
    wt = json.load(wild_type_f)['wild_type']
    return wt


def load_tokenizer(model_name):
    model_file = ''
    if 'progen2' in model_name:
        model_file = 'progen2'
    file = f'./plmfit/language_models/{model_file}/tokenizer.json'

    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


def one_hot_encode(seqs):
    return torch.tensor([0])


def categorical_encode(seqs, tokenizer, max_len, add_bos=False, add_eos=False, logger = None):
    if logger != None:
        logger.log(f'Initiating categorical encoding')
        logger.log(f'Memory needed for encoding: {len(seqs) * max_len * 4}B')

    # Adjust max_len if BOS or EOS tokens are to be added
    internal_max_len = max_len + int(add_bos) + int(add_eos)

    seq_tokens = tokenizer.get_vocab()['<|pad|>'] * torch.ones((len(seqs), internal_max_len), dtype=int)
    for itr, seq in enumerate(seqs):
         # Encode the sequence without adding special tokens by the tokenizer itself
        encoded_seq_ids = tokenizer.encode(seq, add_special_tokens=False).ids

        # Prepare sequence with space for BOS and/or EOS if needed
        sequence = []
        if add_bos:
            sequence.append(tokenizer.get_vocab()['<|bos|>'])
        sequence.extend(encoded_seq_ids[:max_len])  # Ensure the core sequence does not exceed user-specified max_len
        if add_eos:
            sequence.append(tokenizer.get_vocab()['<|eos|>'])

        # Truncate the sequence if it exceeds internal_max_len
        truncated_sequence = sequence[:internal_max_len]

        # Update the seq_tokens tensor
        seq_len = len(truncated_sequence)
        seq_tokens[itr, :seq_len] = torch.tensor(truncated_sequence, dtype=torch.long)

        if itr == 0 and logger is not None:
            logger.log(f'First sequence tokens: {seq_tokens[0].tolist()}')
    if logger != None:
        logger.log(f'Categorical encoding finished')
    return seq_tokens


def get_parameters(model, print_w_mat=False, logger=None):
    s = 0
    c = 0
    for name, p in model.named_parameters():

        c += 1
        if print_w_mat:
            print(f' {name} size : {p.shape} trainable:{p.requires_grad}')
        if logger is not None:
            logger.log(f' {name} size : {p.shape} trainable:{p.requires_grad}')
        s += p.numel()

    return s

def trainable_parameters_summary(model, logger=None):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    output = f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {trainable_percentage:.3f}"
    if logger != None:
        logger.log(output)
    else:
        print(f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {trainable_percentage:.3f}")

def set_trainable_parameters(model, ft='all'):

    for name, p in model.named_parameters():
        p.requires_grad = True

    return

def unset_trainable_parameters_after_layer(model):
    # Set layers after self.layer_to_use to non-trainable
    for i, layer in enumerate(model.py_model.transformer.h):
        if i > model.layer_to_use:
            for param in layer.parameters():
                param.requires_grad = False


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

def log_model_info(log_file_path, data_params, model_params, training_params, eval_metrics):
    with open(log_file_path, 'w') as log_file:
        log_file.write("Data Parameters:\n")
        for param, value in data_params.items():
            log_file.write(f"{param}: {value}\n")

        log_file.write("\nModel Parameters:\n")
        for param, value in model_params.items():
            log_file.write(f"{param}: {value}\n")
        
        log_file.write("\nTraining Parameters:\n")
        for param, value in training_params.items():
            log_file.write(f"{param}: {value}\n")
        
        log_file.write("\nEvaluation Metrics:\n")
        for metric, value in eval_metrics.items():
            log_file.write(f"{metric}: {value}\n")
    
    print(f"Model information logged to {log_file_path}")

def convert_to_number(s):
    try:
        # First, try to convert the string to an integer
        return int(s)
    except ValueError:
        # If converting to an integer fails, try to convert it to a float
        try:
            return float(s)
        except ValueError:
            # If both conversions fail, return the original string or an indication that it's not a number
            return None  # or return s to return the original string

def print_gpu_utilization(memory_usage):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2
