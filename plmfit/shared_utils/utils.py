from sklearn.preprocessing import StandardScaler
import torch
import json
import pandas as pd
from tokenizers import Tokenizer
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from pynvml import *
import os
import torch.nn as nn
import psutil
import importlib.util
from plmfit.logger import Logger

path = os.getenv('DATA_DIR')
data_dir = f'{path}/data'
config_dir = f'{path}/models/configurations'

def set_path(base_path):
    global path, data_dir, config_dir
    path = base_path
    data_dir = f'{base_path}/data'
    config_dir = f'{base_path}/models/configurations'

def load_dataset(data_type):
    return pd.read_csv(f'{data_dir}/{data_type}/{data_type}_data_full.csv')


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
        emb_path = f'{data_dir}/{data_type}/embeddings/{data_type}_{model}_embs_layer{layer}_{reduction}.pt'
    
    try:
        embeddings = torch.load(f"{emb_path}/{data_type}_{model}_embs_{layer}_{reduction}/{data_type}_{model}_embs_{layer}_{reduction}.pt", map_location=torch.device(device))
        #embeddings = embeddings.numpy() if embeddings.is_cuda else embeddings
        return embeddings.clone().detach().to(dtype=torch.float32)
    except:
        return None


def create_data_loaders(dataset, scores, split=None, test_size=0.2, validation_size=0.1, batch_size=64, scaler=None, dtype=torch.float32, num_workers=0, meta_data=None):
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
        scaler (bool): If to use feature scaling with a standard scaler.

    Returns:
        dict: Dictionary containing DataLoader objects for train, validation, and test.
    """
    if meta_data is None:
        # Return an empty array if meta_data is None
        meta_data = [np.array([], dtype=np.int8) for _ in range(len(dataset))]
    else:
        # Process each list within the list of lists
        processed_data = []
        for sub_list in meta_data:
            if sub_list:  # Check if the sub_list is not empty
                processed_data.append(np.array(sub_list, dtype=np.int8))
            else:
                processed_data.append(np.array([], dtype=np.int8))
        meta_data = processed_data

    if split is None:
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
                dataset, scores, meta_data, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
                X_train, y_train, meta_train, test_size=validation_size/(1-test_size), random_state=42)
    
    else:
        # Use the provided split
        X_train, X_val, X_test = dataset[split == 'train'], dataset[split == 'validation'], dataset[split == 'test']
        y_train, y_val, y_test = scores[split == 'train'], scores[split == 'validation'], scores[split == 'test']
        meta_train, meta_val, meta_test = meta_data[split == 'train'], meta_data[split == 'validation'], meta_data[split == 'test']

    # Scale the features if scaler is provided
    if scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Assuming X_train, X_val, X_test, y_train, y_val, y_test could be either NumPy arrays or PyTorch tensors
    X_train = convert_or_clone_to_tensor(X_train, dtype=dtype)
    X_val = convert_or_clone_to_tensor(X_val, dtype=dtype)
    X_test = convert_or_clone_to_tensor(X_test, dtype=dtype)
    y_train = convert_or_clone_to_tensor(y_train, dtype=torch.float32)
    y_val = convert_or_clone_to_tensor(y_val, dtype=torch.float32)
    y_test = convert_or_clone_to_tensor(y_test, dtype=torch.float32)
    meta_train, meta_val, meta_test = torch.tensor(meta_train), torch.tensor(meta_val), torch.tensor(meta_test)

    # Create DataLoader for training, validation, and testing
    train_dataset = TensorDataset(X_train, y_train, meta_train)
    val_dataset = TensorDataset(X_val, y_val, meta_val)
    test_dataset = TensorDataset(X_test, y_test, meta_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=num_workers>0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=num_workers>0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=num_workers>0)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def convert_or_clone_to_tensor(data, dtype):
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype)
    elif torch.is_tensor(data):
        return data.detach().clone().to(dtype=dtype)
    else:
        raise TypeError("Input data must be either a NumPy array or a PyTorch tensor.")
    
def get_epoch_dataloaders(dataloader, epoch_size=0):
    if epoch_size == 0:
        return dataloader

    # Access the original dataset from the DataLoader
    train = dataloader['train'].dataset
    val = dataloader['val'].dataset
    
    # Randomly sample indices from the dataset
    if epoch_size < 1: epoch_size = int(len(train) * epoch_size)
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

def load_config(config_file_name):
    """
    Load a head configuration file from the plmfit/models/configurations directory.

    Parameters:
        config_file_name (str): Name of the head configuration file.

    Returns:
        dict: Loaded configuration.
    """
    # Construct the full path to the configuration file
    config_file_path = f'{config_dir}/{config_file_name}'

    # Load the configuration from the file
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    return config
    


def get_activation_function(name):
    """Returns the activation function based on its name."""
    if name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    # Add more activation functions as needed
    else:
        raise f"Unsupported activation function: {name}"

def get_wild_type(data_type):
    file = f'{path}/data/{data_type}'
    wild_type_f = open(f'{file}/wild_type.json')
    wt = json.load(wild_type_f)['wild_type']
    return wt


def load_tokenizer(model_name):
    if 'progen2' in model_name:
        model_file = 'progen2'
    elif 'bert' in model_name:
        model_file = 'proteinbert'
    else:
        raise 'Model tokenizer not defined'
    
    file = f'{path}/language_models/{model_file}/tokenizer.json'

    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


def one_hot_encode(seqs):
    return torch.tensor([0])


def categorical_encode(seqs, tokenizer, max_len, add_bos=False, add_eos=False, logger = None, model_name='progen2'):
    if logger != None:
        logger.log(f'Initiating categorical encoding')
        logger.log(f'Memory needed for encoding: {len(seqs) * max_len * 4}B')

    if 'progen2' in model_name:
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
    elif 'bert' in model_name:
        # Adjust max_len if BOS or EOS tokens are to be added
        internal_max_len = max_len + int(add_bos) + int(add_eos)

        seq_tokens = tokenizer.get_vocab()['<pad>'] * torch.ones((len(seqs), internal_max_len), dtype=int)
        for itr, seq in enumerate(seqs):
            # Encode the sequence without adding special tokens by the tokenizer itself
            encoded_seq_ids = tokenizer.encode(seq, add_special_tokens=False).ids

            # Prepare sequence with space for BOS and/or EOS if needed
            sequence = []
            if add_bos:
                sequence.append(tokenizer.get_vocab()['<cls>'])
            sequence.extend(encoded_seq_ids[:max_len])  # Ensure the core sequence does not exceed user-specified max_len
            if add_eos:
                sequence.append(tokenizer.get_vocab()['<sep>'])

            # Truncate the sequence if it exceeds internal_max_len
            truncated_sequence = sequence[:internal_max_len]

            # Update the seq_tokens tensor
            seq_len = len(truncated_sequence)
            seq_tokens[itr, :seq_len] = torch.tensor(truncated_sequence, dtype=torch.long)

            if itr == 0 and logger is not None:
                logger.log(f'First sequence tokens: {seq_tokens[0].tolist()}')
    elif 'esm' in model_name:
        seq_tokens =  tokenizer.get_vocab()['<pad>'] * torch.ones((len(seqs) , max_len + 2) , dtype = int) ### Adding  to max_len because ESMTokenizer adds cls and eos tokens in the begging and the neding of aa_seq
        for itr , seq in enumerate(seqs):
            tok_seq = torch.tensor(tokenizer.encode(seq))
            seq_tokens[itr][:tok_seq.shape[0]] = tok_seq
    else:
        raise 'Model tokenizer not defined'
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

def check_module_states(model, logger=None):
    for name, module in model.named_modules():
        state = "Train" if module.training else "Eval"
        message = f'Module: {name} State: {state}'
        
        if logger is not None:
            logger.log(message)
        else:
            print(message)

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

def freeze_parameters(model):
    for name, p in model.named_parameters():
        p.requires_grad = False

    return

def disable_dropout(model):
    """
    Recursively set dropout probability to 0 for all Dropout layers.
    """
    for child in model.children():
        if isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            child.p = 0.0  # Set dropout probability to 0
        else:
            disable_dropout(child)

def set_modules_to_train_mode(model, module_name):
    """
    This function iterates through all modules in the given model.
    If it finds a module that has 'module_name' in its name,
    it sets that module to training mode.
    """
    for name, module in model.named_modules():
        # Identify modules by checking if 'module_name' is in their name.
        if module_name in name:
            module.train()  # Set the identified 'module_name' module to training mode

    # Note: This does not change the global training/evaluation mode of the model,
    # but specifically sets 'module_name' modules to training mode.

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

def print_gpu_utilization(memory_usage, device='cuda'):
    if 'cuda' in device:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used//1024**2
    else:
        memory = psutil.virtual_memory()
        return memory.used // 1024 ** 2  # Convert from Bytes to Megabytes

def print_cpu_utilization():
    # Get CPU utilization percentage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Utilization: {cpu_percent}%")

    # Get memory usage
    memory = psutil.virtual_memory()
    memory_used = memory.used // 1024 ** 2  # Convert from Bytes to Megabytes
    memory_total = memory.total // 1024 ** 2  # Convert from Bytes to Megabytes
    print(f"Memory Used: {memory_used} MB")
    print(f"Total Memory: {memory_total} MB")

    return cpu_percent, memory_used

def adjust_config_to_int(config, int_keys=[('training_parameters', 'batch_size'), ('architecture_parameters', 'hidden_dim')]):
    """
    Adjusts specific keys in a nested config dictionary to have integer values.

    Parameters:
    - config (dict): The nested configuration dictionary produced by the optimization algorithm.
    - int_keys (list of tuples): A list of tuples where each tuple contains the path to a key within the config that should have its value rounded to the nearest integer.

    Returns:
    - dict: The adjusted configuration dictionary with specified keys rounded to integers.
    """
    adjusted_config = config.copy()  # Make a copy to avoid modifying the original config

    for path in int_keys:
        # Navigate through the path to get to the desired key
        current_dict = adjusted_config
        for key in path[:-1]:  # Traverse to the parent dictionary of the target key
            current_dict = current_dict.get(key, {})
        
        # Adjust the final key in the path
        final_key = path[-1]
        if final_key in current_dict:
            current_dict[final_key] = int(round(current_dict[final_key]))

    return adjusted_config