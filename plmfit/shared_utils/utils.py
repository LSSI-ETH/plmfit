from sklearn.preprocessing import StandardScaler
import torch
import json
import pandas as pd
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    Subset,
    random_split,
    WeightedRandomSampler,
)
from sklearn.model_selection import train_test_split
import numpy as np
from pynvml import *
import os
import torch.nn as nn
import psutil
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset
from plmfit.models.pretrained_models import (
    Antiberty,
    ESMFamily,
    ProGenFamily,
    ProteinBERTFamily,
    AnkhFamily,
    ESMCFamily
)
from dotenv import load_dotenv
import blosum as bl
from collections import Counter
import torch.nn.functional as F
import ast
from plmfit.shared_utils.random_state import get_random_state, get_numpy_random_state
from concurrent.futures import ProcessPoolExecutor, as_completed
from plmfit.shared_utils.samplers import LabelWeightedSampler
from esm.utils import encoding
from optuna.trial import Trial

load_dotenv()
plmfit_path = os.getenv("PLMFIT_PATH", "./plmfit")
data_dir = os.getenv("DATA_DIR", "./data")
config_dir = os.getenv("CONFIG_DIR", "./config")

def set_plmfit_path(base_path):
    global plmfit_path
    plmfit_path = base_path
    os.environ["PLMFIT_PATH"] = plmfit_path

def set_data_dir(base_path):
    global data_dir
    data_dir = base_path
    os.environ["DATA_DIR"] = data_dir

def set_config_dir(base_path):
    global config_dir
    config_dir = base_path
    os.environ["CONFIG_DIR"] = config_dir

def set_path(base_path):
    global path, data_dir, config_dir
    path = base_path
    data_dir = os.getenv("DATA_DIR", "./data")
    config_dir = os.getenv("CONFIG_DIR", "./config")


def load_dataset(data_type):
    try:
        return pd.read_csv(f"{data_dir}/{data_type}/{data_type}_data_full.csv")
    except:
        return pd.read_csv(data_type) # Assume it is a full path to dataset


def load_embeddings(
    emb_path=None,
    data_type="aav",
    layer="last",
    model="progen2-small",
    reduction="mean",
    device="cpu",
):
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
        emb_path = f"{data_dir}/{data_type}/embeddings/{data_type}_{model}_embs_layer{layer}_{reduction}.pt"

    try:
        embeddings = torch.load(
            f"{emb_path}/{data_type}_{model}_embs_{layer}_{reduction}/{data_type}_{model}_embs_{layer}_{reduction}.pt",
            map_location=torch.device(device),
        )
        # embeddings = embeddings.numpy() if embeddings.is_cuda else embeddings
        return embeddings
    except:
        try:
            embeddings = torch.load(
                f"{emb_path}",
                map_location=torch.device(device),
            )
            # embeddings = embeddings.numpy() if embeddings.is_cuda else embeddings
            return embeddings
        except:
            return None


def create_data_loaders(
    dataset,
    scores,
    split=None,
    test_size=0.2,
    validation_size=0.1,
    batch_size=64,
    scaler=None,
    dtype=torch.float16,
    num_workers=0,
    weights=None,
    sampler=False,
    dataset_type="tensor",
):
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
    random_state = get_random_state()
    if split is None:
        random_state = get_numpy_random_state()
        X_train, X_test, y_train, y_test = train_test_split(
            dataset, scores, test_size=test_size, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_size / (1 - test_size),
            random_state=random_state,
        )

        if weights is not None:
            # Splitting with weights for the initial train-test split
            X_train, X_test, y_train, y_test, weights_train, weights_test = (
                train_test_split(
                    dataset,
                    scores,
                    weights,
                    test_size=test_size,
                    random_state=random_state,
                )
            )

            # Splitting with weights for the train-validation split
            X_train, X_val, y_train, y_val, weights_train, weights_val = (
                train_test_split(
                    X_train,
                    y_train,
                    weights_train,
                    test_size=validation_size / (1 - test_size),
                    random_state=random_state,
                )
            )

    else:
        # Use the provided split
        X_train, X_val, X_test = (
            dataset[split == "train"],
            dataset[split == "validation"],
            dataset[split == "test"],
        )
        y_train, y_val, y_test = (
            scores[split == "train"],
            scores[split == "validation"],
            scores[split == "test"],
        )
        if weights is not None:
            weights_train, weights_val, weights_test = (
                weights[split == "train"],
                weights[split == "validation"],
                weights[split == "test"],
            )

        # Check if the validation set is empty and split the training data if necessary
        if X_val.shape[0] == 0 or y_val.shape[0] == 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=validation_size, random_state=random_state
            )
            if weights is not None:
                X_train, X_val, y_train, y_val, weights_train, weights_val = (
                    train_test_split(
                        X_train,
                        y_train,
                        weights_train,
                        test_size=validation_size,
                        random_state=random_state,
                    )
                )

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
    # Add to X_test an identifier
    test_ids = torch.arange(X_test.size(0))

    y_train = convert_or_clone_to_tensor(y_train, dtype=torch.float16)
    y_val = convert_or_clone_to_tensor(y_val, dtype=torch.float16)
    y_test = convert_or_clone_to_tensor(y_test, dtype=torch.float16)

    if weights is not None:
        weights_train = convert_or_clone_to_tensor(weights_train, dtype=torch.float16)
        weights_val = convert_or_clone_to_tensor(weights_val, dtype=torch.float16)
        weights_test = convert_or_clone_to_tensor(weights_test, dtype=torch.float16)

    if dataset_type == "tensor":
        Dataset = TensorDataset
    elif dataset_type == "one_hot":
        Dataset = OneHotDataset
    else:
        raise ValueError("dataset_type must be either 'tensor' or 'one_hot'")

    # Create DataLoader for training, validation, and testing
    if weights is not None and sampler is False:
        train_dataset = Dataset(X_train, y_train, weights_train)
        val_dataset = Dataset(X_val, y_val, weights_val)
        test_dataset = Dataset(X_test, y_test, test_ids, weights_test)
    else:
        train_dataset = Dataset(X_train, y_train)
        val_dataset = Dataset(X_val, y_val)
        test_dataset = Dataset(X_test, y_test, test_ids)

    if sampler:
        train_sampler = init_weighted_sampler(
            train_dataset,
            weights_train,
            num_samples_method="min_weighted",
            sampler="weighted_random" if sampler is True else sampler
        )
        val_sampler = init_weighted_sampler(
            val_dataset,
            weights_val,
            num_samples_method="min_weighted",
            sampler="weighted_random" if sampler is True else sampler
        )
        test_sampler = None
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler == False,  # If sampler is used, shuffle is not needed
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        sampler=train_sampler,
        generator=random_state,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        sampler=val_sampler,
        generator=random_state,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        sampler=test_sampler,
        generator=random_state,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}

def create_predict_data_loader(
    dataset,
    batch_size=64,
    dtype=torch.int8,
    num_workers=0,
    dataset_type="tensor",
):
    """
    Create DataLoader objects for prediction.

    Parameters:
        dataset (numpy.ndarray): Input dataset.
        batch_size (int): Batch size for DataLoader (default is 64).

    Returns:
        DataLoader: DataLoader object for prediction.
    """
    X = convert_or_clone_to_tensor(dataset, dtype=dtype)

    if dataset_type == "tensor":
        Dataset = TensorDataset
    elif dataset_type == "one_hot":
        Dataset = OneHotDataset
    else:
        raise ValueError("dataset_type must be either 'tensor' or 'one_hot'")

    dataset = Dataset(X)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )


class OneHotDataset(TensorDataset):
    """
    A custom dataset class that one-hot encodes the first tensor in the dataset.
    set_num_classes must be called before using this dataset, to set the number of classes.
    """

    def __init__(self, *tensors, flatten=True):
        assert all(
            tensors[0].size(0) == tensor.size(0) for tensor in tensors
        ), "Size mismatch between tensors"
        self.tensors = tensors
        self.flatten = True

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_flatten(self, flatten):
        self.flatten = flatten

    def __getitem__(self, index):
        # one hot the first tensor index and the others as is and then return
        return tuple(
            one_hot_encode(tensor[index], self.num_classes, self.flatten) if i == 0 else tensor[index]
            for i, tensor in enumerate(self.tensors)
        )


def one_hot_encode(seqs, num_classes, flatten=True):
    # get dtype and save it
    dtype = seqs.dtype
    # convert to long
    seqs = seqs.long()
    # one hot encode
    encs = F.one_hot(seqs, num_classes)
    # convert back to original dtype
    encs = encs.to(dtype=dtype)
    # return the tensor flattened
    return encs.flatten() if flatten else encs


def init_weighted_sampler(dataset, weights, num_samples_method="min", sampler="weighted_random"):
    if num_samples_method == "min":
        # Count the occurrences of each class in the dataset
        labels = dataset.tensors[1].numpy()  # Assuming that labels are in the second tensor
        class_counts = Counter(labels)
        # Find the class with the least count
        min_class_count = min(class_counts.values())
        # Calculate the number of unique classes
        num_unique_classes = len(class_counts)

        # Set num_samples to the product of the least count and the number of unique classes
        num_samples = min_class_count * num_unique_classes
    elif num_samples_method == "min_weighted":
        value_counts = pd.Series(weights.numpy()).value_counts()
        # Find the class with the least count
        min_class_count = value_counts.min()
        # Calculate the number of unique classes
        num_unique_classes = len(value_counts)

        # Set num_samples to the product of the least count and the number of unique classes
        num_samples = int(min_class_count * num_unique_classes)

        ## -----------------------------------------------------------
        # Map each unique weight value to an integer label: 0 .. N-1
        # -----------------------------------------------------------
        # 1) Collect the unique weight values from the Series index
        unique_weight_values = value_counts.index.tolist()  # e.g., [0.1, 0.3, 1.0, ...]

        # 2) Create a dictionary mapping each weight value -> new label index
        weight_to_label = {w: i for i, w in enumerate(unique_weight_values)}

        # 3) Convert the original `weights` (one weight per sample) into integer "labels"
        labels = torch.tensor(
            [weight_to_label[w.item()] for w in weights],
            dtype=torch.int16
        )
    else:
        raise ValueError("num_samples_method must be 'min'")

    if sampler == "weighted_random":
        # Create the WeightedRandomSampler using these weights
        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,  # To allow resampling
            generator=get_random_state()
        )
    elif sampler == "label_weighted":
        return LabelWeightedSampler(
            label_weights=unique_weight_values,
            labels=labels,
            num_samples=num_samples,
            replacement=True,  # To allow resampling
            generator=get_random_state()
        )


def convert_or_clone_to_tensor(data, dtype):
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype)
    elif torch.is_tensor(data):
        return data.detach().clone().to(dtype=dtype)
    elif isinstance(data, list):
        return torch.tensor(data, dtype=dtype)
    elif isinstance(data, pd.Series):
        return torch.tensor(data.values, dtype=dtype)
    else:
        raise TypeError(
            "Input data must be a NumPy array, a DataFrame Series, a list or a PyTorch tensor."
        )


def convert_string_list_to_list_of_int_lists(data):
    """
    Converts a list of strings, where each string represents a list of integers,
    into a list of lists of integers.

    Args:
        data (list of str): List of strings, each representing a list of integers.

    Returns:
        list of list of int: List of lists of integers.
    """
    list_data = [ast.literal_eval(sample) for sample in data]

    return list_data


def pad_list_of_lists(
    data,
    max_len,
    pad_value=0,
    convert_to_np=False,
    prepend_single_pad=False,
    append_single_pad=False,
):
    """
    Pads a list of lists with a specified pad value to a maximum length.

    Args:
        data (list of list): List of lists to pad.
        max_len (int): Maximum length to pad each list to.
        pad_value (int): Value to use for padding.

    Returns:
        list of list: Padded list of lists.
    """
    if prepend_single_pad:
        max_len += 1
    if append_single_pad:
        max_len += 1
    if convert_to_np:
        return np.array(
            [
                pad_list(
                    sample,
                    max_len,
                    pad_value,
                    prepend_single_pad,
                    append_single_pad,
                )
                for sample in data
            ]
        )
    return [
        pad_list(sample, max_len, pad_value, prepend_single_pad, append_single_pad)
        for sample in data
    ]


def pad_list(
    data, max_len, pad_value=0, prepend_single_pad=False, append_single_pad=False
):
    """
    Pads a list with a specified pad value to a maximum length.

    Args:
        data (list): List to pad.
        max_len (int): Maximum length to pad the list to.
        pad_value (int): Value to use for padding.

    Returns:
        list: Padded list.
    """
    if prepend_single_pad:
        data = [pad_value] + data
    if append_single_pad:
        data = data + [pad_value]
    return data + [pad_value] * (max_len - len(data))


def get_epoch_dataloaders(dataloader, epoch_size=0):
    if epoch_size == 0:
        return dataloader

    # Access the original dataset from the DataLoader
    train = dataloader["train"].dataset
    val = dataloader["val"].dataset

    # Randomly sample indices from the dataset
    if epoch_size < 1:
        epoch_size = int(len(train) * epoch_size)
    train_inds = np.random.choice(len(train), epoch_size, replace=False)
    val_inds = np.random.choice(
        len(val), int(len(val) * epoch_size / len(train)), replace=False
    )

    # Create a Subset for the sampled indices
    train_set = Subset(train, train_inds)
    val_set = Subset(val, val_inds)

    # Create a new DataLoader for the subset
    # Preserve the original DataLoader's batch size, shuffle, and other parameters as needed
    train_dataloader = DataLoader(
        train_set,
        batch_size=dataloader["train"].batch_size,
        shuffle=True,
        num_workers=dataloader["train"].num_workers,
        pin_memory=dataloader["train"].pin_memory,
        generator=get_random_state(),
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=dataloader["val"].batch_size,
        shuffle=False,
        num_workers=dataloader["val"].num_workers,
        pin_memory=dataloader["val"].pin_memory,
        generator=get_random_state(),
    )

    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": dataloader["test"],
    }


def load_config(config_file_name):
    """
    Load a head configuration file from config directory.

    Parameters:
        config_file_name (str): Name of the head configuration file.

    Returns:
        dict: Loaded configuration.
    """
    # Construct the full path to the configuration file
    config_file_path = f"{config_dir}/{config_file_name}"

    # Load the configuration from the file
    with open(config_file_path, "r") as file:
        config = json.load(file)

    return config


def get_activation_function(name):
    """Returns the activation function based on its name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    elif name == "tanh":
        return nn.Tanh()
    # Add more activation functions as needed
    else:
        raise f"Unsupported activation function: {name}"


def get_wild_type(data_type):
    file = f"{data_dir}/{data_type}"
    wild_type_f = open(f"{file}/wild_type.json")
    wt = json.load(wild_type_f)["wild_type"]
    return wt


def load_tokenizer(model_name):
    if "progen2" in model_name:
        model_file = "progen2"
    elif "bert" in model_name:
        model_file = "proteinbert"
    else:
        raise "Model tokenizer not defined"

    file = f"{plmfit_path}/language_models/{model_file}/tokenizer.json"

    with open(file, "r") as f:
        return Tokenizer.from_str(f.read())


def load_transformer_tokenizer(model_name, tokenizer):
    if "progen2" in model_name:
        tokenizer.post_processor = TemplateProcessing(
            single="<|bos|> $A <|eos|>",
            pair="<|bos|> $A <|eos|> <|bos|> $B:1 <|eos|>:1",
            special_tokens=[
                ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
                ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
            ],
        )
        tokenizer = PreTrainedTokenizerFast(
            bos_token="<|bos|>",
            eos_token="<|eos|>",
            pad_token="<|pad|>",
            tokenizer_object=tokenizer,
        )
        return tokenizer
    elif "bert" in model_name:
        tokenizer.post_processor = TemplateProcessing(
            single="<cls> $A <sep>",
            pair="<cls> $A <sep> $B:1 <sep>:1",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<sep>", tokenizer.token_to_id("<sep>")),
            ],
        )
        tokenizer = PreTrainedTokenizerFast(
            cls_token="<cls>",
            sep_token="<sep>",
            pad_token="<pad>",
            mask_token="<mask>",
            unk_token="<unk>",
            tokenizer_object=tokenizer,
        )
        return tokenizer
    elif "esm" in model_name:
        return tokenizer
    else:
        raise "Transformer tokenizer not supported (yet)"


def blosum62_encode(sequences, pad_to_length, logger=None):
    # Load the BLOSUM62 matrix from your custom library
    BLOSUM62 = bl.BLOSUM(62)
    encoded_sequences = []
    i = 0
    for seq in sequences:
        i += 1
        encoded_seq = []
        for acid in seq:
            # Fetch the BLOSUM62 row for the current amino acid
            row = BLOSUM62[acid]
            # Extract scores for the sequence from the row corresponding to each amino acid
            encoded_row = [
                row.get(aa, 0) for aa in seq
            ]  # default to 0 if pair not found

            # Convert list to torch tensor and pad the encoded row to ensure all rows have the same number of columns
            encoded_row = torch.tensor(encoded_row, dtype=torch.int8)
            if encoded_row.size(0) < pad_to_length:
                encoded_row = torch.nn.functional.pad(
                    encoded_row,
                    (0, pad_to_length - encoded_row.size(0)),
                    mode="constant",
                    value=0,
                )

            encoded_seq.append(encoded_row)

        # Stack all rows to create a 2D tensor for each sequence
        encoded_seq = torch.stack(encoded_seq)

        # Pad the encoded sequence if it has fewer rows than `pad_to_length`
        if encoded_seq.size(0) < pad_to_length:
            padding = torch.zeros((pad_to_length - encoded_seq.size(0), pad_to_length))
            encoded_seq = torch.cat((encoded_seq, padding), dim=0)

        if logger is not None and i % 1000 == 0:
            logger.log(f"Encoded sequence {i}")

        encoded_sequences.append(encoded_seq)

    # Stack all sequence tensors to create a 3D tensor for the batch
    return torch.stack(encoded_sequences)

def categorical_encode(
    seqs,
    tokenizer,
    max_len,
    add_bos=False,
    add_eos=False,
    logger=None,
    model_name="progen2",
    progress_interval=100000,
):
    """
    Encodes a list of sequences in chunks to avoid high memory usage.
    
    Args:
        seqs (List[str]): The sequences to encode.
        tokenizer: The tokenizer object with `encode` and `get_vocab` methods.
        max_len (int): Maximum length of the core sequence (excluding optional BOS/EOS).
        add_bos (bool): If True, add a BOS token (depends on model_name).
        add_eos (bool): If True, add an EOS token (depends on model_name).
        logger (Optional): If provided, used for logging messages.
        model_name (str): Name of the model ("progen2", "bert", "esm", ...).
        progress_interval (int): How often to log the progress in each chunk.
        chunk_size (int): Number of sequences to process in one parallel chunk.

    Returns:
        torch.Tensor: A tensor of shape (len(seqs), internal_max_len).
    """

    # --- Initial Logging ---
    total_seqs = len(seqs)
    if logger is not None:
        logger.log(f"Initiating categorical encoding for {total_seqs} sequences.")
        logger.log(f"Memory needed for encoding (approx): {total_seqs * max_len * 4}B")

    # Determine the internal max length depending on the model
    if "esm" in model_name:
        internal_max_len = max_len + 2  # ESM adds <cls> and <eos> automatically
    else:
        internal_max_len = max_len + int(add_bos) + int(add_eos)

    # Identify the correct padding token
    if "progen2" in model_name:
        pad_token = tokenizer.get_vocab()["<|pad|>"]
    elif "bert" in model_name:
        pad_token = tokenizer.get_vocab()["<pad>"]
    elif "esm2" in model_name:
        pad_token = tokenizer.get_vocab()["<pad>"]
    elif "esmc" in model_name:
        pad_token = tokenizer.pad_token_id
    else:
        raise ValueError("Model tokenizer not defined")

    # Pre-allocate the final tensor
    seq_tokens = pad_token * torch.ones((total_seqs, internal_max_len), dtype=torch.int8)

    # Decide whether to run in parallel
    n_cpus = os.cpu_count() or 1
    use_parallel = False # TODO enable parallel encoding
    if logger is not None:
        if use_parallel:
            logger.log(f"Running parallel encoding with up to {n_cpus} workers.")
        else:
            logger.log("Only 1 CPU core detected. Will run in serial mode.")

    # Track how many sequences have been processed so far
    processed_so_far = 0

    # --- Serial Mode ---
    for local_i, seq in enumerate(seqs):
        idx = local_i
        encoded_seq = encode_sequence(seq, tokenizer, max_len, add_bos, add_eos, model_name)
        seq_len = len(encoded_seq)
        seq_tokens[idx, :seq_len] = encoded_seq

        processed_so_far += 1
        if logger is not None and processed_so_far % progress_interval == 0:
            logger.log(f"Encoded {processed_so_far}/{total_seqs} sequences...")

    # --- Final Logging ---
    if logger is not None and total_seqs > 0:
        logger.log(f"First sequence tokens: {seq_tokens[0].tolist()}")
        logger.log("Categorical encoding finished")

    return seq_tokens

def chunkify(data, chunk_size):
    """Generator that yields (start_index, sublist) for each chunk."""
    for i in range(0, len(data), chunk_size):
        yield i, data[i : i + chunk_size]

def encode_sequence(seq, tokenizer, max_len, add_bos, add_eos, model_name):
    """Helper function to process encoding of a single sequence."""
    if "progen2" in model_name:
        internal_max_len = max_len + int(add_bos) + int(add_eos)
        encoded_seq_ids = tokenizer.encode(seq, add_special_tokens=False).ids

        sequence = []
        if add_bos:
            sequence.append(tokenizer.get_vocab()["<|bos|>"])
        sequence.extend(encoded_seq_ids[:max_len])
        if add_eos:
            sequence.append(tokenizer.get_vocab()["<|eos|>"])

        truncated_sequence = sequence[:internal_max_len]
        return torch.tensor(truncated_sequence, dtype=torch.long)

    elif "bert" in model_name:
        internal_max_len = max_len + int(add_bos) + int(add_eos)
        encoded_seq_ids = tokenizer.encode(seq, add_special_tokens=False).ids

        sequence = []
        if add_bos:
            sequence.append(tokenizer.get_vocab()["<cls>"])
        sequence.extend(encoded_seq_ids[:max_len])
        if add_eos:
            sequence.append(tokenizer.get_vocab()["<sep>"])

        truncated_sequence = sequence[:internal_max_len]
        return torch.tensor(truncated_sequence, dtype=torch.int8)

    elif "esm2" in model_name:
        # ESMTokenizer automatically adds <cls> and <eos> tokens
        # We'll ensure the final tensor doesn't exceed max_len+2
        tok_seq = torch.tensor(tokenizer.encode(seq))
        return tok_seq[: max_len + 2]
    elif "esmc" in model_name:
        # ESMTokenizer automatically adds <cls> and <eos> tokens
        # We'll ensure the final tensor doesn't exceed max_len+2
        tok_seq = encoding.tokenize_sequence(seq, tokenizer, add_special_tokens=True)
        return tok_seq[: max_len + 2]
    else:
        raise ValueError("Model tokenizer not defined")

def get_parameters(model, print_w_mat=False, logger=None):
    s = 0
    c = 0
    for name, p in model.named_parameters():

        c += 1
        if print_w_mat:
            print(f" {name} size : {p.shape} trainable:{p.requires_grad}")
        if logger is not None:
            logger.log(f" {name} size : {p.shape} trainable:{p.requires_grad}")
        s += p.numel()

    return s


def check_module_states(model, logger=None):
    for name, module in model.named_modules():
        state = "Train" if module.training else "Eval"
        message = f"Module: {name} State: {state}"

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
        print(
            f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {trainable_percentage:.3f}"
        )


def set_trainable_parameters(model, ft="all"):

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


def set_modules_to_train_mode(model, module_name="all"):
    """
    This function iterates through all modules in the given model.
    If it finds a module that has 'module_name' in its name,
    it sets that module to training mode.
    """
    for name, module in model.named_modules():
        # Identify modules by checking if 'module_name' is in their name.
        if module_name in name or module_name == "all":
            module.train()  # Set the identified 'module_name' module to training mode

    # Note: This does not change the global training/evaluation mode of the model,
    # but specifically sets 'module_name' modules to training mode.


def set_trainable_layers(model: nn.Module, layers_to_train: list):
    """
    Sets the specified layers to trainable and freezes all other layers.
    Disables dropout for all modules in the frozen layers.

    Args:
    - model (nn.Module): The model to modify.
    - layers_to_train (list): List of layer indices to set as trainable.
    """
    # Iterate over each layer in the model
    for name, layer in model.named_modules():
        # Extract the layer index from the name (assuming standard naming conventions)
        layer_index = None
        # 'layer' for BERT based PLMs, 'h' for ProGen
        if "layer" in name or "h" in name:
            try:
                layer_index = int(name.split("layer")[1].split(".")[1])
            except (ValueError, IndexError):
                try:
                    layer_index = int(name.split("h")[1].split(".")[1])
                except:
                    continue
        # If the layer index is in the list of layers to train
        if layer_index is not None:
            if layer_index in layers_to_train:
                # Set the layer to training mode
                layer.train()
                # Set requires_grad to True for all parameters in this layer
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                # Freeze the layer by setting requires_grad to False
                layer.eval()
                for param in layer.parameters():
                    if hasattr(param, "requires_grad"):
                        param.requires_grad = False
                # Disable dropout for this layer
                layer.apply(disable_dropout)


def read_fasta(file_path):
    sequences = {}
    current_sequence_id = None
    current_sequence = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # This line contains the sequence identifier
                if current_sequence_id is not None:
                    sequences[current_sequence_id] = "".join(current_sequence)
                current_sequence_id = line[1:]
                current_sequence = []
            else:
                # This line contains sequence data
                if current_sequence_id is not None:
                    current_sequence.append(line)

    # Add the last sequence to the dictionary
    if current_sequence_id is not None:
        sequences[current_sequence_id] = "".join(current_sequence)

    return sequences


def log_model_info(
    log_file_path, data_params, model_params, training_params, eval_metrics
):
    with open(log_file_path, "w") as log_file:
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


def print_gpu_utilization(memory_usage, device="cuda"):
    if "cuda" in device:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used // 1024**2
    else:
        memory = psutil.virtual_memory()
        return memory.used // 1024**2  # Convert from Bytes to Megabytes


def get_loss_weights(labels):
    pos = torch.sum(labels == 1)
    neg = torch.sum(labels == 0)
    obs = pos + neg
    return (neg / obs, pos / obs)


def print_cpu_utilization():
    # Get CPU utilization percentage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Utilization: {cpu_percent}%")

    # Get memory usage
    memory = psutil.virtual_memory()
    memory_used = memory.used // 1024**2  # Convert from Bytes to Megabytes
    memory_total = memory.total // 1024**2  # Convert from Bytes to Megabytes
    print(f"Memory Used: {memory_used} MB")
    print(f"Total Memory: {memory_total} MB")

    return cpu_percent, memory_used


def adjust_config_to_int(
    config,
    int_keys=[
        ("training_parameters", "batch_size"),
        ("architecture_parameters", "hidden_dim"),
    ],
):
    """
    Adjusts specific keys in a nested config dictionary to have integer values.

    Parameters:
    - config (dict): The nested configuration dictionary produced by the optimization algorithm.
    - int_keys (list of tuples): A list of tuples where each tuple contains the path to a key within the config that should have its value rounded to the nearest integer.

    Returns:
    - dict: The adjusted configuration dictionary with specified keys rounded to integers.
    """
    adjusted_config = (
        config.copy()
    )  # Make a copy to avoid modifying the original config

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


def find_mutation_positions(seq, ref, padding_id=None):
    """
    Returns a list of positions where the sequence differs from the reference.
    """
    return [i for i, (s, r) in enumerate(zip(seq, ref)) if s != r]


def init_plm(model_name, logger, task="regression"):
    model = None
    supported_progen2 = ["progen2-small", "progen2-medium", "progen2-xlarge"]
    supported_ESM = [
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D",
    ]
    # supported_Ankh = ["ankh-base", "ankh-large", "ankh2-large"]
    supported_Proteinbert = ["proteinbert"]
    supported_ESMC = [
        "esmc_300m",
        "esmc_600m",
    ]

    if "progen" in model_name:
        assert model_name in supported_progen2, "Progen version is not supported"
        model = ProGenFamily(model_name, logger, task)

    elif "esm2" in model_name:
        assert model_name in supported_ESM, "ESM version is not supported"
        model = ESMFamily(model_name, logger, task)
    # elif "ankh" in model_name:
    #     assert model_name in supported_Ankh, "Ankh version is not supported"
    #     model = AnkhFamily(model_name)
    # elif "antiberty" in model_name:
    #     model = Antiberty()
    elif "proteinbert" in model_name:
        assert (
            model_name in supported_Proteinbert
        ), "ProteinBERT version is not supported"
        model = ProteinBERTFamily(logger, task)
    elif "esmc" in model_name:
        assert model_name in supported_ESMC, "ESMC version is not supported"
        model = ESMCFamily(model_name, logger, task)
    else:
        raise "PLM not supported"

    assert model != None, "Model is not initialized"
    return model


def masking_collator(
    tokenizer, features, mlm_probability=0.15, mutation_boost_factor=6.66
):
    """
    Create masked inputs for MLM from tokenized data, with boosted masking probability for specified mutations.

    Args:
    - tokenizer: The tokenizer used for tokenizing the data.
    - features (dict): A dictionary containing the encoded sequences.
    - mlm_probability (float): The base probability of each token being masked.
    - mutation_boost_factor (float): Factor to boost the mlm_probability for mutated positions.

    Returns:
    - dict: A dictionary containing the masked input_ids, attention_masks, and labels for MLM.
    """
    input_ids = features["input_ids"]

    labels = input_ids.clone()  # Prepare labels for MLM

    # Base probability matrix for masking
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # TODO fix for mut mask
    if "mutation_mask" in features:
        probability_matrix += features["mutation_mask"] * (
            mlm_probability * (mutation_boost_factor - 1)
        )

    # Set probability for special tokens to 0 to avoid masking
    special_tokens_mask = features.get(
        "special_tokens_mask", torch.zeros_like(input_ids).bool()
    )
    if (
        isinstance(special_tokens_mask, torch.Tensor)
        and special_tokens_mask.dtype != torch.bool
    ):
        special_tokens_mask = special_tokens_mask.bool()
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    random_state = get_random_state()

    # Create mask array
    masked_indices = torch.bernoulli(probability_matrix, generator=random_state).bool()

    # Apply 80-10-10 masking strategy:
    # 80% MASK
    indices_replaced_with_mask = (
        torch.bernoulli(torch.full(labels.shape, 0.8), generator=random_state).bool()
        & masked_indices
    )
    input_ids[indices_replaced_with_mask] = tokenizer.mask_token_id

    # 10% random token
    indices_replaced_with_random = (
        torch.bernoulli(torch.full(labels.shape, 0.1), generator=random_state).bool()
        & masked_indices
        & ~indices_replaced_with_mask
    )
    random_tokens = torch.randint(low=0, high=tokenizer.vocab_size, size=labels.shape, generator=random_state)
    input_ids[indices_replaced_with_random] = random_tokens[
        indices_replaced_with_random
    ]

    # 10% unchanged - no action needed since indices are not masked

    labels[~masked_indices] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": features["attention_mask"],
        "labels": labels,
    }


class MaskedLMDataset(Dataset):
    def __init__(
        self, encodings, tokenizer, mlm_probability, mutation_boost_factor=6.66
    ):
        self.encodings = encodings
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mutation_boost_factor = mutation_boost_factor

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        masked_inputs = masking_collator(
            self.tokenizer, item, self.mlm_probability, self.mutation_boost_factor
        )
        return masked_inputs


def create_mlm_data_loaders(
    data,
    tokenizer,
    batch_size=16,
    mlm_probability=0.15,
    mutation_boost_factor=6.66,
    split_ratios=(0.7, 0.15, 0.15),
):
    dataset = MaskedLMDataset(data, tokenizer, mlm_probability, mutation_boost_factor)

    # Determine split sizes
    train_size = int(len(dataset) * split_ratios[0])
    val_size = int(len(dataset) * split_ratios[1])
    test_size = len(dataset) - train_size - val_size

    random_state = get_random_state()

    # Randomly split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=random_state
    )

    # Create data loaders for each split
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=random_state
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=random_state)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=random_state)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def data_pipeline(dataset, split=None, weights=None, sampler=None, dev=False):
    # Load dataset
    dataset = load_dataset(dataset)

    # For development purposes, we can sample the dataset to speed up the process
    if dev:
        dataset = dataset[:100000]

    # This checks if args.split is set to 'sampled' and if 'sampled' is not in data, or if args.split is not a key in data.
    split = (
        None if split == "sampled" and "sampled" not in dataset else dataset.get(split)
    )

    # If weights are provided, load them
    weights = None if weights is None else dataset.get(weights)

    # If a sampler is provided, load it
    sampler = False if sampler is None else sampler

    return dataset, split, weights, sampler

def suggest_number_of_type(trial: Trial, name, min, max, type):
    if type == "int":
        return trial.suggest_int(name, min, max)
    elif type == "float":
        return trial.suggest_float(name, min, max)
    else:
        raise ValueError("Type of hyperparameter not supported")