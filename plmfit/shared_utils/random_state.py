# plmfit/shared_utils/random_state.py
import numpy as np
import torch
import random

# Global variable for numpy
GLOBAL_RANDOM_STATE = None
SEED = 42  # Default seed value

def set_seed(seed):
    global GLOBAL_RANDOM_STATE
    global SEED
    SEED = seed
    np.random.seed(seed)
    GLOBAL_RANDOM_STATE = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)

def get_random_state():
    return GLOBAL_RANDOM_STATE

def get_seed():
    return SEED
