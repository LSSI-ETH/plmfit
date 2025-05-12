# plmfit/shared_utils/random_state.py
import numpy as np
import torch
import random

# Initialize the global random state
seed = 42
GLOBAL_RANDOM_STATE = np.random.RandomState(seed=seed)

def get_random_state():
    return GLOBAL_RANDOM_STATE

def set_seed(seed):

    global GLOBAL_RANDOM_STATE
    GLOBAL_RANDOM_STATE = torch.Generator()
    GLOBAL_RANDOM_STATE.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_numpy_random_state():
    np.random.RandomState(seed=seed)
