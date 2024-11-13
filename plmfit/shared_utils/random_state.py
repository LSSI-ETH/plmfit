# plmfit/shared_utils/random_state.py
import numpy as np
import torch

# Initialize the global random state
GLOBAL_RANDOM_STATE = np.random.RandomState(seed=42)


def get_random_state():
    return GLOBAL_RANDOM_STATE

def set_seed(seed):
    global GLOBAL_RANDOM_STATE
    GLOBAL_RANDOM_STATE = torch.Generator()
    GLOBAL_RANDOM_STATE.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
