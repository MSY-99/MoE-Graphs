import os
import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)



def scatter_add_torch(row_idx: torch.Tensor, src: torch.Tensor, dim_size: int):
    """
    Implementation of scatter_add using built-in PyTorch operations (fixed at dim=0).

    Args:
        row_idx: [E] Target indices for the scattering operation.
        src:     [E, F] Source values to be accumulated.
        dim_size: N (Total number of nodes).
    """
    out = src.new_zeros(dim_size, src.size(-1))
    out.index_add_(0, row_idx, src)
    return out


