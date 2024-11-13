import torch
import torch.nn as nn
import numpy as np
import scipy
from dataset import *

# scale the tensor to have dummy position equal to 1
def affine_coord(tensor, dummy_pos=None):
    # tensor: B*T*K
    if dummy_pos is not None:
        return tensor / tensor[..., dummy_pos].unsqueeze(-1)
    else:
        return tensor

# so(n) Lie algebra
def so(n):
    L = np.zeros((n*(n-1)//2, n, n))
    k = 0
    for i in range(n):
        for j in range(i):
            L[k,i,j] = 1
            L[k,j,i] = -1
            k += 1
    return torch.tensor(L, dtype=torch.float32)
