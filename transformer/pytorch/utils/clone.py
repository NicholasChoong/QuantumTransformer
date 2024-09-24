import torch.nn as nn
from copy import deepcopy


def get_clones(module: nn.Module, N: int):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])
