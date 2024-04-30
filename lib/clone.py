from copy import deepcopy
from torch.nn import Module, ModuleList


def get_clones(module: Module, N: int):
    return ModuleList([deepcopy(module) for _ in range(N)])
