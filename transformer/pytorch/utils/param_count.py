import torch.nn as nn


def count_parameters(model: nn.Module):
    print([p.numel() for p in model.parameters() if p.requires_grad])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
