from torch.nn import Module


def count_parameters(model: Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
