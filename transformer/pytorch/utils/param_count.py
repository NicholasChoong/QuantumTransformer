import torch.nn as nn


def count_parameters(model: nn.Module):
    print(f"{'Layer Name':<40} {'Number of Parameters'}")
    print("=" * 60)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<40} {param.numel()}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
