import torch

import numpy as np

from sklearn.metrics import roc_auc_score

from .binary_acc import binary_accuracy

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR


from config import dev


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: _Loss,
    scheduler: StepLR,
    max_seq_len: int,
    batch_size: int,
    progress_bar,
):
    epoch_loss = []
    epoch_acc = []
    epoch_true = []
    epoch_pred = []

    param_updates = {}
    param_grads = {}

    model.train()
    for i, (lab, input) in enumerate(dataloader):
        optimizer.zero_grad()
        input = input.to(dev)
        model.to(dev)
        predictions = model(input).squeeze(1)

        label = lab.to(dev)
        # label = label.unsqueeze(1)
        loss = criterion(predictions, label)
        # loss = F.nll_loss(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if name not in param_updates:
                param_updates[name] = []
            if name not in param_grads:
                param_grads[name] = []

            param_updates[name].append(param.data.norm().item())
            if param.grad is not None:
                param_grads[name].append(param.grad.norm().item())

        epoch_loss.append(loss.item())
        epoch_acc.append(acc.item())
        epoch_true.extend(label.tolist())
        epoch_pred.extend(predictions.sigmoid().tolist())

        progress_bar.update(1)

        # print(epoch_loss, epoch_acc, len(dataloader.dataset))

    avg_param_updates = {
        name: np.mean(updates) for name, updates in param_updates.items()
    }
    avg_param_grads = {name: np.mean(grads) for name, grads in param_grads.items()}

    epoch_auc = 100.0 * roc_auc_score(epoch_true, epoch_pred, multi_class="ovr")
    scheduler.step()
    # divide the total loss by the total number of batches per epoch
    return (
        100.0 * np.mean(epoch_loss),
        100.0 * np.mean(epoch_acc),
        epoch_auc,
        avg_param_updates,
        avg_param_grads,
    )
