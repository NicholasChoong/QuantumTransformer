import torch

import numpy as np

from sklearn.metrics import roc_auc_score

from .binary_acc import binary_accuracy

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR

from accelerate import Accelerator


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: _Loss,
    scheduler: StepLR,
    max_seq_len: int,
    batch_size: int,
    progress_bar,
    accelerator: Accelerator,
):
    epoch_loss = []
    epoch_acc = []
    epoch_true = []
    epoch_pred = []

    model.train()
    for i, (label, input) in enumerate(dataloader):
        optimizer.zero_grad()
        print(f"Model is on device: {next(model.parameters()).device}")
        predictions = model(input).squeeze(1)

        predictions, label = accelerator.gather_for_metrics((predictions, label))

        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        accelerator.backward(loss)
        optimizer.step()

        epoch_loss.append(loss.item())
        epoch_acc.append(acc.item())
        epoch_true.extend(label.tolist())
        epoch_pred.extend(predictions.sigmoid().tolist())

        accelerator.wait_for_everyone()
        progress_bar.update(dataloader.batch_size)

    # print(epoch_loss, epoch_acc, len(dataloader.dataset))

    epoch_auc = 100.0 * roc_auc_score(epoch_true, epoch_pred, multi_class="ovr")
    scheduler.step()
    # divide the total loss by the total number of batches per epoch
    return np.mean(epoch_loss), np.mean(epoch_acc), epoch_auc
