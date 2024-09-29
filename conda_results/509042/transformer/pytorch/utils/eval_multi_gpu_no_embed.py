import torch
import numpy as np

from sklearn.metrics import roc_auc_score

from .binary_acc import binary_accuracy

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

from accelerate import Accelerator


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: _Loss,
    max_seq_len: int,
    accelerator: Accelerator,
):
    epoch_loss = []
    epoch_acc = []
    epoch_true = []
    epoch_pred = []

    model.eval()
    with torch.no_grad():
        for i, (label, input) in enumerate(dataloader):
            predictions = model(input).squeeze(1)

            predictions, label = accelerator.gather_for_metrics((predictions, label))

            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())
            epoch_true.extend(label.tolist())
            epoch_pred.extend(predictions.sigmoid().tolist())

    epoch_auc = 100.0 * roc_auc_score(epoch_true, epoch_pred, multi_class="ovr")

    # divide the total loss by the total number of batches per epoch
    return np.mean(epoch_loss), np.mean(epoch_acc), epoch_auc
