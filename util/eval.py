import torch
import numpy as np

from lib.binary_accuracy import binary_accuracy

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

from config import dev


def evaluate(model: Module, dataloader: DataLoader, criterion: _Loss, max_seq_len: int):

    epoch_loss = []
    epoch_acc = []

    model.eval()
    with torch.no_grad():
        for i, (lab, text) in enumerate(dataloader):
            inputs = torch.LongTensor(text.T).to(dev)
            if inputs.size(1) > max_seq_len:
                inputs = inputs[:, :max_seq_len]
            predictions = model(inputs).squeeze(1)

            label = lab.to(dev)
            # label = label.unsqueeze(1)
            loss = criterion(predictions, label)
            # loss = F.nll_loss(predictions, label)
            acc = binary_accuracy(predictions, label)

            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())

    # divide the total loss by the total number of batches per epoch
    return np.mean(epoch_loss), np.mean(epoch_acc)
