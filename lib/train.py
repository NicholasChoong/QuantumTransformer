import torch

from lib.binary_accuracy import binary_accuracy

from torchtext.data import BucketIterator
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer


def train(
    model: Module,
    iterator: BucketIterator,
    optimizer: Optimizer,
    criterion: _Loss,
    max_seq_len: int,
):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        inputs = torch.LongTensor(batch.text[0])
        if inputs.size(1) > max_seq_len:
            inputs = inputs[:, :max_seq_len]
        predictions = model(inputs).squeeze(1)

        label = batch.label - 1
        # label = label.unsqueeze(1)
        loss = criterion(predictions, label)
        # loss = F.nll_loss(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
