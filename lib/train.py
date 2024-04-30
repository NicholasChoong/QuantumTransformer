import torch


from lib.binary_accuracy import binary_accuracy

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer


from config import dev


def train(
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: _Loss,
    max_seq_len: int,
):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for i, batch in enumerate(dataloader):

        optimizer.zero_grad()
        inputs = torch.LongTensor(batch[1].T).to(dev)
        if inputs.size(1) > max_seq_len:
            inputs = inputs[:, :max_seq_len]
        model.to(dev)
        predictions = model(inputs).squeeze(1)

        label = batch[0].to(dev)
        # label = label.unsqueeze(1)
        loss = criterion(predictions, label)
        # loss = F.nll_loss(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader.dataset), epoch_acc / len(dataloader.dataset)
