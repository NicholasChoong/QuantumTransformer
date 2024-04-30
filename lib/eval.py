import torch

from lib.binary_accuracy import binary_accuracy

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

from config import dev


def evaluate(model: Module, dataloader: DataLoader, criterion: _Loss, max_seq_len: int):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = torch.LongTensor(batch[1].T).to(dev)
            if inputs.size(1) > max_seq_len:
                inputs = inputs[:, :max_seq_len]
            predictions = model(inputs).squeeze(1)

            label = batch[0].to(dev)
            # label = label.unsqueeze(1)
            loss = criterion(predictions, label)
            # loss = F.nll_loss(predictions, label)
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader.dataset), epoch_acc / len(dataloader.dataset)
