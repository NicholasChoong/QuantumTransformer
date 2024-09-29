import math

import torch
import torch.nn as nn

from torch import Tensor

from config import dev


class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=128, device=dev):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(0, 1)
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        x = self.dropout(x)
        return x.transpose(0, 1)
