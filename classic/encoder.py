from torch import Tensor
import torch.nn as nn

from classic.feedforward import FeedForward
from classic.multihead import MultiHeadedAttention


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        mask: Tensor | None = None,
    ):
        super(Encoder, self).__init__()

        self.attn = MultiHeadedAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, mask=mask
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        attn_output: Tensor = self.attn(x)
        x = self.norm1(attn_output + x)
        x = self.dropout1(x)

        ff_output: Tensor = self.ffn(x)
        x = self.norm2(ff_output + x)
        x = self.dropout2(x)

        return x
