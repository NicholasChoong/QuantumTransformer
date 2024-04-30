from torch import Tensor
import torch.nn as nn

from classic.attention import attention
from lib.clone import get_clones


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout=0.1,
        mask: Tensor | None = None,
        use_bias=False,
    ):
        super(MultiHeadedAttention, self).__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask = mask
        # We assume dim_v always equals dim_k
        self.dim_k = embed_dim // num_heads  # projection dimensions

        # The linear layers for the query, key, and value projections
        self.linears = get_clones(nn.Linear(embed_dim, embed_dim, bias=use_bias), 3)
        # The linear layer to combine the heads
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.attn_weights: Tensor | None = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ):
        batch_size, seq_len, embed_dim = query.size()
        # 1) Do all the linear projections in batch from embed_dim => num_heads x dim_k
        query, key, value = [
            lin(x).view(batch_size, -1, self.num_heads, self.dim_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x: Tensor
        x, self.attn_weights = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, embed_dim)
        x = self.combine_heads(x)
        return x
