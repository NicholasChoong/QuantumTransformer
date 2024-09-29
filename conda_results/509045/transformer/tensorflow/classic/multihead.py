import tensorflow as tf
from tensorflow.keras import layers

from .attention import attention
from ..utils.clone import get_clones


class MultiHeadedAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1, mask=None, use_bias=False):
        super(MultiHeadedAttention, self).__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask = mask
        self.dim_k = embed_dim // num_heads

        self.linears = [layers.Dense(embed_dim, use_bias=use_bias) for _ in range(3)]
        self.combine_heads = layers.Dense(embed_dim, use_bias=use_bias)
        self.attn_weights = None
        self.dropout = layers.Dropout(rate=dropout)

    def call(self, query, key, value, mask=None):
        batch_size, seq_len, embed_dim = (
            tf.shape(query)[0],
            tf.shape(query)[1],
            tf.shape(query)[2],
        )

        query, key, value = [
            tf.transpose(
                tf.reshape(lin(x), (batch_size, -1, self.num_heads, self.dim_k)),
                perm=(0, 2, 1, 3),
            )
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn_weights = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (batch_size, -1, embed_dim))
        x = self.combine_heads(x)
        return x
