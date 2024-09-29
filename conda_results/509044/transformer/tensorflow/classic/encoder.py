import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from .feedforward import FeedForward
from .multihead import MultiHeadedAttention


class Encoder(keras.Model):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, mask=None):
        super(Encoder, self).__init__()

        self.attn = MultiHeadedAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, mask=mask
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)

        self.ffn = FeedForward(embed_dim, ffn_dim, dropout=dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x):
        attn_output = self.attn(x, x, x)
        x = self.norm1(attn_output + x)
        x = self.dropout1(x)

        ff_output = self.ffn(x)
        x = self.norm2(ff_output + x)
        x = self.dropout2(x)

        return x
