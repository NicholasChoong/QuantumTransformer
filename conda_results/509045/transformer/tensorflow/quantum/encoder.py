import tensorflow as tf
from tensorflow.keras import layers
from typing import Literal

from .multihead import MultiHeadedAttention
from .feedforward import FeedForward


class Encoder(tf.keras.Model):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        mask=None,
        n_qubits_transformer=0,
        n_qubits_ffn=0,
        n_qlayers=1,
        batch=True,
        circuit_type: Literal["pennylane", "tensorcircuit"] = "tensorcircuit",
        q_device="default.qubit",
    ):
        super(Encoder, self).__init__()

        self.n_qubits_transformer = n_qubits_transformer
        self.n_qubits_ffn = n_qubits_ffn
        self.n_qlayers = n_qlayers

        # Multi-head attention with quantum integration
        self.attn = MultiHeadedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            n_qubits=n_qubits_transformer,
            n_qlayers=n_qlayers,
            batch=batch,
            circuit_type=circuit_type,
            q_device=q_device,
        )
        # Layer normalization
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout
        self.dropout1 = layers.Dropout(dropout)

        # FeedForward network with quantum integration
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            n_qubits=n_qubits_ffn,
            n_qlayers=n_qlayers,
            dropout=dropout,
            batch=batch,
            circuit_type=circuit_type,
            q_device=q_device,
        )
        # Layer normalization
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x):
        # Multi-head attention with residual connection and normalization
        attn_output = self.attn(x, x, x)
        x = self.norm1(attn_output + x)
        x = self.dropout1(x)

        # FeedForward network with residual connection and normalization
        ff_output = self.ffn(x)
        x = self.norm2(ff_output + x)
        x = self.dropout2(x)

        return x
