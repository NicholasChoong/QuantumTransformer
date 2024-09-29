import tensorflow as tf
from tensorflow.keras import layers
from typing import Literal
from ..classic.attention import attention

from ._tensorcircuit import QuantumLayer as tc_QuantumLayer
from ._pennylane import QuantumLayer as qml_QuantumLayer


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout=0.1,
        mask=None,
        use_bias=False,
        n_qubits=4,
        n_qlayers=1,
        batch=True,
        circuit_type: Literal["pennylane", "tensorcircuit"] = "tensorcircuit",
        q_device="default.qubit",
    ):
        super(MultiHeadedAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask = mask
        # We assume dim_v always equals dim_k
        self.dim_k = embed_dim // num_heads  # projection dimensions

        # print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        # Choose the appropriate quantum layer (PennyLane or TensorCircuit)
        if circuit_type == "pennylane":
            QuantumLayer = qml_QuantumLayer
        else:
            QuantumLayer = tc_QuantumLayer

        # Quantum layers for query, key, and value projections
        self.k_linear = QuantumLayer(n_qubits, n_qlayers, q_device=q_device)
        self.q_linear = QuantumLayer(n_qubits, n_qlayers, q_device=q_device)
        self.v_linear = QuantumLayer(n_qubits, n_qlayers, q_device=q_device)
        # Quantum layer to combine the heads
        self.combine_heads = QuantumLayer(n_qubits, n_qlayers, q_device=q_device)

        self.attn_weights = None
        self.dropout = layers.Dropout(rate=dropout)

    def call(self, query, key, value):
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]
        embed_dim = tf.shape(query)[2]

        assert (
            embed_dim == self.embed_dim
        ), f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        Q, K, V = [
            tf.reshape(linear(x), (batch_size, seq_len, self.num_heads, self.dim_k))
            for linear, x in zip(
                [self.k_linear, self.q_linear, self.v_linear], [key, query, value]
            )
        ]

        Q, K, V = [tf.transpose(tensor, perm=[0, 2, 1, 3]) for tensor in [Q, K, V]]

        # Apply attention (assuming a custom attention function exists)
        x, self.attn_weights = attention(Q, K, V, dropout=self.dropout)

        # Reshape the output to the original dimensions
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, seq_len, self.embed_dim))

        # Combine the heads using the quantum layer
        output = self.combine_heads(x)

        return output
