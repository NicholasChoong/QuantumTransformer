import tensorflow as tf
from typing import Literal
from tensorflow.keras import layers
from ._tensorcircuit import QuantumLayer as tc_QuantumLayer
from ._pennylane import QuantumLayer as qml_QuantumLayer


class FeedForward(tf.keras.layers.Layer):
    """
    The input `x` that is passed to the `call` method is a tensor of shape (batch_size, sequence_length, embedding_dimension),
    and the feed-forward layer is applied position-wise to the last dimension (the embedding dimension) for each batch
    and for each position in the sequence.
    """

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int,
        n_qlayers=1,
        dropout=0.1,
        batch=True,
        circuit_type: Literal["pennylane", "tensorcircuit"] = "tensorcircuit",
        q_device="default.qubit",
    ):
        super(FeedForward, self).__init__()
        self.ffn_dim = n_qubits
        self.batch = batch

        # Choose the correct quantum layer based on the circuit type
        if circuit_type == "pennylane":
            QuantumLayer = qml_QuantumLayer
        else:
            QuantumLayer = tc_QuantumLayer

        # Define the classical layers in TensorFlow
        self.linear_1 = layers.Dense(self.ffn_dim)
        self.linear_2 = layers.Dense(embed_dim)

        # Quantum layer
        self.vqc = QuantumLayer(
            n_qubits=n_qubits, n_qlayers=n_qlayers, q_device=q_device
        )

        # Activation and dropout layers
        self.gelu = layers.Activation(tf.nn.gelu)
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        x = self.linear_1(x)
        x = self.vqc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
