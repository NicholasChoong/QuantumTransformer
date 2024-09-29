import tensorflow as tf
from typing import Literal
from .utils.clone import get_clones

from .classic.encoder import Encoder
from .quantum.encoder import Encoder as QuantumEncoder


class TextClassifier(tf.keras.Model):
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        num_heads: int,
        num_blocks: int,
        num_classes: int,
        vocab_size: int,
        input_dim: int = 768,
        ffn_dim=32,
        dropout=0.1,
        n_qubits_transformer=0,
        n_qubits_ffn=0,
        n_qlayers=1,
        batch=True,
        circuit_type: Literal["pennylane", "tensorcircuit"] = "tensorcircuit",
        q_device="default.qubit",
    ):
        super(TextClassifier, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        self.squeeze = tf.keras.layers.Dense(embed_dim)

        print(f"++ There will be {num_blocks} transformer blocks")

        if n_qubits_transformer > 0:
            print(
                f"++ Transformer will use {n_qubits_transformer} qubits and {n_qlayers} q layers"
            )
            if n_qubits_ffn > 0:
                print(f"The feed-forward head will use {n_qubits_ffn} qubits")
            else:
                print("The feed-forward head will be classical")

            if circuit_type == "pennylane":
                print(f"Using PennyLane quantum device {q_device}")
            else:
                print("Using TensorCircuit")

            self.transformers = [
                QuantumEncoder(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer=n_qubits_transformer,
                    n_qubits_ffn=n_qubits_ffn,
                    n_qlayers=n_qlayers,
                    batch=batch,
                    circuit_type=circuit_type,
                    q_device=q_device,
                )
                for _ in range(num_blocks)
            ]

        else:
            self.transformers = [
                Encoder(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)
            ]

        self.class_logits = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training=False):

        x = self.squeeze(x)

        for transformer in self.transformers:
            x = transformer(x)

        x = tf.reduce_mean(x, axis=1)  # global average pooling, works in 1D
        x = self.dropout(x, training=training)
        x = self.class_logits(x)
        return x
