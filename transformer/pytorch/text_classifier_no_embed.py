from typing import Literal
import torch
import torch.nn as nn

from torch import Tensor

from transformer.pytorch.quantum.pennylane.angle_amp import PennyLaneArgs

from .utils.clone import get_clones

from .classic.encoder import Encoder
from .quantum.encoder import Encoder as QuantumEncoder


class TextClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        num_heads: int,
        num_blocks: int,
        num_classes: int,
        vocab_size: int,
        pennylane_args: PennyLaneArgs,
        input_dim: int = 768,
        ffn_dim=32,
        dropout=0.1,
        n_qubits_transformer=0,
        n_qubits_ffn=0,
        n_qlayers=1,
        batch=True,
        pooling_method: Literal["CLS", "MEAN", "MAX", "CONCAT"] = "CLS",
        circuit_type: Literal["pennylane", "tensorcircuit"] = "tensorcircuit",
        encoding_type: Literal["angle_amp", "block"] = "angle_amp",
        q_device="default.qubit",
    ):
        super(TextClassifier, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.pooling_method = pooling_method

        self.squeeze = nn.Linear(input_dim, embed_dim)

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

            self.transformers = nn.ModuleList(
                [
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
                        pennylane_args=pennylane_args,
                        encoding_type=encoding_type,
                    )
                    for _ in range(num_blocks)
                ]
            )

        else:
            self.transformers = nn.ModuleList(
                [Encoder(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.class_logits = nn.Linear(embed_dim, 1)

    def forward(self, x: Tensor):

        x = self.squeeze(x)

        for transformer in self.transformers:
            x = transformer(x)

        # x = x.mean(dim=1)  # global average pooling, works in 1D
        x = self.dropout(x)
        x = self.layer_norm(x)
        # raise Exception(" Stop here ")

        if self.pooling_method == "MEAN":
            # Mean pooling
            x = x.mean(dim=1)
        elif self.pooling_method == "MAX":
            # Max pooling
            x, _ = x.max(dim=1)
        else:
            # CLS token
            x = x[:, 0, :]

        x = self.class_logits(x)
        return x
