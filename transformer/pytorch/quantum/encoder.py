from math import e
from typing import Literal
import torch.nn as nn
from torch import Tensor

from transformer.pytorch.quantum.pennylane.angle_amp import PennyLaneArgs

from .multihead import MultiHeadedAttention
from .feedforward import FeedForward


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        pennylane_args: PennyLaneArgs,
        dropout: float = 0.1,
        mask: Tensor | None = None,
        n_qubits_transformer=0,
        n_qubits_ffn=0,
        n_qlayers=1,
        batch=True,
        circuit_type: Literal["pennylane", "tensorcircuit"] = "pennylane",
        encoding_type: Literal["angle_amp", "block"] = "angle_amp",
        q_device="default.qubit",
    ):
        super(Encoder, self).__init__()

        self.n_qubits_transformer = n_qubits_transformer
        self.n_qubits_ffn = n_qubits_ffn
        self.n_qlayers = n_qlayers

        self.attn = MultiHeadedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            n_qubits=n_qubits_transformer,
            n_qlayers=n_qlayers,
            batch=batch,
            circuit_type=circuit_type,
            q_device=q_device,
            pennylane_args=pennylane_args,
            encoding_type=encoding_type,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(
            embed_dim,
            n_qubits_ffn,
            n_qlayers=n_qlayers,
            dropout=dropout,
            batch=batch,
            circuit_type=circuit_type,
            q_device=q_device,
            pennylane_args=pennylane_args,
            encoding_type=encoding_type,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        attn_output = self.attn(x, x, x)
        x = self.norm1(attn_output + x)
        x = self.dropout1(x)

        ff_output: Tensor = self.ffn(x)
        x = self.norm2(ff_output + x)
        x = self.dropout2(x)

        return x
