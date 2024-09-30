from typing import Literal

import torch.nn as nn
from ..classic.attention import attention

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ._tensorcircuit import QuantumLayer as tc_QuantumLayer
from .pennylane.angle_amp import PennyLaneArgs, QuantumLayer as qml_QuantumLayer


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        pennylane_args: PennyLaneArgs,
        dropout=0.1,
        mask: Tensor | None = None,
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
        self.batch = batch

        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        if circuit_type == "pennylane":
            QuantumLayer = qml_QuantumLayer
        else:
            QuantumLayer = tc_QuantumLayer

        # The quantum layers for the query, key, and value projections
        self.k_linear = QuantumLayer(
            n_qubits, n_qlayers, q_device=q_device, pennylane_args=pennylane_args
        )
        self.q_linear = QuantumLayer(
            n_qubits, n_qlayers, q_device=q_device, pennylane_args=pennylane_args
        )
        self.v_linear = QuantumLayer(
            n_qubits, n_qlayers, q_device=q_device, pennylane_args=pennylane_args
        )
        # The quantum layer to combine the heads
        self.combine_heads = QuantumLayer(
            n_qubits, n_qlayers, q_device=q_device, pennylane_args=pennylane_args
        )

        self.attn_weights: Tensor | None = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        batch_size, seq_len, embed_dim = query.size()
        assert (
            embed_dim == self.embed_dim
        ), f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        if self.batch:
            Q, K, V = [
                linear(x)
                .reshape(batch_size, seq_len, self.num_heads, self.dim_k)
                .transpose(1, 2)
                for linear, x in zip(
                    [self.k_linear, self.q_linear, self.v_linear], [query, key, value]
                )
            ]
        else:
            K = [self.k_linear(key[:, t, :]) for t in range(seq_len)]
            Q = [self.q_linear(query[:, t, :]) for t in range(seq_len)]
            V = [self.v_linear(value[:, t, :]) for t in range(seq_len)]

            K = Tensor(pad_sequence(K))
            Q = Tensor(pad_sequence(Q))
            V = Tensor(pad_sequence(V))

        x: Tensor
        x, self.attn_weights = attention(Q, K, V, dropout=self.dropout)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        output = self.combine_heads(x)
        return output
