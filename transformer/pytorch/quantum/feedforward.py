from typing import Literal

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


from ._tensorcircuit import QuantumLayer as tc_QuantumLayer
from .pennylane.angle_amp import PennyLaneArgs, QuantumLayer as qml_QuantumLayer


class FeedForward(nn.Module):
    """
    The x that is passed to the forward method is a tensor of shape (batch_size, sequence_length, embedding_dimension),
    rather than a flattened version of it (with shape (batch_size, sequence_length * embedding_dimension)).
    The (same) feed-forward layer applies to the last dimension only (the embedding dimension) for each batch and
    for each position in the sequence, hence position-wise.
    """

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int,
        pennylane_args: PennyLaneArgs,
        n_qlayers=1,
        dropout=0.1,
        batch=True,
        circuit_type: Literal["pennylane", "tensorcircuit"] = "tensorcircuit",
        q_device="default.qubit",
    ):
        super(FeedForward, self).__init__()
        self.ffn_dim = n_qubits
        self.batch = batch

        if circuit_type == "pennylane":
            QuantumLayer = qml_QuantumLayer
        else:
            QuantumLayer = tc_QuantumLayer

        self.linear_1 = nn.Linear(embed_dim, self.ffn_dim)
        self.linear_2 = nn.Linear(self.ffn_dim, embed_dim)
        self.vqc = QuantumLayer(
            n_qubits, n_qlayers, q_device=q_device, pennylane_args=pennylane_args
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        batch_size, seq_len, embed_dim = x.size()
        x = self.linear_1(x)
        if self.batch:
            x = self.vqc(x)
        else:
            X = [self.vqc(x[:, t, :]) for t in range(seq_len)]
            x = Tensor(pad_sequence(X))
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
