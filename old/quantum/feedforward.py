import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pennylane as qml
from pennylane import templates
from pennylane.qnn.torch import TorchLayer

from torch import Tensor


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
        n_qlayers=1,
        dropout=0.1,
        q_device="default.qubit",
    ):
        super(FeedForward, self).__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = n_qubits
        self.n_qubits = n_qubits

        self.dev = qml.device(q_device, wires=self.n_qubits, torch_device="cuda")

        def _circuit(inputs, weights):
            templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")
            templates.BasicEntanglerLayers(
                weights, wires=range(n_qubits), rotation=qml.RZ
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.linear_1 = nn.Linear(embed_dim, self.ffn_dim)
        self.qlayer = qml.QNode(_circuit, self.dev, interface="torch")
        self.linear_2 = nn.Linear(self.ffn_dim, embed_dim)

        self.weight_shapes = {"weights": (n_qlayers, n_qubits)}
        self.vqc = TorchLayer(self.qlayer, self.weight_shapes)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        batch_size, seq_len, embed_dim = x.size()
        x = self.linear_1(x)
        X = [self.vqc(x[:, t, :]) for t in range(seq_len)]
        x = torch.Tensor(pad_sequence(X))
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
