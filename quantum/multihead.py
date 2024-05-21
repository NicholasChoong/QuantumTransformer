import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pennylane as qml
from pennylane import templates
from pennylane.qnn.torch import TorchLayer

from classic.attention import attention

from torch import Tensor


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout=0.1,
        mask: Tensor | None = None,
        use_bias=False,
        n_qubits=4,
        n_qlayers=1,
        q_device="default.qubit",
    ):
        super(MultiHeadedAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask = mask
        # We assume dim_v always equals dim_k
        self.dim_k = embed_dim // num_heads  # projection dimensions

        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.q_device = q_device
        if "qulacs" in q_device:
            self.dev = qml.device(q_device, wires=self.n_qubits, gpu=True)
        elif "braket" in q_device:
            self.dev = qml.device(q_device, wires=self.n_qubits, parallel=True)
        else:
            self.dev = qml.device(q_device, wires=self.n_qubits, torch_device="cuda")

        @qml.qnode(self.dev, interface="torch")
        def qlayer(inputs, weights):
            templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")
            templates.BasicEntanglerLayers(
                weights, wires=range(n_qubits), rotation=qml.RZ
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {self.n_qubits})")

        # The quantum layers for the query, key, and value projections
        self.k_linear = TorchLayer(qlayer, self.weight_shapes)
        self.q_linear = TorchLayer(qlayer, self.weight_shapes)
        self.v_linear = TorchLayer(qlayer, self.weight_shapes)
        # The quantum layer to combine the heads
        self.combine_heads = TorchLayer(qlayer, self.weight_shapes)

        self.attn_weights: Tensor | None = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ):
        batch_size, seq_len, embed_dim = query.size()
        assert (
            embed_dim == self.embed_dim
        ), f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        K = [self.k_linear(key[:, t, :]) for t in range(seq_len)]
        Q = [self.q_linear(query[:, t, :]) for t in range(seq_len)]
        V = [self.v_linear(value[:, t, :]) for t in range(seq_len)]

        K = torch.Tensor(pad_sequence(K))
        Q = torch.Tensor(pad_sequence(Q))
        V = torch.Tensor(pad_sequence(V))

        x: Tensor
        x, self.attn_weights = attention(Q, K, V, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, embed_dim)

        output = [self.combine_heads(x[:, t, :]) for t in range(seq_len)]
        output = torch.Tensor(pad_sequence(output))
        return output
