import torch
import torch.nn as nn

from torch import Tensor

from lib.clone import get_clones

# from classic.positional_encoding import PositionalEncoder

from classic.positional_encoding import PyTorchPositionalEncoding as PositionalEncoder
from classic.encoder import Encoder
from quantum.encoder import Encoder as QuantumEncoder


class TextClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        num_classes: int,
        vocab_size: int,
        ffn_dim=32,
        dropout=0.1,
        n_qubits_transformer=0,
        n_qubits_ffn=0,
        n_qlayers=1,
        q_device="default.qubit",
    ):
        super(TextClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        print(f"++ There will be {num_blocks} transformer blocks")

        if n_qubits_transformer > 0:
            print(
                f"++ Transformer will use {n_qubits_transformer} qubits and {n_qlayers} q layers"
            )
            if n_qubits_ffn > 0:
                print(f"The feed-forward head will use {n_qubits_ffn} qubits")
            else:
                print("The feed-forward head will be classical")
            print(f"Using quantum device {q_device}")

            self.transformers = get_clones(
                QuantumEncoder(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer=n_qubits_transformer,
                    n_qubits_ffn=n_qubits_ffn,
                    n_qlayers=n_qlayers,
                    q_device=q_device,
                ),
                num_blocks,
            )

        else:
            self.transformers = get_clones(
                Encoder(embed_dim, num_heads, ffn_dim), num_blocks
            )
        self.class_logits = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)

        # # Normalise the data to range [0, 2*pi]
        # min_val = x.min()
        # max_val = x.max()

        # x = (x - min_val) / (max_val - min_val) * (2 * torch.pi)

        # print(f"Minimum value: {x.min()}, Maximum value: {x.max()}")

        for transformer in self.transformers:
            x = transformer(x)

        x = x.mean(dim=1)  # global average pooling, works in 1D
        x = self.dropout(x)
        x = self.class_logits(x)
        return x
