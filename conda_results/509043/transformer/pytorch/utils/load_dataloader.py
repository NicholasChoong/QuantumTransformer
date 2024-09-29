import torch
import numpy as np
from typing import Literal

from torch.utils.data import DataLoader

from transformer.pytorch.classic.pos_encoder import PositionalEncoder
from transformer.pytorch.utils.load_dataset import load_dataset, sampling


def load_dataloader(
    dataset_name: Literal["amazon", "imdb", "yelp"] = "imdb",
    max_seq_len: int = 128,
    batch_size: int = 64,
    sample_size: int = 0,
    label_count: bool = True,
):

    dataset = load_dataset(dataset_name)
    train_labels, train_embeddings = dataset["train"]
    val_labels, val_embeddings = dataset["val"]
    test_labels, test_embeddings = dataset["test"]

    pos_encoder = PositionalEncoder(
        train_embeddings.shape[-1], max_len=max_seq_len, device=torch.device("cpu")
    )
    train_embeddings = pos_encoder(train_embeddings)
    val_embeddings = pos_encoder(val_embeddings)
    test_embeddings = pos_encoder(test_embeddings)

    if sample_size:
        train_labels, train_embeddings = sampling(
            train_labels, train_embeddings, int(sample_size * 0.8)
        )

    val_test_size = int(
        len(train_labels) * 0.25
    )  # 25% of train data which is 20% of total data

    val_labels, val_embeddings = sampling(val_labels, val_embeddings, val_test_size)

    test_labels, test_embeddings = sampling(test_labels, test_embeddings, val_test_size)

    train_data = list(zip(train_labels, train_embeddings))
    val_data = list(zip(val_labels, val_embeddings))
    test_data = list(zip(test_labels, test_embeddings))

    train_data = [(float(label), embedding) for label, embedding in train_data]
    val_data = [(float(label), embedding) for label, embedding in val_data]
    test_data = [(float(label), embedding) for label, embedding in test_data]

    if label_count:
        print("Train, Val, Test size: ", len(train_data), len(val_data), len(test_data))

        print("pos: ", len([label for label, embedding in train_data if label == 0]))
        print("neg: ", len([label for label, embedding in train_data if label == 1]))
        print("pos: ", len([label for label, embedding in val_data if label == 0]))
        print("neg: ", len([label for label, embedding in val_data if label == 1]))
        print("pos: ", len([label for label, embedding in test_data if label == 0]))
        print("neg: ", len([label for label, embedding in test_data if label == 1]))

    train_loader = DataLoader(
        train_data,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader
