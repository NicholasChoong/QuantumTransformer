import numpy as np
import torch
import glob
import pickle
import gc

from typing import Literal
from tqdm import tqdm
import os

import psutil


def sampling(labels, embeddings, sample_size):
    indices = np.random.choice(len(labels), size=sample_size, replace=False)
    labels = [labels[i] for i in indices]
    embeddings = [embeddings[i].detach().clone() for i in indices]
    combined_tensor = torch.stack(embeddings, dim=0)
    return labels, combined_tensor


def load_tensor(file):
    return torch.load(file, weights_only=True)


def get_dataset(
    name: Literal["amazon", "imdb", "yelp"], type: Literal["train", "val", "test"]
):
    # file_list = glob.glob(f"data/word_embeddings/{name}_{type}_batch_*.pt")
    group_dir = os.getenv("MYGROUP")
    glob.glob(
        f"{group_dir}/QuantumTransformer/data/word_embeddings/{name}_{type}_batch_*.pt"
    )
    tensor_list = [
        load_tensor(f) for f in tqdm(sorted(file_list), desc=f"Loading {type} tensors")
    ]
    combined_tensor = torch.cat(tensor_list, dim=0)
    with open(f"data/word_labels/{name}_{type}_labels.pkl", "rb") as f:
        word_labels = pickle.load(f)
    return word_labels, combined_tensor


def load_dataset(name: Literal["amazon", "imdb", "yelp"]):
    train_labels, train_data = get_dataset(name, "train")

    sampled_train_labels, sampled_train_data = None, None
    if len(train_labels) > 300_000:
        sampled_train_labels, sampled_train_data = sampling(train_labels, train_data, 0)

        train_labels = None
        train_data = None
        del train_labels, train_data
        gc.collect()
    else:
        sampled_train_labels = train_labels
        sampled_train_data = train_data

    val_test_size = int(
        len(sampled_train_labels) * 0.25
    )  # 25% of train data which is 20% of total data
    print("Sample size: ", val_test_size)

    val_labels, val_data = get_dataset(name, "val")
    sampled_val_labels, sampled_val_data = None, None
    if len(val_labels) > val_test_size:
        sampled_val_labels, sampled_val_data = sampling(
            val_labels, val_data, val_test_size
        )
    else:
        sampled_val_labels = val_labels
        sampled_val_data = val_data

    val_labels = None
    val_data = None
    del val_labels, val_data
    gc.collect()

    test_labels, test_data = get_dataset(name, "test")
    sampled_test_labels, sampled_test_data = None, None
    if len(test_labels) > val_test_size:
        sampled_test_labels, sampled_test_data = sampling(
            test_labels, test_data, val_test_size
        )
    else:
        sampled_test_labels = test_labels
        sampled_test_data = test_data
    print("Test size: ", len(sampled_test_labels))

    test_labels = None
    test_data = None
    del test_labels, test_data
    gc.collect()

    print("Sampled train tensor shape: ", sampled_train_data.shape)
    print("Sampled val tensor shape: ", sampled_val_data.shape)
    print("Sampled test tensor shape: ", sampled_test_data.shape)

    dataset = {}
    dataset["train"] = (sampled_train_labels, sampled_train_data)
    dataset["val"] = (sampled_val_labels, sampled_val_data)
    dataset["test"] = (sampled_test_labels, sampled_test_data)
    return dataset
