import torch
import glob
import pickle

from typing import Literal


def get_dataset(
    name: Literal["amazon", "imdb", "yelp"], type: Literal["train", "val", "test"]
):
    file_list = glob.glob(f"data/word_embeddings/{name}_{type}_batch_*.pt")
    tensor_list = [torch.load(f, weights_only=True) for f in sorted(file_list)]
    combined_tensor = torch.cat(tensor_list, dim=0)
    with open(f"data/word_labels/{name}_{type}_labels.pkl", "rb") as f:
        word_labels = pickle.load(f)
    return combined_tensor, word_labels


def load_dataset(name: Literal["amazon", "imdb", "yelp"]):
    train_data, train_labels = get_dataset(name, "train")
    val_data, val_labels = get_dataset(name, "val")
    test_data, test_labels = get_dataset(name, "test")
    dataset = {}
    dataset["train"] = (train_labels, train_data)
    dataset["val"] = (val_labels, val_data)
    dataset["test"] = (test_labels, test_data)
    return dataset
