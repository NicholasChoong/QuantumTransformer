import torch
import glob
import pickle
import numpy as np
import tensorflow as tf

from typing import Literal


def get_dataset(
    name: Literal["amazon", "imdb", "yelp"], type: Literal["train", "val", "test"]
):
    # Load the PyTorch tensors from .pt files
    file_list = glob.glob(f"data/word_embeddings/{name}_{type}_batch_*.pt")
    tensor_list = [torch.load(f, weights_only=True) for f in sorted(file_list)]

    # Combine tensors and convert them to NumPy
    combined_tensor = torch.cat(tensor_list, dim=0)
    combined_tensor = combined_tensor.numpy()

    # Load labels from the pickle file
    with open(f"data/word_labels/{name}_{type}_labels.pkl", "rb") as f:
        word_labels = pickle.load(f)

    return combined_tensor, word_labels


def load_dataset(name: Literal["amazon", "imdb", "yelp"]):
    # Get the train, validation, and test data and labels
    train_data, train_labels = get_dataset(name, "train")
    val_data, val_labels = get_dataset(name, "val")
    test_data, test_labels = get_dataset(name, "test")

    # Convert the NumPy arrays to TensorFlow tensors
    dataset = {}
    dataset["train"] = (
        tf.convert_to_tensor(train_labels, dtype=tf.float32),
        tf.convert_to_tensor(train_data, dtype=tf.float32),
    )
    dataset["val"] = (
        tf.convert_to_tensor(val_labels, dtype=tf.float32),
        tf.convert_to_tensor(val_data, dtype=tf.float32),
    )
    # dataset["test"] = (
    #     tf.convert_to_tensor(test_labels, dtype=tf.float32),
    #     tf.convert_to_tensor(test_data, dtype=tf.float32),
    # )

    return dataset
