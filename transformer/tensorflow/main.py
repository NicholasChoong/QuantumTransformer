import os
from time import time
from typing import Literal
import numpy as np
import tensorflow as tf

from transformer.tensorflow.classic.pos_encoder import PositionalEncoder
from transformer.tensorflow.utils.load_dataset import load_dataset

from .text_classifier import TextClassifier
from .utils.epoch import epoch_time
from .utils.eval import evaluate
from .utils.param_count import count_parameters
from .utils.train import train

from tqdm import tqdm


def main(
    dataset_name: Literal["amazon", "imdb", "yelp"] = "imdb",
    max_seq_len=128,
    batch_size=32,
    sample_size=0,
    n_epochs=30,
    lr=0.001,
    embed_dim=8,
    num_heads=2,
    num_blocks=2,
    num_classes=2,
    vocab_size=50000,
    ffn_dim=16,
    n_qubits_transformer=0,
    n_qubits_ffn=0,
    n_qlayers=0,
    dropout_rate=0.1,
    tqdm_disabled=False,
    q_device="default.qubit",
    batch=True,
    circuit_type: Literal["pennylane", "tensorcircuit"] = "tensorcircuit",
):

    save_dir = f".models_{n_epochs}_{n_qubits_transformer}_{n_qubits_ffn}_{n_qlayers}"
    save_path = os.path.join(save_dir, "model_and_metrics_epoch_{}.h5")
    os.makedirs(save_dir, exist_ok=True)

    size = sample_size

    dataset = load_dataset(dataset_name)
    train_labels, train_embeddings = dataset["train"]
    val_labels, val_embeddings = dataset["val"]
    # test_labels, test_embeddings = dataset["test"]
    test_labels, test_embeddings = dataset["val"]

    pos_encoder = PositionalEncoder(train_embeddings.shape[-1], max_len=max_seq_len)
    train_embeddings = pos_encoder(train_embeddings)
    val_embeddings = pos_encoder(val_embeddings)
    test_embeddings = pos_encoder(test_embeddings)

    with tf.device("/CPU:0"):
        train_data = list(zip(train_labels, train_embeddings))
        val_data = list(zip(val_labels, val_embeddings))
        test_data = list(zip(test_labels, test_embeddings))

        if sample_size:
            train_indices = np.random.choice(
                len(train_labels), size=int(size * 0.8), replace=False
            )
            train_labels = [train_labels[i] for i in train_indices]
            train_embeddings = [train_embeddings[i] for i in train_indices]

            val_indices = np.random.choice(
                len(val_labels), size=int(size * 0.2), replace=False
            )
            val_labels = [val_labels[i] for i in val_indices]
            val_embeddings = [val_embeddings[i] for i in val_indices]

            test_indices = np.random.choice(
                len(test_labels), size=int(size * 0.5), replace=False
            )
            test_labels = [test_labels[i] for i in test_indices]
            test_embeddings = [test_embeddings[i] for i in test_indices]

            train_data = list(zip(train_labels, train_embeddings))
            val_data = list(zip(val_labels, val_embeddings))
            test_data = list(zip(test_labels, test_embeddings))

        print("pos: ", len([label for label, embedding in train_data if label == 0]))
        print("neg: ", len([label for label, embedding in train_data if label == 1]))
        print("pos: ", len([label for label, embedding in val_data if label == 0]))
        print("neg: ", len([label for label, embedding in val_data if label == 1]))
        print("pos: ", len([label for label, embedding in test_data if label == 0]))
        print("neg: ", len([label for label, embedding in test_data if label == 1]))

        print(train_data[0][1].shape)

        train_labels, train_embeddings = zip(*train_data)
        val_labels, val_embeddings = zip(*val_data)
        test_labels, test_embeddings = zip(*test_data)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (list(train_labels), list(train_embeddings))
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (list(val_labels), list(val_embeddings))
        )
        # test_dataset = tf.data.Dataset.from_tensor_slices(
        #     list(test_labels), list(test_embeddings)
        # )

    train_dataset = (
        train_dataset.shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        val_dataset.shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    # test_dataset = (
    #     test_dataset.shuffle(buffer_size=10000)
    #     .batch(batch_size)
    #     .prefetch(tf.data.AUTOTUNE)
    # )

    model = TextClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        num_classes=num_classes,
        vocab_size=vocab_size,
        ffn_dim=ffn_dim,
        n_qubits_transformer=n_qubits_transformer,
        n_qubits_ffn=n_qubits_ffn,
        n_qlayers=n_qlayers,
        dropout=dropout_rate,
        max_seq_len=max_seq_len,
        q_device=q_device,
        batch=batch,
        circuit_type=circuit_type,
    )

    print(f"The model has {count_parameters(model):,} trainable parameters")
    start_time = time()

    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=5 * len(train_data) // batch_size,
        decay_rate=0.1,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Training loop
    best_val_loss = float("inf")
    best_val_acc, best_epoch_acc = 0.0, 0
    best_val_auc, best_epoch_auc = 0.0, 0
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    train_auc_list, val_auc_list = [], []

    for iepoch in range(n_epochs):
        with tqdm(
            total=len(train_dataset),
            desc=f"Epoch {iepoch+1:3}/{n_epochs}",
            unit="batch",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=tqdm_disabled,
        ) as progress_bar:
            operation_start_time = time()

            train_loss, train_acc, train_auc = train(
                model,
                train_dataset,
                optimizer,
                criterion,
                lr_schedule,
                max_seq_len,
                batch_size,
                progress_bar,
            )

            val_loss, val_acc, val_auc = evaluate(
                model, val_dataset, criterion, max_seq_len
            )

            end_time = time()

            epoch_mins, epoch_secs = epoch_time(operation_start_time, end_time)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            train_auc_list.append(train_auc)
            val_auc_list.append(val_auc)

            progress_bar.set_postfix_str(
                (
                    f"Epoch = {epoch_mins}m {epoch_secs}s, Loss = {train_loss:.4f}|{val_loss:.4f}, "
                    f"Acc = {train_acc:.3f}|{val_acc:.3f}, AUC = {train_auc:.3f}|{val_auc:.3f}"
                )
            )

            if tqdm_disabled:
                ep_time = end_time - operation_start_time
                batch_time = len(train_dataset) / ep_time

                print(
                    (
                        f"Epoch {iepoch+1:02}: {batch_time:.2f}batch/s, Epoch = {epoch_mins}m {epoch_secs}s, "
                        f"Loss = {train_loss:.4f}|{val_loss:.4f}, Acc = {train_acc:.3f}|{val_acc:.3f}, "
                        f"AUC = {train_auc:.3f}|{val_auc:.3f}"
                    )
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch_acc = iepoch + 1
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch_auc = iepoch + 1

            # model.save_weights(save_path.format(iepoch + 1))

    print(f"TOTAL TIME = {time()-start_time:.2f}s")
    print(f"BEST ACC = {best_val_acc:.2f}% AT EPOCH {best_epoch_acc}")
    print(f"BEST AUC = {best_val_auc:.2f} AT EPOCH {best_epoch_auc}")

    best_dict = {
        "best_val_acc": best_val_acc,
        "best_epoch_acc": best_epoch_acc,
        "best_val_auc": best_val_auc,
        "best_epoch_auc": best_epoch_auc,
    }

    return (
        train_loss_list,
        train_acc_list,
        val_loss_list,
        val_acc_list,
        train_auc_list,
        val_auc_list,
        best_dict,
    )
