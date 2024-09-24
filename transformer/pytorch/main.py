import os
from time import time
from typing import Literal
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator

from .text_classifier import TextClassifier
from .utils.data_loader import yield_tokens, collate_batch, batch_sampler
from .utils.epoch import epoch_time
from .utils.eval import evaluate
from .utils.param_count import count_parameters
from .utils.train import train as train_single_gpu
from .utils.train_multi_gpu import train as train_multi_gpu

from tqdm import tqdm


from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split


def main(
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
    multi_gpu=False,
):

    save_dir = f".models_{n_epochs}_{n_qubits_transformer}_{n_qubits_ffn}_{n_qlayers}"
    save_path = os.path.join(save_dir, "model_and_metrics_epoch_{}.pt")
    os.makedirs(save_dir, exist_ok=True)

    # train_iter = IMDB(root="./.datatext", split="train")
    # test_iter = IMDB(root="./.datatext", split="test")

    # train_data = to_map_style_dataset(train_iter)
    # test_data = to_map_style_dataset(test_iter)

    size = sample_size

    # test_data = np.array(test_data)[
    #     np.random.choice(len(test_data), size=size, replace=False)
    # ].tolist()

    train_iter = IMDB(root="./.datatext", split="train")

    train_data = to_map_style_dataset(train_iter)

    if sample_size:
        train_data = np.array(train_data)[
            np.random.choice(len(train_data), size=size, replace=False)
        ].tolist()

    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_data = [(int(label), text) for label, text in train_data]
    val_data = [(int(label), text) for label, text in val_data]

    print("pos: ", len([label for label, text in train_data if label == 1]))
    print("neg: ", len([label for label, text in train_data if label == 2]))

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        yield_tokens(train_data, tokenizer),
        specials=["<unk>", "<pad>"],
        max_tokens=vocab_size,
    )
    vocab.set_default_index(vocab["<unk>"])

    train_loader = DataLoader(
        train_data,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, vocab, tokenizer),
    )
    val_loader = DataLoader(
        val_data,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, vocab, tokenizer),
    )

    if multi_gpu:
        train = train_multi_gpu
    else:
        train = train_single_gpu

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

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()  # logits -> sigmoid -> loss
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # training loop
    best_val_loss = float("inf")
    best_val_acc, best_epoch_acc = 0.0, 0
    best_val_auc, best_epoch_auc = 0.0, 0
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    train_auc_list, val_auc_list = [], []
    for iepoch in range(n_epochs):
        with tqdm(
            total=len(train_loader.dataset),  # type: ignore
            desc=f"Epoch {iepoch+1:3}/{n_epochs}",
            unit="batch",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=tqdm_disabled,
        ) as progress_bar:
            operation_start_time = time()

            train_loss, train_acc, train_auc = train(
                model,
                train_loader,
                optimizer,
                criterion,
                scheduler,
                max_seq_len,
                progress_bar,
            )

            val_loss, val_acc, val_auc = evaluate(
                model, val_loader, criterion, max_seq_len
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
                batch_time = len(train_loader.dataset) / ep_time  # type: ignore

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

            torch.save(
                {
                    "epoch": iepoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss_list,
                    "train_acc": train_acc_list,
                    "train_auc": train_auc_list,
                    "val_loss": val_loss_list,
                    "val_acc": val_acc_list,
                    "val_auc": val_auc_list,
                },
                save_path.format(iepoch + 1),
            )

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(model.state_dict(), "model.pt")

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
