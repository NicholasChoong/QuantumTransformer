import os
from time import time
from typing import Literal
import numpy as np
import torch

from torch.utils.data import DataLoader


from transformer.pytorch.classic.pos_encoder import PositionalEncoder
from transformer.pytorch.quantum.pennylane.angle_amp import PennyLaneArgs
from transformer.pytorch.utils.load_dataloader import load_dataloader

from .text_classifier_no_embed import TextClassifier
from .utils.epoch import epoch_time
from .utils.eval_no_embed import evaluate
from .utils.param_count import count_parameters
from .utils.train_no_embed import train

from tqdm import tqdm


from torch.optim.lr_scheduler import StepLR


def main(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
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
    pooling_method: Literal["CLS", "MEAN", "MAX", "CONCAT"] = "CLS",
    ffn_dim=16,
    n_qubits_transformer=0,
    n_qubits_ffn=0,
    n_qlayers=0,
    dropout_rate=0.1,
    tqdm_disabled=False,
    q_device="default.qubit",
    batch=True,
    circuit_type: Literal["pennylane", "tensorcircuit"] = "tensorcircuit",
    encoding_type: Literal["angle_amp", "block"] = "angle_amp",
    pennylane_args: PennyLaneArgs = {
        "hadamard": False,
        "encoder": "angle",
        "angle_rot": "Z",
        "entangler": "basic",
        "rot": "Z",
        "imprimitive": "Z",
    },
    multi_gpu: bool = False,
):

    save_dir = f".models_{n_epochs}_{n_qubits_transformer}_{n_qubits_ffn}_{n_qlayers}"
    save_path = os.path.join(save_dir, "model_and_metrics_epoch_{}.pt")
    os.makedirs(save_dir, exist_ok=True)

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
        pennylane_args=pennylane_args,
        pooling_method=pooling_method,
        encoding_type=encoding_type,
    )

    print(model)

    if multi_gpu:
        model = torch.nn.DataParallel(model)

    print(f"The model has {count_parameters(model):,} trainable parameters")
    start_time = time()

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())  # type: ignore
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
            total=len(train_dataloader),
            desc=f"Epoch {iepoch+1:3}/{n_epochs}",
            unit="batch",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=tqdm_disabled,
        ) as progress_bar:
            operation_start_time = time()

            train_loss, train_acc, train_auc = train(
                model,
                train_dataloader,
                optimizer,
                criterion,
                scheduler,
                max_seq_len,
                batch_size,
                progress_bar,
            )
            progress_bar.set_postfix_str(("Evaluating on validation set..."))

            val_loss, val_acc, val_auc = evaluate(
                model, val_dataloader, criterion, max_seq_len
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
                batch_time = len(train_dataloader.dataset) / ep_time  # type: ignore

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
