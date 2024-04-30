from time import time
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from zmq import device

from text_classifier.text_classifier import TextClassifier
from lib.data_loader import yield_tokens, collate_batch, batch_sampler
from lib.epoch import epoch_time
from lib.eval import evaluate
from lib.parameter_count import count_parameters
from lib.train import train

from config import dev


QUANTUM = True


def main():

    MAX_SEQ_LEN = 64

    batch_size = 32
    n_epochs = 5
    lr = 0.001

    embed_dim = 2 if QUANTUM else 8
    num_heads = 2
    num_blocks = 1
    num_classes = 2
    vocab_size = 20000
    ffn_dim = 8
    n_qubits_transformer = 2 if QUANTUM else 0
    n_qubits_ffn = 2 if QUANTUM else 0
    n_qlayers = 1 if QUANTUM else 0
    q_device = "default.qubit.torch"
    dropout_rate = 0.1

    train_iter = IMDB(root="./.datatext", split="train")
    test_iter = IMDB(root="./.datatext", split="test")

    train_data = to_map_style_dataset(train_iter)
    test_data = to_map_style_dataset(test_iter)

    # size = 320
    # train_data = np.array(train_data)[
    #     np.random.choice(len(train_data), size=size, replace=False)
    # ].tolist()
    # test_data = np.array(test_data)[
    #     np.random.choice(len(test_data), size=size, replace=False)
    # ].tolist()

    # train_data = [(int(label), text) for label, text in train_data]
    # test_data = [(int(label), text) for label, text in test_data]

    tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(
        yield_tokens(train_data, tokenizer),
        specials=["<unk>", "<pad>"],
        max_tokens=vocab_size,
    )
    vocab.set_default_index(vocab["<unk>"])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, vocab, tokenizer),
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, vocab, tokenizer),
    )

    # train_loader = DataLoader(
    #     train_data,
    #     batch_sampler=batch_sampler(train_data, batch_size, tokenizer),
    #     collate_fn=lambda batch: collate_batch(batch, vocab, tokenizer),
    # )

    # test_loader = DataLoader(
    #     test_data,
    #     batch_sampler=batch_sampler(test_data, batch_size, tokenizer),
    #     collate_fn=lambda batch: collate_batch(batch, vocab, tokenizer),
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
        q_device=q_device,
    )
    print(f"The model has {count_parameters(model):,} trainable parameters")

    model.to(dev)

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    if num_classes < 3:
        criterion = torch.nn.BCEWithLogitsLoss()  # logits -> sigmoid -> loss
    else:
        criterion = torch.nn.CrossEntropyLoss()  # logits -> log_softmax -> NLLloss

    # training loop
    best_valid_loss = float("inf")
    for iepoch in range(n_epochs):
        start_time = time()

        print(f"Epoch {iepoch+1}/{n_epochs}")

        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, MAX_SEQ_LEN
        )
        valid_loss, valid_acc = evaluate(model, test_loader, criterion, MAX_SEQ_LEN)

        end_time = time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "model.pt")

        print(f"Epoch: {iepoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
