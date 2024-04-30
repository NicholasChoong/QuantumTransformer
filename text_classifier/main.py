from time import time
import torch
from torchtext import data, datasets

from text_classifier.text_classifier import TextClassifier
from lib.epoch import epoch_time
from lib.eval import evaluate
from lib.parameter_count import count_parameters
from lib.train import train


if __name__ == "__main__":

    MAX_SEQ_LEN = 64

    batch_size = 32
    n_epochs = 5
    lr = 0.001

    embed_dim = 8
    num_heads = 2
    num_blocks = 1
    num_classes = 2
    vocab_size = 20000
    ffn_dim = 8
    n_qubits_transformer = 0
    n_qubits_ffn = 0
    n_qlayers = 1
    q_device = 1
    dropout_rate = 0.1

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    # LABEL = data.Field(sequential=False)
    LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    print(f"Training examples: {len(train_data)}")
    print(f"Testing examples:  {len(test_data)}")

    TEXT.build_vocab(train_data, max_size=vocab_size - 2)  # exclude <UNK> and <PAD>
    LABEL.build_vocab(train_data)

    train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data), batch_size=batch_size
    )

    model = TextClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        num_classes=num_classes,
        vocab_size=vocab_size,
        ffn_dim=ffn_dim,
        # n_qubits_transformer=n_qubits_transformer,
        # n_qubits_ffn=n_qubits_ffn,
        # n_qlayers=n_qlayers,
        # dropout=dropout_rate,
        # q_device=q_device,
    )
    print(f"The model has {count_parameters(model):,} trainable parameters")

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
            model, train_iter, optimizer, criterion, MAX_SEQ_LEN
        )
        valid_loss, valid_acc = evaluate(model, test_iter, criterion, MAX_SEQ_LEN)

        end_time = time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "model.pt")

        print(f"Epoch: {iepoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
