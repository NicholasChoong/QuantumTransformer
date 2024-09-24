import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def plot_metrics(train_loss, val_loss, train_acc, val_acc, train_auc, val_auc):

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 10))

    # Top row: Train and Val Loss, Train and Val Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, "b-", label="Train Loss")
    plt.plot(epochs, val_loss, "r-", label="Val Loss")
    plt.title("Train and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_acc, "b-", label="Train Accuracy")
    plt.plot(epochs, val_acc, "r-", label="Val Accuracy")
    plt.title("Train and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Bottom row: Train and Val AUC
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_auc, "b-", label="Train AUC")
    plt.plot(epochs, val_auc, "r-", label="Val AUC")
    plt.title("Train and Validation AUC")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()

    plt.tight_layout()
    plt.show()
