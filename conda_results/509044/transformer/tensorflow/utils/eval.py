import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from .binary_acc import binary_accuracy


def evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    criterion: tf.keras.losses.Loss,
    max_seq_len: int,
):

    epoch_loss = []
    epoch_acc = []
    epoch_true = []
    epoch_pred = []

    for i, (lab, input) in enumerate(dataset):
        input = tf.convert_to_tensor(input)
        lab = tf.convert_to_tensor(lab)
        input = tf.cast(input, tf.float32)
        lab = tf.cast(lab, tf.float32)

        # Perform inference on the model
        predictions = tf.squeeze(model(input), axis=1)

        # Calculate loss
        loss = criterion(lab, predictions)

        # Calculate accuracy
        acc = binary_accuracy(predictions, lab)

        epoch_loss.append(loss.numpy())  # Convert the tensor to a numpy scalar
        epoch_acc.append(acc.numpy())
        epoch_true.extend(lab.numpy().tolist())
        epoch_pred.extend(tf.sigmoid(predictions).numpy().tolist())

    # Calculate AUC
    epoch_auc = 100.0 * roc_auc_score(epoch_true, epoch_pred, multi_class="ovr")

    # Divide total loss by total number of batches per epoch
    return np.mean(epoch_loss), np.mean(epoch_acc), epoch_auc
