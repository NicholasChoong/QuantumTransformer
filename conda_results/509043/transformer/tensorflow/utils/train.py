import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from .binary_acc import binary_accuracy


def train(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    criterion: tf.keras.losses.Loss,
    scheduler: tf.keras.optimizers.schedules.LearningRateSchedule,
    max_seq_len: int,
    batch_size: int,
    progress_bar,
):
    epoch_loss = []
    epoch_acc = []
    epoch_true = []
    epoch_pred = []

    model.trainable = True  # Set model to training mode

    for i, (lab, input) in enumerate(dataset):
        with tf.GradientTape() as tape:
            input = tf.convert_to_tensor(input)
            lab = tf.convert_to_tensor(lab)

            predictions = tf.squeeze(model(input), axis=1)

            # Calculate loss
            loss = criterion(lab, predictions)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Calculate accuracy
        acc = binary_accuracy(predictions, lab)

        epoch_loss.append(loss.numpy())
        epoch_acc.append(acc.numpy())
        epoch_true.extend(lab.numpy().tolist())
        epoch_pred.extend(tf.sigmoid(predictions).numpy().tolist())

        progress_bar.update(batch_size)

    # Compute AUC
    epoch_auc = 100.0 * roc_auc_score(epoch_true, epoch_pred, multi_class="ovr")

    # Update learning rate using scheduler
    # if isinstance(scheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    #     optimizer.learning_rate.assign(scheduler(optimizer.iterations))

    return np.mean(epoch_loss), np.mean(epoch_acc), epoch_auc
