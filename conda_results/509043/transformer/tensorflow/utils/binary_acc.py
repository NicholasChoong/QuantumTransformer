import tensorflow as tf


def binary_accuracy(preds: tf.Tensor, y: tf.Tensor):
    """
    Returns accuracy per batch, i.e., if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = tf.round(tf.sigmoid(preds))

    correct = tf.cast(
        rounded_preds == y, dtype=tf.float32
    )  # convert to float for division
    acc = tf.reduce_sum(correct) / tf.cast(tf.size(correct), tf.float32)
    return acc
