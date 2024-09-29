import tensorflow as tf
import numpy as np


def count_parameters(model: tf.keras.Model):
    return np.sum([np.prod(v.shape) for v in model.trainable_variables])
