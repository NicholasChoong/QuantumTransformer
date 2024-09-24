import copy
import tensorflow as tf


def get_clones(layer: tf.keras.layers.Layer, N: int):
    return [copy.deepcopy(layer) for _ in range(N)]
