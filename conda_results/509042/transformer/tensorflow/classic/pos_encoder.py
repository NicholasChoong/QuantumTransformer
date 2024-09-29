import math
import tensorflow as tf


class PositionalEncoder(tf.keras.layers.Layer):

    def __init__(self, d_model: int, dropout=0.1, max_len=128):
        super(PositionalEncoder, self).__init__()
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) * (-math.log(10000.0) / d_model)
        )
        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)

        # Stack them along the last dimension to alternate between sin and cos values
        pe = tf.concat([pe_sin, pe_cos], axis=-1)

        # Ensure the correct shape by reshaping
        pe = tf.reshape(pe, (max_len, 1, d_model))

        # Store the positional encodings in a constant tensor
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        with tf.device("/CPU:0"):
            x = tf.transpose(x, perm=[1, 0, 2])
            x = x + self.pe[:seq_len]
            x = self.dropout(x)
            x = tf.transpose(x, perm=[1, 0, 2])
        return x
