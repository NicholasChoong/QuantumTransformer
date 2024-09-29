import math
import tensorflow as tf
from tensorflow.keras import layers


def attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    mask: tf.Tensor | None = None,
    dropout: layers.Dropout | None = None,
):
    """Scaled Dot Product Attention"""
    dim_k = tf.cast(tf.shape(query)[-1], tf.float32)  # type: ignore
    # scaled = tf.matmul(query, key, transpose_b=True) / math.sqrt(dim_k)
    scaled = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(dim_k)
    if mask is not None:
        mask = tf.expand_dims(mask, 1)
        scaled = tf.where(mask == 0, -1e9, scaled)
    scores = tf.nn.softmax(scaled, axis=-1)
    if dropout is not None:
        scores = dropout(scores)
    # attn = tf.matmul(scores, value)
    attn = tf.matmul(scores, value)
    return attn, scores
