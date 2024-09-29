import tensorflow as tf
import tensorflow.keras as keras


class FeedForward(keras.Model):
    """
    The x that is passed to the call method is a tensor of shape (batch_size, sequence_length, embedding_dimension),
    rather than a flattened version of it (with shape (batch_size, sequence_length * embedding_dimension)).
    The (same) feed-forward layer applies to the last dimension only (the embedding dimension) for each batch and
    for each position in the sequence, hence position-wise.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = keras.layers.Dense(ffn_dim, activation="relu")
        self.linear_2 = keras.layers.Dense(embed_dim)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x):
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
