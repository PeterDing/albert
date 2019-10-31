import math

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

from .util import LayerStore


def transpose(tensor, seq_length, head_size, per_hidden_size):
    tensor = tf.reshape(tensor, [-1, seq_length, head_size, per_hidden_size])
    tensor = tf.transpose(tensor, [0, 2, 1, 3])
    return tensor


class Attention(layers.Layer, LayerStore):

    def __init__(
        self,
        head_size=16,
        per_hidden_size=64,
        dropout=0.0,
        init_stddev=0.02,
        return_2d=True,
        shared=False,
        shared_output=False,
        **kwargs
    ):
        super(Attention, self).__init__(**kwargs)
        self.head_size = head_size
        self.per_hidden_size = per_hidden_size
        self.dropout = dropout
        self.input_hidden = head_size * per_hidden_size
        self.init_stddev = init_stddev
        self.shared = shared
        self.return_2d = return_2d

        self.query_dense = self.get_layer(
            shared,
            'query_dense',
            layers.Dense,
            self.input_hidden,
            kernel_initializer=initializers.TruncatedNormal(stddev=init_stddev),
        )
        self.key_dense = self.get_layer(
            shared,
            'key_dense',
            layers.Dense,
            self.input_hidden,
            kernel_initializer=initializers.TruncatedNormal(stddev=init_stddev),
        )
        self.value_dense = self.get_layer(
            shared,
            'value_dense',
            layers.Dense,
            self.input_hidden,
            kernel_initializer=initializers.TruncatedNormal(stddev=init_stddev),
        )

    # inputs = [ [batch_size, from_seq_size, per_hidden_size], [batch_size, to_seq_size, per_hidden_size] ]
    def call(self, inputs):
        query, key, attention_mask, seq_length = inputs
        assert query.shape == query.shape

        # 1. Reshape inputs to matrix for following dense layers
        query = tf.reshape(query, [-1, self.input_hidden])
        key = tf.reshape(query, [-1, self.input_hidden])

        # 2. Use different denses for inputs
        query = self.query_dense(query)
        value = self.value_dense(key)
        key = self.key_dense(key)

        # 3. Transpose for calculating content scores
        # query = [batch_size, head_size, seq_length, per_hidden_size]
        # key, value as same
        query = transpose(query, seq_length, self.head_size, self.per_hidden_size)
        key = transpose(key, seq_length, self.head_size, self.per_hidden_size)
        value = transpose(value, seq_length, self.head_size, self.per_hidden_size)

        # 4. Compute scores
        # scores = [batch_size, head_size, seq_length, seq_length]
        scores = tf.matmul(query, key, transpose_b=True) / math.sqrt(self.per_hidden_size)

        # 5. Mask for attention
        if attention_mask is not None:
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # attention_mask = [batch_size, 1, seq_length, seq_length]
            attention_mask = tf.expand_dims(attention_mask, 1)
            attention_mask = (1 - tf.cast(attention_mask, tf.float32)) * -10000.0
            scores += attention_mask

        # 6. Normalize the attension score to possibilities
        attention_probs = tf.nn.softmax(scores)  # axis = -1

        # 7. dropout from original paper, but for albert it is useless
        attention_probs = layers.Dropout(self.dropout)(attention_probs)

        # 8. Get context
        # context = [batch_size, head_size, seq_length, per_hidden_size]
        context = tf.matmul(attention_probs, value)
        # context = [batch_size, seq_length, head_size, per_hidden_size]
        context = tf.transpose(context, [0, 2, 1, 3])

        if self.return_2d:
            # return [batch_size * seq_length, head_size * per_hidden_size]
            return tf.reshape(context, [-1, self.head_size * self.per_hidden_size])
        else:
            # return [batch_size, seq_length, head_size * per_hidden_size]
            return tf.reshape(context, [-1, seq_length, self.head_size * self.per_hidden_size])
