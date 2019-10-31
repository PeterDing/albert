import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

from .attention import Attention
from .feed_forward import FeedForward
from .util import LayerStore, to_2d


class Transformer(layers.Layer, LayerStore):

    def __init__(
        self,
        feedforward_config,
        attention_config,
        seq_length,
        layer_norm_output=False,
        **kwargs,
    ):
        super(Transformer, self).__init__(**kwargs)
        self.attention_config = attention_config
        self.feedforward_config = feedforward_config
        self.input_hidden = feedforward_config['hidden_size']
        self.seq_length = seq_length
        self.layer_norm_output = layer_norm_output

        self.attention = Attention(**attention_config)
        self.feedforward = FeedForward(**feedforward_config)

        self.attention_output_dense = self.get_layer(
            attention_config['shared_output'],
            'attention_output_dense',
            layers.Dense,
            self.input_hidden,
            kernel_initializer=initializers.TruncatedNormal(
                stddev=attention_config['init_stddev']
            ),
        )
        self.attention_output_layer_norm = self.get_layer(
            attention_config['shared_output'],
            'attention_output_layer_norm',
            layers.LayerNormalization,
        )
        if layer_norm_output:
            self.feedforward_output_layer_norm = self.get_layer(
                feedforward_config['shared_output'],
                'feedforward_output_layer_norm',
                layers.LayerNormalization,
            )

    def call(self, inputs):
        # x = [batch_size, seq_length, input_hidden]
        # or = [batch_size * seq_length, input_hidden]
        pre_input, attention_mask = inputs

        assert len(pre_input.shape) in [2, 3]

        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        pre_input = to_2d(pre_input)

        x = pre_input

        # Do attention
        x = self.attention([x, x, attention_mask, self.seq_length])

        # Do attention output
        x = self.attention_output_dense(x)
        x = layers.Dropout(self.attention_config['dropout'])(x)

        # Residential
        attention_output = x + pre_input

        # Layer Norm
        x = tf.reshape(attention_output, [-1, self.seq_length, self.input_hidden])
        x = self.attention_output_layer_norm(x)
        x = to_2d(x)

        # FeedForward Layer
        x = self.feedforward(x)

        # Residential
        output = x + attention_output

        if self.layer_norm_output:
            output = tf.reshape(output, [-1, self.seq_length, self.input_hidden])
            output = self.feedforward_output_layer_norm(output)
            output = to_2d(output)

        return output
