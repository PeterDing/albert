from tensorflow.keras import layers
from tensorflow.keras import initializers

from .activations import gelu

from .util import LayerStore


class FeedForward(layers.Layer, LayerStore):

    def __init__(
        self,
        hidden_size=1024,
        intermediate_hidden_size=4096,
        dropout=0.0,
        init_stddev=0.02,
        shared_intermediate=False,
        shared_output=False,
        **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_hidden_size = intermediate_hidden_size
        self.dropout = dropout
        self.init_stddev = init_stddev
        self.shared_intermediate = shared_intermediate

        self.intermediate_dense = self.get_layer(
            shared_intermediate,
            'ff_intermediate_dense',
            layers.Dense,
            intermediate_hidden_size,
            kernel_initializer=initializers.TruncatedNormal(stddev=init_stddev),
            activation=gelu,
        )
        self.output_dense = self.get_layer(
            shared_output,
            'ff_output_dense',
            layers.Dense,
            hidden_size,
            kernel_initializer=initializers.TruncatedNormal(stddev=init_stddev),
        )

    def call(self, inputs):
        x = self.intermediate_dense(inputs)
        x = self.output_dense(x)
        x = layers.Dropout(self.dropout)(x)
        return x
