import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

from .transformer import Transformer
from .activations import gelu
from .util import to_2d


class Bert(layers.Layer):

    def __init__(self, config, **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.config = config

        self.vocab_size = self.config['vocab_size']
        self.embedding_size = self.config['embedding_size']
        self.input_hidden = self.config['input_hidden']
        self.segment_size = self.config['segment_size']
        self.max_seq_length = self.config['max_seq_length']
        self.dropout = self.config['dropout']
        self.factorize_embedding = self.config['factorize_embedding']

        if not self.factorize_embedding:
            assert self.embedding_size == self.input_hidden

        self.layer_num = self.config['layer_num']
        self.layer_norm_output = self.config['layer_norm_output']
        self.share_all = self.config['share_all']

        self.embedding = layers.Embedding(
            self.vocab_size,
            self.embedding_size,
            embeddings_initializer=initializers.TruncatedNormal(stddev=self.config['init_stddev'])
        )
        self.segment_embedding = layers.Embedding(
            self.segment_size,
            self.input_hidden,
            embeddings_initializer=initializers.TruncatedNormal(stddev=self.config['init_stddev'])
        )
        self.position_embedding = layers.Embedding(
            self.max_seq_length,
            self.input_hidden,
            embeddings_initializer=initializers.TruncatedNormal(stddev=self.config['init_stddev'])
        )
        if self.factorize_embedding:
            self.projection = self.add_weight(
                shape=(self.embedding_size, self.input_hidden), name='projection'
            )
        self.first_layer_norm = layers.LayerNormalization(name='attention_output_layer_norm')

        # Create transform layers
        self.transformers = []
        for i in range(self.layer_num):
            if i == 0:
                transformer = Transformer(
                    self.config['feedforward_config'],
                    self.config['attention_config'],
                    self.max_seq_length,
                    layer_norm_output=self.layer_norm_output
                )
                self.transformers.append(transformer)
            else:
                if self.share_all:
                    transformer = self.transformers[-1]
                else:
                    transformer = Transformer(
                        self.config['feedforward_config'],
                        self.config['attention_config'],
                        self.max_seq_length,
                        layer_norm_output=self.layer_norm_output
                    )
                    self.transformers.append(transformer)

    def call(self, inputs):
        # x = [batch_size, seq_size]
        # segment_ids = [batch_size, segment_size]
        x, attention_mask, segment_ids = inputs
        seq_length = x.shape[1]

        # Embedding
        # x = [batch_size, seq_size, embedding_size]
        x = self.embedding(x)

        # x = [batch_size * seq_size, embedding_size]
        x = to_2d(x)

        # Project
        # x = [batch_size * seq_size, input_hidden]
        if self.factorize_embedding:
            x = tf.matmul(x, self.projection)

        # Add type embedding
        x += to_2d(self.segment_embedding(segment_ids))

        # Add position embedding
        seq_tensor = tf.reshape(tf.range(seq_length), [1, seq_length])
        seq_tensor = self.position_embedding(seq_tensor)
        # x = [batch_size, seq_size, input_hidden]
        x = tf.reshape(x, [-1, seq_length, self.input_hidden])
        x += seq_tensor

        # First layer norm
        x = self.first_layer_norm(x)

        x = layers.Dropout(self.dropout)(x)

        layer_outputs = []
        for transformer in self.transformers:
            x = transformer([x, attention_mask])
            layer_outputs.append(x)

        # x = [batch_size * seq_size, input_hidden]
        x = layer_outputs[-1]
        _x = tf.reshape(x, [-1, self.max_seq_length, self.input_hidden])
        first_tokens = _x[:, 0, :]
        return first_tokens, x


class PreTrainLMPredictor(layers.Layer):

    def __init__(self, input_hidden, class_num, seq_length, init_stddev=0.02, **kwargs):
        super(PreTrainLMPredictor, self).__init__(**kwargs)
        self.input_hidden = input_hidden
        self.class_num = class_num
        self.seq_length = seq_length

        self.dense = layers.Dense(
            input_hidden,
            activation=gelu,
            kernel_initializer=initializers.TruncatedNormal(stddev=init_stddev),
        )
        self.layer_norm = layers.LayerNormalization()
        self.bias = self.add_weight(
            shape=(class_num,),
            initializer=initializers.Zeros(),
            name='pretrain-lm-bias',
        )

    def call(self, inputs):
        # bert_outputs = [batch_size * seq_length, input_hidden]
        bert_outputs, embedding, projection = inputs

        x = to_2d(bert_outputs)
        x = self.dense(x)

        x = tf.reshape(x, [-1, self.seq_length, self.input_hidden])
        x = self.layer_norm(x)

        # [batch_size, max_seq_length, input_hidden] x [embedding_size, input_hidden]^T
        # -> [batch_size, max_seq_length, embedding_size]
        if projection is not None:
            x = tf.matmul(x, projection, transpose_b=True)

        # [batch_size, max_seq_length, embedding_size] x [vocab_size, embedding_size]^T
        # -> [batch_size, max_seq_length, vocab_size]
        logits = tf.matmul(x, embedding, transpose_b=True)
        logits += self.bias

        return logits


class PreTrainNextSentencePredictor(layers.Layer):

    def __init__(self, class_num, init_stddev=0.02, **kwargs):
        super(PreTrainNextSentencePredictor, self).__init__(**kwargs)
        self.dense = layers.Dense(
            class_num,
            kernel_initializer=initializers.TruncatedNormal(stddev=init_stddev),
        )

    def call(self, inputs):
        x = to_2d(inputs)
        logits = self.dense(x)
        return logits
