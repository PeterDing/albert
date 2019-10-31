import tensorflow as tf


class LayerStore:
    STORE = {}

    def __init__(self):
        pass

    def get_layer(self, shared, name, func, *args, **kwargs):
        layer = None
        if shared:
            layer = LayerStore.STORE.get(name)

        if not layer:
            layer = func(*args, **kwargs)

        if shared:
            self.set_layer(name, layer)

        return layer

    def set_layer(self, name, layer):
        LayerStore.STORE[name] = layer


def to_2d(tensor):
    """Transform a tensor to 2 dimension"""

    shape = tensor.shape
    if len(shape) < 3:
        return tensor

    return tf.reshape(tensor, [-1, shape[-1]])


def create_attention_mask(input_mask):
    """input_mask = [batch_size, seq_length]

    As the attention defination, we have two tensors:
    from_tensor = BFH
    to_tensor = BTH
    We use the non-local method to compute the scores. And we don't want to add
    masked positions to the scores, so we need a mask to remove these positions
    before applying softmax.

    Assume we want to mask from_tensor as [1, 1, 1, 0, 0] (the list length is F)
    and to_tensor as [1, 1, 1, 1, 0] (the list length is T), then we can add following
    matrix to the scores to remove these unneeded positions.

                T
        -------------------
        |  1, 1, 1, 1, 0  |   -> use the row to softmax
        |  1, 1, 1, 1, 0  |
      F |  1, 1, 1, 1, 0  |
        |  0, 0, 0, 0, 0  |
        |  0, 0, 0, 0, 0  |

    or:
                T
        -------------------
        |  1, 1, 1, 1, 0  |
        |  1, 1, 1, 1, 0  |
      F |  1, 1, 1, 1, 0  |
        |  1, 1, 1, 1, 0  |
        |  1, 1, 1, 1, 0  |
    """

    # a_mask = [batch_size, 1, seq_length]
    a_mask = tf.expand_dims(input_mask, 1)
    # ones = [batch_size, seq_length, 1]
    ones = tf.ones_like(tf.expand_dims(input_mask, 2))
    return ones * a_mask
