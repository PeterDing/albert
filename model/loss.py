import tensorflow as tf


def sparse_categorical_crossentropy(y_true, y_pred, weights=None, from_logits=False, axis=-1):
    depth = y_pred.shape[-1]

    if from_logits:
        y_prob = tf.nn.log_softmax(y_pred, axis=-1)
    else:
        y_prob = y_pred

    y_true = tf.reshape(y_true, [-1])
    y_true_one_hot = tf.one_hot(y_true, depth=depth)

    # loss = [batch_size * seq_length, 1]
    loss = -tf.reduce_sum(y_prob * y_true_one_hot, axis=[-1])

    if weights is None:
        weights = tf.ones_like(y_true)
    weights = tf.cast(weights, dtype=loss.dtype)
    weights = tf.reshape(weights, (-1,))

    numerator = tf.reduce_sum(loss * weights)
    denominator = tf.reduce_sum(weights) + 1e-5
    loss = numerator / denominator

    return loss
