import tensorflow as tf
import numpy as np


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """

    # \frac{1}{2}\bigg(1 + tanh\big(\sqrt{\frac{1}{\pi}}(x + 0.044 * x^3))\big)\bigg)x
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
