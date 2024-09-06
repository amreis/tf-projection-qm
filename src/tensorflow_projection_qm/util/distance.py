import tensorflow as tf


@tf.function
def psqdist(X) -> tf.Tensor:
    row_norms = tf.reshape(tf.reduce_sum(X**2, 1), [-1, 1])

    return row_norms - 2 * tf.matmul(X, X, transpose_b=True) + tf.reshape(row_norms, [1, -1])
