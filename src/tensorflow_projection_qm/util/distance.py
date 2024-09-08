import tensorflow as tf


@tf.function
def sort_distances(D) -> tf.Tensor:
    """Sorts a distance matrix using tf.argsort with a corrective factor

    This function adds a "correction" to D, ensuring that, O_i0 = i for all i
    (where O is the output of this function). This is not guaranteed by
    tf.argsort for matrices where D_ij is 0 in a given row i for more than a
    single j. However, since we use distance matrices for k-nearest neighbors
    computations, it is common to assume that the first nearest neighbor of
    element i is itself (hence the need for this function).
    This function internally creates an *invalid* distance matrix
    due to the negative entries added to it, but it prevents such invalid
    matrices from leaking out to the caller.

    Args:
        D (tf.Tensor, np.ndarray): A matrix with distances (all entries should be nonnegative)

    Returns:
        tf.Tensor: a Tensor of the same shape as D, with indices that sort the rows
        (see tf.argsort).
    """
    return tf.argsort(D - 1.0 * tf.eye(tf.shape(D)[0], dtype=tf.float64))


@tf.function
def nearest_k(D, k) -> tuple[tf.Tensor, tf.Tensor]:
    return tf.math.top_k(-(D - tf.eye(tf.shape(D)[0])), k=k)


@tf.function
def psqdist(X) -> tf.Tensor:
    row_norms = tf.reshape(tf.reduce_sum(X**2, 1), [-1, 1])

    return row_norms - 2 * tf.matmul(X, X, transpose_b=True) + tf.reshape(row_norms, [1, -1])


@tf.function
def csqdist(X, Y) -> tf.Tensor:
    x_norms = tf.reshape(tf.reduce_sum(X**2, 1), [-1, 1])
    y_norms = tf.reshape(tf.reduce_sum(Y**2, 1), [1, -1])

    return x_norms - 2 * tf.matmul(X, Y, transpose_b=True) + y_norms
