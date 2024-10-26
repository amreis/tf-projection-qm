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

    >>> D = tf.convert_to_tensor([
    ...         [0.0, 1.0, 0.0],
    ...         [1.0, 0.0, 0.5],
    ...         [0.0, 0.5, 0.0]], dtype=tf.float64)
    >>> sort_distances(D).numpy()
    array([[0, 2, 1],
           [1, 2, 0],
           [2, 0, 1]], dtype=int32)
    >>> tf.argsort(D).numpy()
    array([[0, 2, 1],
           [1, 2, 0],
           [0, 2, 1]], dtype=int32)

    Notice that for tf.argsort on the third row, the first element is *not* 2.

    Args:
        D (tf.Tensor, np.ndarray): A matrix with distances (all entries should be nonnegative)

    Returns:
        tf.Tensor: a Tensor of the same shape as D, with indices that sort the rows
        (see tf.argsort).
    """
    return tf.argsort(D - 1.0 * tf.eye(tf.shape(D)[0], dtype=D.dtype))


@tf.function
def nearest_k(D, k) -> tuple[tf.Tensor, tf.Tensor]:
    """Adaptation of tf.math.top_k to return smallest k instead.

    Please keep in mind that this function follows the invariant that the closest neighbor
    of a given data point is *itself*. If you want to find k-nearest neighbors without
    counting the element itself, use k+1 as an argument for this function and slice the
    return values accordingly. For instance:

    >>> D = tf.convert_to_tensor([
    ...         [0.0, 1.0, 0.0],
    ...         [1.0, 0.0, 0.5],
    ...         [0.0, 0.5, 0.0]], dtype=tf.float64)
    >>> values, indices_2nn = nearest_k(D, k=2)
    >>> indices_2nn.numpy()
    array([[0, 2],
           [1, 2],
           [2, 0]], dtype=int32)
    >>> values, indices_3nn = nearest_k(D, k=2 + 1)
    >>> indices_2nn_excl = indices_3nn[..., 1:]  # or indices_3nn[:, 1:]
    >>> indices_2nn_excl.numpy()
    array([[2, 1],
           [2, 0],
           [0, 1]], dtype=int32)

    Args:
        D (tf.Tensor): an n x n Tensor containing distances (a distance matrix)
        k (tf.Tensor | int): a scalar, the number of nearest neighbors to return

    Returns:
        nearest_k (tf.Tensor, tf.Tensor): same as `tf.math.top_k` but for the smallest k.
    """
    return tf.math.top_k(-(D - tf.eye(tf.shape(D)[0], dtype=D.dtype)), k=k)


@tf.function
def psqdist(X) -> tf.Tensor:
    X = tf.cast(X, tf.float64)
    row_norms = tf.reshape(tf.reduce_sum(X**2, 1), [-1, 1])

    dists = row_norms - 2 * tf.matmul(X, X, transpose_b=True) + tf.reshape(row_norms, [1, -1])
    return tf.maximum(dists, 0.0)


@tf.function
def csqdist(X, Y) -> tf.Tensor:
    X = tf.cast(X, tf.float64)
    Y = tf.cast(Y, tf.float64)

    x_norms = tf.reshape(tf.reduce_sum(X**2, 1), [-1, 1])
    y_norms = tf.reshape(tf.reduce_sum(Y**2, 1), [1, -1])

    dists = x_norms - 2 * tf.matmul(X, Y, transpose_b=True) + y_norms
    return tf.maximum(dists, 0.0)


@tf.function
def flat_psqdist(X) -> tf.Tensor:
    D = psqdist(X)

    mask = tf.ones_like(D)
    mask = tf.linalg.band_part(mask, 0, -1) - tf.linalg.band_part(mask, 0, 0)
    mask = tf.cast(mask, tf.bool)

    return tf.boolean_mask(D, mask)
