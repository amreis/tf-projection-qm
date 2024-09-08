"""
Module implementing Mean Relative Rank Error metrics

The metrics are implemented as described in Nonlinear Dimensionality Reduction,
by Lee and Verleysen (2007), Chapter 6.
"""

import tensorflow as tf

from tensorflow_projection_qm.util import distance


@tf.function
def mrre_data_impl(X, X_2d, k):
    n = tf.shape(X)[0]
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    nn_orig = distance.sort_distances(D_high)
    ixs_orig = tf.argsort(nn_orig)
    nn_proj = distance.sort_distances(D_low)
    knn_proj = nn_proj[:, 1 : k + 1]

    orig_ranks = tf.gather(ixs_orig, knn_proj, batch_dims=-1)
    unnormalized = tf.reduce_sum(
        tf.abs(orig_ranks - tf.expand_dims(tf.range(1, k + 1), 0)) / orig_ranks
    )

    C = tf.cast(n, tf.float64) * tf.reduce_sum(
        tf.abs(2 * tf.range(1, k + 1) - n - 1) / tf.range(1, k + 1)
    )

    return (1 / C) * unnormalized


@tf.function
def mrre_proj_impl(X, X_2d, k):
    n = tf.shape(X)[0]
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    nn_orig = distance.sort_distances(D_high)
    knn_orig = nn_orig[:, 1 : k + 1]
    nn_proj = distance.sort_distances(D_low)
    ixs_proj = tf.argsort(nn_proj)

    proj_ranks = tf.gather(ixs_proj, knn_orig, batch_dims=-1)
    unnormalized = tf.reduce_sum(
        tf.abs(proj_ranks - tf.expand_dims(tf.range(1, k + 1), 0)) / proj_ranks
    )

    C = tf.cast(n, tf.float64) * tf.reduce_sum(
        tf.abs(2 * tf.range(1, k + 1) - n - 1) / tf.range(1, k + 1)
    )

    return (1 / C) * unnormalized


def mrre_data(X, X_2d, k):
    """Compute the MRRE considering the data-space neighborhoods as ground truth.

    This corresponds to Equation 6.4 in Nonlinear Dimensionality Reduction.

    Args:
        X (tf.Tensor, np.ndarray): the data matrix for high-dimensional data
        X_2d (tf.Tensor, np.ndarray): the data matrix for projected data
        k (int): the neighborhood size to consider.

    Returns:
        tf.Tensor: a Tensor containing a single scalar, the metric value.
    """
    return mrre_data_impl(X, X_2d, tf.constant(k))


def mrre_proj(X, X_2d, k):
    """Compute the MRRE considering the projection-space neighborhoods as ground truth.

    This corresponds to Equation 6.3 in Nonlinear Dimensionality Reduction.

    Args:
        X (tf.Tensor, np.ndarray): the data matrix for high-dimensional data
        X_2d (tf.Tensor, np.ndarray): the data matrix for projected data
        k (int): the neighborhood size to consider.

    Returns:
        tf.Tensor: a Tensor containing a single scalar, the metric value.
    """
    return mrre_proj_impl(X, X_2d, tf.constant(k))
