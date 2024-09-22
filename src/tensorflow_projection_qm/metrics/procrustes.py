"""Implementation of the Procrustes quality metric.

Described in "Local procrustes for manifold embedding: a measure of embedding quality
and embedding algorithms", by Goldbert and Ritov (2009).

The implementation below reflects equation (7), where the procedure aims to "align" the
high and low dimensional spaces with rotation, translation, *and* scaling. Since this
support for scaling is aimed at conformal maps, we might drop it or make it optional.
In any case, the public API is the same.

In the paper, you'll see the matrix H being used quite often. H is a centering matrix
(i.e., an idempotent matrix that makes the mean of the points it's applied to become 0).
We have explicitly used mean subtraction here as it's more space and time efficient.
Also, the paper uses tr(X'X) and ||X||_F^2 (which are the same thing); while we could use
tensorflow's `norm` function, we choose to just square and add the elements of the matrices,
as that's what's really happening in this notation.
"""

from typing import Optional

import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import LocalizableMetric
from tensorflow_projection_qm.util import distance


def _svd_part(Z_i):
    s_i, u_i, v_i = tf.linalg.svd(Z_i)
    return s_i, u_i, v_i


@tf.function
def procrustes_impl(X, X_2d, k):
    n = tf.shape(X)[0]
    k = tf.minimum(k, n)
    knn_orig = distance.nearest_k(distance.psqdist(X), k=k)[1]
    # batch of k-nearest neighbors per point, including the point itself.
    data_neighborhoods = tf.gather(X, knn_orig)  # shape = (n, k, data_dim)
    # ZADU's implementation uses knn_proj here and it's *wrong*. They're
    # comparing two different sets of points. We compare the same set of
    # points. The data neighborhoods and *their* embeddings.
    proj_neighborhoods = tf.gather(X_2d, knn_orig)  # shape = (n, k, proj_dim)

    Z = tf.matmul(
        data_neighborhoods,
        proj_neighborhoods - tf.reduce_mean(proj_neighborhoods, -2, keepdims=True),
        transpose_a=True,
    )
    if tf.shape(X)[1] ** 2 >= 2_000_000:
        s_i, u_i, v_i = tf.map_fn(
            _svd_part,
            Z,
            fn_output_signature=(
                tf.TensorSpec(shape=[None], dtype=X.dtype),
                tf.TensorSpec(shape=[None, None], dtype=X.dtype),
                tf.TensorSpec(shape=[None, None], dtype=X.dtype),
            ),
            swap_memory=True,
        )
    else:
        s_i, u_i, v_i = tf.linalg.svd(Z)

    c_i = tf.reduce_sum(s_i, -1) / tf.reduce_sum(proj_neighborhoods**2, axis=(-2, -1))

    A_i = tf.matmul(u_i, v_i, transpose_b=True)
    procrustes_error = data_neighborhoods - tf.matmul(
        proj_neighborhoods, c_i[:, tf.newaxis, tf.newaxis] * A_i, transpose_b=True
    )
    procrustes_i = tf.reduce_sum(
        (procrustes_error - tf.reduce_mean(procrustes_error, -2, keepdims=True)) ** 2,
        axis=(-2, -1),
    )

    centered_data_neighs = data_neighborhoods - tf.reduce_mean(
        data_neighborhoods, -2, keepdims=True
    )
    denom = tf.maximum(tf.reduce_sum(centered_data_neighs**2, axis=(-2, -1)), 1e-12)

    return procrustes_i / denom


def procrustes(X, X_2d, k):
    return tf.reduce_mean(procrustes_impl(X, X_2d, tf.constant(k)))


def procrustes_with_local(X, X_2d, k):
    per_point = procrustes_impl(X, X_2d, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


class Procrustes(LocalizableMetric):
    name = "procrustes"
    _fn = procrustes_impl

    def __init__(self, k: Optional[int] = None) -> None:
        super().__init__()
        self.k = k

    @property
    def config(self):
        return {"k": self.k}

    def measure(self, X, X_2d):
        return self._measure_impl(X, X_2d, self.k)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
