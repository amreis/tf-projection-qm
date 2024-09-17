from typing import Optional
import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import LocalizableMetric
from tensorflow_projection_qm.util import distance


@tf.function
def procrustes_impl(X, X_2d, k):
    # TODO(amreis): when adding docs, please note that we support comparing to
    # a rotated *and rescaled* version of the 2D embedding.
    n = tf.shape(X)[0]
    k = tf.minimum(k, n)

    H = tf.eye(k, dtype=X.dtype)
    H -= tf.ones_like(H) / tf.cast(k, H.dtype)

    knn_orig = distance.nearest_k(distance.psqdist(X), k=k)[1]
    # batch of k-nearest neighbors per point, including the point itself.
    data_neighborhoods = tf.gather(X, knn_orig)  # shape = (n, k, data_dim)
    # ZADU's implementation uses knn_proj here and it's *wrong*. They're
    # comparing two different sets of points. We compare the same set of
    # points. The data neighborhoods and their embeddings.
    proj_neighborhoods = tf.gather(X_2d, knn_orig)  # shape = (n, k, proj_dim)

    Z = tf.linalg.einsum("bij,il,blm->bjm", data_neighborhoods, H, proj_neighborhoods)
    s_i, u_i, v_i = tf.linalg.svd(Z)
    c_i = tf.reduce_sum(s_i, -1) / tf.norm(proj_neighborhoods, ord="fro", axis=(-2, -1)) ** 2

    A_i = tf.matmul(u_i, v_i, transpose_b=True)
    procrustes_error = data_neighborhoods - tf.matmul(
        proj_neighborhoods, c_i[:, tf.newaxis, tf.newaxis] * A_i, transpose_b=True
    )
    procrustes_i = (
        tf.norm(tf.matmul(H[tf.newaxis, ...], procrustes_error), ord="fro", axis=(-2, -1)) ** 2
    )

    centered_data_neighs = tf.matmul(H[tf.newaxis, ...], data_neighborhoods)
    denom = tf.norm(centered_data_neighs, ord="fro", axis=(-2, -1)) ** 2

    return procrustes_i / denom


def procrustes(X, X_2d, k):
    return tf.reduce_mean(procrustes_impl(X, X_2d, tf.constant(k)))


def procrustes_with_local(X, X_2d, k):
    per_point = procrustes_impl(X, X_2d, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


class Procrustes(LocalizableMetric):
    name = "procrustes"

    def __init__(self, k: Optional[int] = None) -> None:
        super().__init__()
        self.k = k

    @property
    def config(self):
        return {"k": self.k}

    def measure(self, X, X_2d):
        if self._with_local:
            return procrustes_with_local(X, X_2d, self.k)
        return procrustes(X, X_2d, self.k)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
