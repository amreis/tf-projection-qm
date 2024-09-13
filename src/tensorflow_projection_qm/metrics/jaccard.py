from typing import Optional, Union

import tensorflow as tf

from tensorflow_projection_qm.util import distance

from .metric import LocalizableMetric


@tf.function
def jaccard_impl(X, X_2d, k) -> tf.Tensor:
    k = tf.cast(k, tf.int32)
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    n = tf.shape(D_high)[0]
    k = tf.minimum(k, n - 1)

    knn_orig = distance.nearest_k(D_high, k=k + 1)[1][..., 1:]
    knn_proj = distance.nearest_k(D_low, k=k + 1)[1][..., 1:]

    intersection = tf.sets.intersection(knn_orig, knn_proj)
    union = tf.sets.union(knn_orig, knn_proj)

    n_intersection = tf.sparse.reduce_sum(tf.sparse.map_values(tf.ones_like, intersection), axis=-1)
    n_union = tf.sparse.reduce_sum(tf.sparse.map_values(tf.ones_like, union), axis=-1)

    return n_intersection / n_union
    # intersection_sizes = tf.sparse.reduce_count ?


def jaccard(X, X_2d, k, *, with_local=False) -> Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
    """Calculate the Jaccard quality metric for a projection

    Args:
        X (tf.Tensor, np.ndarray): the high-dimensional data matrix
        X_2d (tf.Tensor, np.ndarray): the projection data matrix
        k (tf.Tensor): a scalar tensor containing the number of neighbors to consider
        with_local (bool, optional): if true, returns both the scalar value of the metric as well
            as the value for each data point. Defaults to False.

    Returns:
        - tf.Tensor: the scalar value of the Jaccard metric for the whole projection.
        - tf.Tensor, optional: the per-point value of the Jaccard metric, when `with_local`=True
    """

    per_point = jaccard_impl(X, X_2d, tf.constant(k))
    agg = tf.reduce_mean(per_point)
    if with_local:
        return agg, per_point
    return agg


def jaccard_with_local(X, X_2d, k) -> tuple[tf.Tensor, tf.Tensor]:
    per_point = jaccard_impl(X, X_2d, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


class Jaccard(LocalizableMetric):
    name = "jaccard"

    def __init__(self, k: Optional[int] = None) -> None:
        super().__init__()
        self.k = k

    @property
    def config(self):
        return {"k": self.k}

    def measure(self, X, X_2d, *, k: Optional[int] = None):
        k = self.k if self.k is not None else k
        if self._with_local:
            return jaccard_with_local(X, X_2d, k)
        return jaccard(X, X_2d, k)

    def measure_from_dict(self, data_dict: dict[str, object]):
        return self.measure(data_dict["X"], data_dict["X_2d"])