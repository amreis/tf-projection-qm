from typing import Optional

import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import LocalizableMetric
from tensorflow_projection_qm.util import distance


@tf.function
def false_neighbors_impl(X, X_2d, k):
    k = tf.cast(k, tf.int32)

    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    _, knn_orig = distance.nearest_k(D_high, k)
    _, knn_proj = distance.nearest_k(D_low, k)

    false_neighbors = tf.sparse.reduce_sum(
        tf.sparse.map_values(tf.ones_like, tf.sets.difference(knn_proj, knn_orig)), axis=-1
    )
    false_neighbors /= k

    return false_neighbors


@tf.function
def missing_neighbors_impl(X, X_2d, k):
    k = tf.cast(k, tf.int32)

    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    _, knn_orig = distance.nearest_k(D_high, k)
    _, knn_proj = distance.nearest_k(D_low, k)

    missing_neighbors = tf.sparse.reduce_sum(
        tf.sparse.map_values(tf.ones_like, tf.sets.difference(knn_orig, knn_proj)), axis=-1
    )
    missing_neighbors /= k

    return missing_neighbors


@tf.function
def true_neighbors_impl(X, X_2d, k):
    k = tf.cast(k, tf.int32)

    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    _, knn_orig = distance.nearest_k(D_high, k)
    _, knn_proj = distance.nearest_k(D_low, k)

    true_neighbors = tf.sparse.reduce_sum(
        tf.sparse.map_values(tf.ones_like, tf.sets.intersection(knn_orig, knn_proj)), axis=-1
    )
    true_neighbors /= k
    return true_neighbors


def false_neighbors(X, X_2d, k):
    return tf.reduce_mean(false_neighbors_impl(X, X_2d, tf.constant(k)))


def false_neighbors_with_local(X, X_2d, k):
    per_point = false_neighbors_impl(X, X_2d, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


def missing_neighbors(X, X_2d, k):
    return tf.reduce_mean(missing_neighbors_impl(X, X_2d, tf.constant(k)))


def missing_neighbors_with_local(X, X_2d, k):
    per_point = missing_neighbors_impl(X, X_2d, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


def true_neighbors(X, X_2d, k):
    return tf.reduce_mean(true_neighbors_impl(X, X_2d, tf.constant(k)))


def true_neighbors_with_local(X, X_2d, k):
    per_point = true_neighbors_impl(X, X_2d, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


class FalseNeighbors(LocalizableMetric):
    name = "false_neighbors"

    def __init__(self, k: Optional[int] = None) -> None:
        super().__init__()
        self.k = k

    @property
    def config(self):
        return {"k": self.k}

    def measure(self, X, X_2d):
        if self._with_local:
            return false_neighbors_with_local(X, X_2d, self.k)
        return false_neighbors(X, X_2d, self.k)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])


class MissingNeighbors(LocalizableMetric):
    name = "missing_neighbors"

    def __init__(self, k: Optional[int] = None) -> None:
        super().__init__()
        self.k = k

    @property
    def config(self):
        return {"k": self.k}

    def measure(self, X, X_2d):
        if self._with_local:
            return missing_neighbors_with_local(X, X_2d, self.k)
        return missing_neighbors(X, X_2d, self.k)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])


class TrueNeighbors(LocalizableMetric):
    name = "true_neighbors"

    def __init__(self, k: Optional[int] = None) -> None:
        super().__init__()
        self.k = k

    @property
    def config(self):
        return {"k": self.k}

    def measure(self, X, X_2d):
        if self._with_local:
            return true_neighbors_with_local(X, X_2d, self.k)
        return true_neighbors(X, X_2d, self.k)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
