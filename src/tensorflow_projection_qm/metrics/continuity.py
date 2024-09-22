from typing import Optional

import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import LocalizableMetric
from tensorflow_projection_qm.util import distance


@tf.function
def continuity_impl(X, X_2d, k) -> tf.Tensor:
    k = tf.cast(k, tf.int32)
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    n = tf.shape(D_high)[0]
    k = tf.minimum(k, n - 1)

    if 2 * k < n:
        norm_factor = k * (2 * n - 3 * k - 1) / 2
    else:
        norm_factor = (n - k) * (n - k - 1) / 2

    knn_orig = distance.nearest_k(D_high, k=k + 1)[1][:, 1:]
    nn_proj = distance.sort_distances(D_low)
    ixs_proj = tf.argsort(nn_proj)

    knn_proj = nn_proj[:, 1 : k + 1]

    V_i = tf.sparse.to_dense(tf.sets.difference(knn_orig, knn_proj), default_value=-1)
    pre_cont = tf.where(
        V_i >= 0, tf.gather(ixs_proj, tf.where(V_i >= 0, V_i, 0), batch_dims=-1) - k, 0
    )
    cont = tf.reduce_sum(pre_cont, -1)
    cont_t = tf.cast(cont, tf.float64)
    k = tf.cast(k, tf.float64)
    n = tf.cast(n, tf.float64)

    return tf.squeeze(1 - tf.math.divide_no_nan(cont_t, norm_factor))


@tf.function
def class_aware_continuity_impl(X, X_2d, y, k):
    k = tf.cast(k, tf.int32)
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    n = tf.shape(D_high)[0]
    k = tf.minimum(k, n - 1)
    if 2 * k < n:
        norm_factor = k * (2 * n - 3 * k - 1) / 2
    else:
        norm_factor = (n - k) * (n - k - 1) / 2

    nn_proj = distance.sort_distances(D_low)
    ixs_proj = tf.argsort(nn_proj)

    knn_orig = distance.nearest_k(D_high, k=k + 1)[1][:, 1:]
    knn_proj = nn_proj[:, 1 : k + 1]

    missing = tf.sparse.to_dense(tf.sets.difference(knn_orig, knn_proj), default_value=-1)
    classes = tf.where(missing >= 0, tf.gather(y, tf.where(missing >= 0, missing, 0)), -1)

    V_i = tf.where(classes == y[:, None], missing, -1)
    pre_cont = tf.where(
        V_i >= 0, tf.gather(ixs_proj, tf.where(V_i >= 0, V_i, 0), batch_dims=-1) - k, 0
    )
    cont = tf.reduce_sum(pre_cont, -1)
    cont_t = tf.cast(cont, tf.float64)
    k = tf.cast(k, tf.float64)
    n = tf.cast(n, tf.float64)

    return tf.squeeze(1 - tf.math.multiply_no_nan(1 / norm_factor, cont_t))


def continuity(X, X_2d, k: int) -> tf.Tensor:
    return tf.reduce_mean(continuity_impl(X, X_2d, tf.constant(k)))


def continuity_with_local(X, X_2d, k: int) -> tuple[tf.Tensor, tf.Tensor]:
    per_point = continuity_impl(X, X_2d, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


def class_aware_continuity(X, X_2d, y, k):
    return tf.reduce_mean(class_aware_continuity_impl(X, X_2d, y, tf.constant(k)))


def class_aware_continuity_with_local(X, X_2d, y, k):
    per_point = class_aware_continuity_impl(X, X_2d, y, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


class Continuity(LocalizableMetric):
    name = "continuity"
    _fn = continuity_impl

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


class ClassAwareContinuity(LocalizableMetric):
    name = "class_aware_continuity"
    _fn = class_aware_continuity_impl

    def __init__(self, k: Optional[int] = None, n_classes: Optional[int] = None) -> None:
        super().__init__()
        self.k = k
        self.n_classes = n_classes

    @property
    def config(self):
        return {"k": self.k}

    def measure(self, X, X_2d, y):
        return self._measure_impl(X, X_2d, y, self.k)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"], args["y"])
