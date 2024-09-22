from typing import Optional

import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import LocalizableMetric
from tensorflow_projection_qm.util import distance


@tf.function
def trustworthiness_impl(X, X_2d, k) -> tf.Tensor:
    k = tf.cast(k, tf.int32)
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    n = tf.shape(D_high)[0]
    k = tf.minimum(k, n - 1)
    if 2 * k < n:
        norm_factor = k * (2 * n - 3 * k - 1) / 2
    else:
        norm_factor = (n - k) * (n - k - 1) / 2

    nn_orig = distance.sort_distances(D_high)
    ixs_orig = tf.argsort(nn_orig)

    knn_orig = nn_orig[:, 1 : k + 1]
    knn_proj = distance.nearest_k(D_low, k=k + 1)[1][:, 1:]

    U_i = tf.sparse.to_dense(tf.sets.difference(knn_proj, knn_orig), default_value=-1)
    pre_trust = tf.where(
        U_i >= 0, tf.gather(ixs_orig, tf.where(U_i >= 0, U_i, 0), batch_dims=-1) - k, 0
    )
    trust = tf.reduce_sum(pre_trust, -1)
    trust_t = tf.cast(trust, tf.float64)
    k = tf.cast(k, tf.float64)
    n = tf.cast(n, tf.float64)

    return tf.squeeze(1 - tf.math.divide_no_nan(trust_t, norm_factor))


def trustworthiness(X, X_2d, k: int) -> tf.Tensor:
    return tf.reduce_mean(trustworthiness_impl(X, X_2d, tf.constant(k)))


def trustworthiness_with_local(X, X_2d, k: int) -> tuple[tf.Tensor, tf.Tensor]:
    per_point = trustworthiness_impl(X, X_2d, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


@tf.function
def class_aware_trustworthiness_impl(X, X_2d, y, k):
    k = tf.cast(k, tf.int32)
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    n = tf.shape(D_high)[0]
    k = tf.minimum(k, n - 1)
    if 2 * k < n:
        norm_factor = k * (2 * n - 3 * k - 1) / 2
    else:
        norm_factor = (n - k) * (n - k - 1) / 2

    nn_orig = distance.sort_distances(D_high)
    ixs_orig = tf.argsort(nn_orig)

    knn_orig = nn_orig[:, 1 : k + 1]
    knn_proj = distance.nearest_k(D_low, k=k + 1)[1][:, 1:]

    false = tf.sparse.to_dense(tf.sets.difference(knn_proj, knn_orig), default_value=-1)
    classes = tf.where(false >= 0, tf.gather(y, tf.where(false >= 0, false, 0)), -1)

    U_i = tf.where(classes != y[:, None], false, -1)
    pre_trust = tf.where(
        U_i >= 0, tf.gather(ixs_orig, tf.where(U_i >= 0, U_i, 0), batch_dims=-1) - k, 0
    )
    trust = tf.reduce_sum(pre_trust, -1)
    trust_t = tf.cast(trust, tf.float64)
    k = tf.cast(k, tf.float64)
    n = tf.cast(n, tf.float64)

    return tf.squeeze(1 - tf.math.multiply_no_nan(1 / norm_factor, trust_t))


def class_aware_trustworthiness(X, X_2d, y, k):
    return tf.reduce_mean(class_aware_trustworthiness_impl(X, X_2d, y, tf.constant(k)))


def class_aware_trustworthiness_with_local(X, X_2d, y, k):
    per_point = class_aware_trustworthiness_impl(X, X_2d, y, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


class Trustworthiness(LocalizableMetric):
    name = "trustworthiness"
    _fn = trustworthiness_impl

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


class ClassAwareTrustworthiness(LocalizableMetric):
    name = "class_aware_trustworthiness"
    _fn = class_aware_trustworthiness_impl

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
