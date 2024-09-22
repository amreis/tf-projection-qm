from typing import Optional

import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import LocalizableMetric
from tensorflow_projection_qm.util import distance


@tf.function
def neighborhood_hit_impl(X_2d, y, k):
    D_low = distance.psqdist(X_2d)
    n = tf.shape(X_2d)[0]
    k = tf.minimum(k, n - 1)
    _, topk_ixs = distance.nearest_k(D_low, k=k + 1)
    topk_ixs = topk_ixs[:, 1:]

    return tf.reduce_mean(tf.cast(tf.gather(y, topk_ixs) == y[:, tf.newaxis], tf.float64), -1)


def neighborhood_hit(X_2d, y, k):
    return tf.reduce_mean(neighborhood_hit_impl(X_2d, y, tf.constant(k)))


def neighborhood_hit_with_local(X_2d, y, k) -> tuple[tf.Tensor, tf.Tensor]:
    per_point = neighborhood_hit_impl(X_2d, y, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


class NeighborhoodHit(LocalizableMetric):
    name = "neighborhood_hit"
    _fn = neighborhood_hit_impl

    def __init__(self, k: Optional[int] = None) -> None:
        super().__init__()
        self.k = k

    @property
    def config(self):
        return {"k": self.k}

    def measure(self, X_2d, y):
        return self._measure_impl(X_2d, y, self.k)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X_2d"], args["y"])
