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

    nn_orig = distance.sort_distances(D_high)
    nn_proj = distance.sort_distances(D_low)
    ixs_proj = tf.argsort(nn_proj)

    knn_orig = nn_orig[:, 1 : k + 1]
    knn_proj = nn_proj[:, 1 : k + 1]

    V_i = tf.sparse.to_dense(tf.sets.difference(knn_orig, knn_proj), default_value=-1)
    pre_cont = tf.where(
        V_i >= 0, tf.gather(ixs_proj, tf.where(V_i >= 0, V_i, 0), batch_dims=-1) - k, 0
    )
    cont = tf.reduce_sum(pre_cont, -1)
    cont_t = tf.cast(cont, tf.float64)
    k = tf.cast(k, tf.float64)
    n = tf.cast(n, tf.float64)

    return tf.squeeze(1 - tf.math.multiply_no_nan(2 / (k * (2 * n - 3 * k - 1)), cont_t))


def continuity(X, X_2d, k: int) -> tf.Tensor:
    return tf.reduce_mean(continuity_impl(X, X_2d, tf.constant(k)))


def continuity_with_local(X, X_2d, k: int) -> tuple[tf.Tensor, tf.Tensor]:
    per_point = continuity_impl(X, X_2d, tf.constant(k))
    return tf.reduce_mean(per_point), per_point


class Continuity(LocalizableMetric):
    name = "continuity"

    def __init__(self, k: Optional[int] = None) -> None:
        super().__init__()
        self.k = k

    @property
    def config(self):
        return {"k": self.k}

    def measure(self, X, X_2d):
        if self._with_local:
            return continuity_with_local(X, X_2d, self.k)
        return continuity(X, X_2d, self.k)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
