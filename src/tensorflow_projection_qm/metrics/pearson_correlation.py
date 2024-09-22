import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import Metric
from tensorflow_projection_qm.util import distance


@tf.function
def pearson_correlation_impl(X, X_2d):
    flat_d_high = tf.sqrt(distance.flat_psqdist(X))
    flat_d_low = tf.sqrt(distance.flat_psqdist(X_2d))

    mean_high = tf.reduce_mean(flat_d_high, axis=-1)
    mean_low = tf.reduce_mean(flat_d_low, axis=-1)

    diff_mean_high = flat_d_high - mean_high
    diff_mean_low = flat_d_low - mean_low

    return tf.reduce_sum(
        tf.math.l2_normalize(diff_mean_high, axis=-1) * tf.math.l2_normalize(diff_mean_low, axis=-1)
    )


def pearson_correlation(X, X_2d):
    return pearson_correlation_impl(X, X_2d)


class PearsonCorrelation(Metric):
    name = "pearson_correlation"
    _fn = pearson_correlation_impl

    def __init__(self) -> None:
        super().__init__()

    @property
    def config(self):
        return {}

    def measure(self, X, X_2d):
        return self._measure_impl(X, X_2d)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
