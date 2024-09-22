import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import LocalizableMetric
from tensorflow_projection_qm.util import distance


@tf.function
def average_local_error_impl(X, X_2d):
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    D_high /= tf.reduce_max(D_high)
    D_low /= tf.reduce_max(D_low)

    n = tf.shape(D_high)[0]
    avg_local_err = (1 / (n - 1)) * tf.reduce_sum(tf.abs(D_high - D_low), -1)

    return avg_local_err


def average_local_error(X, X_2d):
    return tf.reduce_mean(average_local_error_impl(X, X_2d))


def average_local_error_with_local(X, X_2d):
    per_point = average_local_error_impl(X, X_2d)
    return tf.reduce_mean(per_point), per_point


class AverageLocalError(LocalizableMetric):
    name = "average_local_error"
    _fn = average_local_error_impl

    def __init__(self) -> None:
        super().__init__()

    @property
    def config(self):
        return {}

    def measure(self, X, X_2d):
        return self._measure_impl(X, X_2d)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
