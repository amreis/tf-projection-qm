import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import Metric
from tensorflow_projection_qm.util import distance


@tf.function
def shepard_goodness_impl(X, X_2d):
    flat_d_high = distance.flat_psqdist(X)
    flat_d_low = distance.flat_psqdist(X_2d)
    n = tf.cast(tf.shape(flat_d_high)[0], tf.float32)

    ranks_high = tf.cast(tf.argsort(tf.argsort(flat_d_high)), tf.float32)
    ranks_low = tf.cast(tf.argsort(tf.argsort(flat_d_low)), tf.float32)

    # Assuming no ties, we can use this formula.
    return 1 - 6 * tf.reduce_mean(tf.math.squared_difference(ranks_high, ranks_low)) / (n**2 - 1)


def shepard_goodness(X, X_2d):
    return shepard_goodness_impl(X, X_2d)


class ShepardGoodness(Metric):
    name = "shepard_goodness"
    _fn = shepard_goodness_impl

    def __init__(self) -> None:
        super().__init__()

    @property
    def config(self):
        return {}

    def measure(self, X, X_2d):
        return self._measure_impl(X, X_2d)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
