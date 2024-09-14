import numpy  np
import scipy.stats
import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import Metric
from tensorflow_projection_qm.util import distance


def pearson_correlation_impl(X, X_2d):
    @tf.function
    def _inner(X, X_2d):
        D_high = distance.psqdist(X)
        D_low = distance.psqdist(X_2d)

        mask = tf.ones_like(D_high)
        mask = tf.linalg.band_part(mask, 0, -1) - tf.linalg.band_part(mask, 0, 0)
        mask = tf.cast(mask, bool)

        flat_d_high = tf.boolean_mask(D_high, mask)
        flat_d_low = tf.boolean_mask(D_low, mask)
        return flat_d_high, flat_d_low

    flat_d_high, flat_d_low = _inner(X, X_2d)
    return tf.convert_to_tensor(scipy.stats.pearsonr(flat_d_high, flat_d_low)[0])


def pearson_correlation(X, X_2d):
    return pearson_correlation_impl(X, X_2d)


class PearsonCorrelation(Metric):
    name = "pearson_correlation"

    def __init__(self) -> None:
        super().__init__()

    @property
    def config(self):
        return {}

    def measure(self, X, X_2d):
        return pearson_correlation(X, X_2d)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
