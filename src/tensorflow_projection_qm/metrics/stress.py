import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import Metric
from tensorflow_projection_qm.util import distance


@tf.function
def raw_stress_from_distances(D_high, D_low):
    return tf.reduce_sum((D_high - D_low) ** 2)


@tf.function
def raw_stress(X, X_2d):
    D_high = tf.sqrt(distance.psqdist(X))
    D_low = tf.sqrt(distance.psqdist(X_2d))

    return raw_stress_from_distances(D_high, D_low)


@tf.function
def normalized_stress_from_distances(D_high, D_low):
    return tf.reduce_sum((D_high - D_low) ** 2) / tf.reduce_sum(D_high**2)


@tf.function
def normalized_stress(X, X_2d):
    D_high = tf.sqrt(distance.psqdist(X))
    D_low = tf.sqrt(distance.psqdist(X_2d))

    return normalized_stress_from_distances(D_high, D_low)


@tf.function
def scaled_stress_from_distances(D_high, D_low, alpha=1.0):
    return normalized_stress_from_distances(D_high, alpha * D_low)


@tf.function
def scaled_stress(X, X_2d, alpha=1.0):
    return normalized_stress(X, alpha * X_2d)


@tf.function
def scale_normalized_stress_from_distances(D_high, D_low):
    opt_alpha = tf.reduce_sum(D_high * D_low) / tf.reduce_sum(D_low**2)

    return normalized_stress_from_distances(D_high, opt_alpha * D_low)


@tf.function
def scale_normalized_stress(X, X_2d):
    """A scale-invariant version of the Stress metric.

    From the paper "Normalized Stress" is Not Normalized: How to Interpret Stress Correctly,
    by Smelser et al.

    Args:
        X (tf.Tensor, np.ndarray): The original data matrix.
        X_2d (tf.Tensor, np.ndarray): The projected data matrix.

    Returns:
        tf.Tensor: a tensor containing a single scalar, the value of the metric.
    """
    D_high = tf.sqrt(distance.psqdist(X))
    D_low = tf.sqrt(distance.psqdist(X_2d))

    return scale_normalized_stress_from_distances(D_high, D_low)


class NormalizedStress(Metric):
    name = "normalized_stress"
    _fn = normalized_stress

    def __init__(self) -> None:
        super().__init__()

    @property
    def config(self):
        return {}

    def measure(self, X, X_2d):
        return self._measure_impl(X, X_2d)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])


class ScaleNormalizedStress(Metric):
    name = "scale_normalized_stress"
    _fn = scale_normalized_stress

    def __init__(self) -> None:
        super().__init__()

    @property
    def config(self):
        return {}

    def measure(self, X, X_2d):
        return self._measure_impl(X, X_2d)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
