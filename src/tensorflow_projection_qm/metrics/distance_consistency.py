from typing import Optional

import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import Metric


@tf.function
def distance_consistency_impl(X_2d, y, n_classes):
    y_sort_ixs = tf.argsort(y)

    X_2d = tf.gather(X_2d, y_sort_ixs)  # re-order, grouped per class
    y_sorted = tf.gather(y, y_sort_ixs)
    uniq, _ = tf.unique(y_sorted)
    centroids = tf.reduce_mean(
        tf.ragged.stack_dynamic_partitions(X_2d, y_sorted, n_classes), axis=1
    )

    closest_centroid = tf.argmin(
        tf.linalg.norm(tf.expand_dims(X_2d, 1) - centroids, axis=-1), axis=1
    )
    closest_centroid = tf.gather(uniq, closest_centroid)

    return tf.reduce_mean(tf.cast(closest_centroid == y_sorted, tf.float64))


def distance_consistency(X_2d, y, n_classes):
    return distance_consistency_impl(X_2d, y, n_classes)


class DistanceConsistency(Metric):
    name = "distance_consistency"
    _fn = distance_consistency_impl

    def __init__(self, n_classes: Optional[int] = None) -> None:
        super().__init__()
        self.n_classes = n_classes

    @property
    def config(self):
        return {"n_classes": self.n_classes}

    def measure(self, X_2d, y):
        return self._measure_impl(X_2d, y, self.n_classes)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X_2d"], args["y"])
