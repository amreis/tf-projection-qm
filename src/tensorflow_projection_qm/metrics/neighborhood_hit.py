import tensorflow as tf

from tensorflow_projection_qm.util import distance


@tf.function
def neighborhood_hit_impl(X_2d, y, k):
    D_low = distance.psqdist(X_2d)
    _, topk_ixs = distance.nearest_k(D_low, k)

    return tf.reduce_mean(tf.cast(tf.gather(y, topk_ixs) == y[:, tf.newaxis], tf.float64), -1)


def neighborhood_hit(X_2d, y, k):
    return tf.reduce_mean(neighborhood_hit_impl(X_2d, y, tf.constant(k)))
