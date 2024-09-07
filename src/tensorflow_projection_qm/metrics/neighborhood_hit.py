import tensorflow as tf

from tensorflow_projection_qm.util import distance


@tf.function
def neighborhood_hit_impl(X_2d, y, k):
    D_low = distance.psqdist(X_2d)

    tf.linalg.set_diag(
        D_low, -1e-10 * tf.ones(tf.shape(D_low)[0], dtype=tf.float64)
    )  # make sure closest point is always self.
    _, topk_ixs = tf.math.top_k(-D_low, k=k)

    return tf.reduce_mean(tf.cast(tf.gather(y, topk_ixs) == y[:, tf.newaxis], tf.float64), -1)


def neighborhood_hit(X_2d, y, k):
    return tf.reduce_mean(neighborhood_hit_impl(X_2d, y, tf.constant(k)))
