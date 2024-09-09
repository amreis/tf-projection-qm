import tensorflow as tf

from tensorflow_projection_qm.util import distance


@tf.function
def trustworthiness_impl(X, X_2d, k) -> tf.Tensor:
    k = tf.cast(k, tf.int32)
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    n = tf.shape(D_high)[0]
    k = tf.minimum(k, n - 1)

    nn_orig = distance.sort_distances(D_high)
    nn_proj = distance.sort_distances(D_low)
    ixs_orig = tf.argsort(nn_orig)

    knn_orig = nn_orig[:, 1 : k + 1]
    knn_proj = nn_proj[:, 1 : k + 1]

    U_i = tf.sparse.to_dense(tf.sets.difference(knn_proj, knn_orig), default_value=-1)
    pre_trust = tf.where(
        U_i >= 0, tf.gather(ixs_orig, tf.where(U_i >= 0, U_i, 0), batch_dims=-1) - k, 0
    )
    trust = tf.reduce_sum(pre_trust, -1)
    trust_t = tf.cast(trust, tf.float64)
    k = tf.cast(k, tf.float64)
    n = tf.cast(n, tf.float64)

    return tf.squeeze(1 - tf.math.multiply_no_nan(2 / (k * (2 * n - 3 * k - 1)), trust_t))


def trustworthiness(X, X_2d, k: int) -> tf.Tensor:
    return tf.reduce_mean(trustworthiness_impl(X, X_2d, tf.constant(k)))
