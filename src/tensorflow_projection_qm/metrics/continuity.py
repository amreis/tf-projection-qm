import tensorflow as tf

from tensorflow_projection_qm.util import distance


@tf.function
def continuity_impl(X, X_2d, k) -> tf.Tensor:
    k = tf.cast(k, tf.int32)
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    n = tf.shape(D_high)[0]
    D_high -= 1e-10 * tf.eye(n, dtype=tf.float64)
    D_low -= 1e-10 * tf.eye(n, dtype=tf.float64)

    nn_orig = tf.argsort(D_high)
    nn_proj = tf.argsort(D_low)
    ixs_proj = tf.argsort(nn_proj)

    knn_orig = nn_orig[:, 1 : k + 1]
    knn_proj = nn_proj[:, 1 : k + 1]

    V_i = tf.sparse.to_dense(tf.sets.difference(knn_orig, knn_proj), default_value=-1)
    pre_cont = tf.where(V_i != -1, tf.gather(ixs_proj, V_i, batch_dims=-1) - k, 0)
    cont = tf.reduce_sum(pre_cont, -1)
    cont_t = tf.cast(cont, tf.float64)
    k = tf.cast(k, tf.float64)
    n = tf.cast(n, tf.float64)
    return tf.squeeze(1 - (2 / (k * (2 * n - 3 * k - 1)) * cont_t))


def trustworthiness(X, X_2d, k: int) -> tf.Tensor:
    return continuity_impl(X, X_2d, tf.constant(k))
