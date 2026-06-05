import hashlib
from typing import Optional

import numpy as np
import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import Metric


_DEFAULT_BATCH_SIZE = 4_096


def _int_to_seed(seed: int) -> tf.Tensor:
    """Convert an integer seed to TensorFlow's required two-int stateless seed."""
    seed = int(seed)
    return tf.constant([seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF], dtype=tf.int32)


def _bytes_to_seed(data: bytes) -> tf.Tensor:
    digest = hashlib.blake2b(data, digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="little", signed=False)
    return _int_to_seed(seed)


def _seed_to_tensor(random_seed) -> tf.Tensor:
    """Normalize supported seed inputs to a TensorFlow stateless random seed."""
    if random_seed is None:
        return tf.random.uniform([2], minval=0, maxval=2**31 - 1, dtype=tf.int32)

    if isinstance(random_seed, tf.random.Generator):
        state = tf.cast(random_seed.state, tf.int64)
        state = tf.pad(state[:2], [[0, tf.maximum(0, 2 - tf.shape(state[:2])[0])]])
        return tf.cast(tf.math.floormod(state[:2], 2**31 - 1), tf.int32)

    if isinstance(random_seed, np.random.Generator):
        return _bytes_to_seed(repr(random_seed.bit_generator.state).encode("utf-8"))

    return _int_to_seed(int(random_seed))


@tf.function
def _cluster_info(y):
    """Return class-grouped point indices and cluster metadata."""
    y = tf.reshape(y, [-1])
    flat = tf.cast(tf.argsort(y, stable=True), tf.int32)
    y_sorted = tf.gather(y, flat)

    _, _, cluster_sizes = tf.unique_with_counts(y_sorted)
    cluster_sizes = tf.cast(cluster_sizes, tf.int32)
    offsets = tf.concat([[0], tf.cumsum(cluster_sizes)[:-1]], axis=0)
    valid_cluster_b_idxs = tf.cast(tf.reshape(tf.where(cluster_sizes >= 2), [-1]), tf.int32)

    return flat, offsets, cluster_sizes, valid_cluster_b_idxs


@tf.function
def _cosine_for_triplets(X, x_idxs, y_idxs, z_idxs):
    x = tf.gather(X, x_idxs)
    y = tf.gather(X, y_idxs)
    z = tf.gather(X, z_idxs)

    v1 = y - x
    v2 = z - x
    dot = tf.reduce_sum(v1 * v2, axis=1)
    norm1 = tf.reduce_sum(v1 * v1, axis=1)
    norm2 = tf.reduce_sum(v2 * v2, axis=1)
    denom = tf.sqrt(norm1 * norm2)

    return tf.math.divide_no_nan(dot, denom)


@tf.function
def _sample_uniform_int(unit_random, maxval):
    """Sample integers in [0, maxval), supporting a vector of per-entry max values."""
    maxval = tf.cast(maxval, unit_random.dtype)
    return tf.cast(tf.floor(unit_random * maxval), tf.int32)


@tf.function
def _cadi_batch(X, X_2d, flat, offsets, cluster_sizes, valid_cluster_b_idxs, batch_size, seed):
    n_clusters = tf.shape(cluster_sizes)[0]
    n_valid_b = tf.shape(valid_cluster_b_idxs)[0]

    random_values = tf.random.stateless_uniform([5, batch_size], seed=seed, dtype=tf.float64)

    valid_b_offsets = _sample_uniform_int(random_values[0], n_valid_b)
    cluster_b = tf.gather(valid_cluster_b_idxs, valid_b_offsets)

    raw_cluster_a = _sample_uniform_int(random_values[1], n_clusters - 1)
    cluster_a = raw_cluster_a + tf.cast(raw_cluster_a >= cluster_b, tf.int32)

    size_a = tf.gather(cluster_sizes, cluster_a)
    size_b = tf.gather(cluster_sizes, cluster_b)
    offset_a = tf.gather(offsets, cluster_a)
    offset_b = tf.gather(offsets, cluster_b)

    x_offsets = _sample_uniform_int(random_values[2], size_a)
    y_offsets = _sample_uniform_int(random_values[3], size_b)
    raw_z_offsets = _sample_uniform_int(random_values[4], size_b - 1)
    z_offsets = raw_z_offsets + tf.cast(raw_z_offsets >= y_offsets, tf.int32)

    x_idxs = tf.gather(flat, offset_a + x_offsets)
    y_idxs = tf.gather(flat, offset_b + y_offsets)
    z_idxs = tf.gather(flat, offset_b + z_offsets)

    cos_x = _cosine_for_triplets(X, x_idxs, y_idxs, z_idxs)
    cos_y = _cosine_for_triplets(X_2d, x_idxs, y_idxs, z_idxs)

    return tf.reduce_sum(tf.math.squared_difference(cos_x, cos_y))


@tf.function
def cadi_impl(X, X_2d, y, n_triplets, random_seed, batch_size) -> tf.Tensor:
    X = tf.cast(X, tf.float64)
    X_2d = tf.cast(X_2d, tf.float64)
    y = tf.reshape(y, [-1])
    n = tf.shape(X)[0]

    flat, offsets, cluster_sizes, valid_cluster_b_idxs = _cluster_info(y)
    n_clusters = tf.shape(cluster_sizes)[0]
    n_valid_b = tf.shape(valid_cluster_b_idxs)[0]

    n_triplets = tf.cast(n_triplets, tf.int32)
    n_triplets = tf.where(n_triplets == 0, n * 10, n_triplets)
    batch_size = tf.cast(batch_size, tf.int32)
    random_seed = tf.cast(random_seed, tf.int32)

    assertions = [
        tf.debugging.assert_equal(
            tf.shape(X_2d)[0], n, message="X and X_2d must have the same number of points."
        ),
        tf.debugging.assert_equal(
            tf.shape(y)[0], n, message="X and y must have the same number of points."
        ),
        tf.debugging.assert_greater_equal(
            n, 3, message="Dataset must be composed of at least 3 points."
        ),
        tf.debugging.assert_greater_equal(
            n_clusters, 2, message="Need at least 2 clusters to sample angles between clusters."
        ),
        tf.debugging.assert_positive(
            n_valid_b,
            message="At least one cluster must have length >= 2 to sample y,z in triplet.",
        ),
        tf.debugging.assert_positive(n_triplets, message="n_triplets must be positive or 0."),
        tf.debugging.assert_positive(batch_size, message="batch_size must be positive."),
    ]

    with tf.control_dependencies(assertions):
        n_triplets = tf.identity(n_triplets)
        batch_size = tf.minimum(tf.identity(batch_size), n_triplets)
        random_seed = tf.identity(random_seed)

    def cond(i, sum_sqr):
        del sum_sqr  # tf.while_loop passes every loop variable to the condition.
        return i < n_triplets

    def body(i, sum_sqr):
        current_batch_size = tf.minimum(batch_size, n_triplets - i)
        seed = tf.random.experimental.stateless_fold_in(random_seed, i)
        batch_sum_sqr = _cadi_batch(
            X,
            X_2d,
            flat,
            offsets,
            cluster_sizes,
            valid_cluster_b_idxs,
            current_batch_size,
            seed,
        )
        return i + current_batch_size, sum_sqr + batch_sum_sqr

    _, sum_sqr = tf.while_loop(
        cond,
        body,
        (tf.constant(0, dtype=tf.int32), tf.constant(0.0, dtype=tf.float64)),
    )

    return sum_sqr / (4.0 * tf.cast(n_triplets, tf.float64))


def cadi(X, X_2d, y, n_triplets: int = 0, random_seed=None, batch_size: int = _DEFAULT_BATCH_SIZE):
    return cadi_impl(X, X_2d, y, tf.constant(n_triplets), _seed_to_tensor(random_seed), batch_size)


class CADI(Metric):
    """Class Angular Distortion Index.

    CADI is a cluster-level metric that measure the mean square deviation of inter-class angles
    between a high-dimensional dataset and its projection.

    Parameters
    ----------
    n_triplets : int, default=0
        Number of triplets to sample. If set to 0, the metric samples
        ``10 * n_samples`` triplets.
    random_seed : int, numpy.random.Generator, tensorflow.random.Generator or None, default=None
        Seed or random generator used to sample triplets. Passing the same seed
        produces deterministic samples. If None, a fresh TensorFlow seed is
        generated for each metric evaluation.
    batch_size : int, default=4096
        Maximum number of sampled triplets processed at once. Smaller values
        reduce peak memory use; larger values may improve throughput when enough
        memory is available.

    Notes
    -----
    CADI requires at least two classes, and at least one
    class with two or more points.
    """

    name = "cadi"
    _fn = cadi_impl

    def __init__(
        self,
        n_triplets: Optional[int] = 0,
        random_seed=None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.n_triplets = n_triplets
        self.batch_size = batch_size
        self._random_seed = random_seed

    @property
    def config(self):
        return {
            "n_triplets": self.n_triplets,
            "random_seed": self._random_seed,
            "batch_size": self.batch_size,
        }

    def measure(self, X, X_2d, y):
        return self._measure_impl(
            X, X_2d, y, self.n_triplets, _seed_to_tensor(self._random_seed), self.batch_size
        )

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"], args["y"])
