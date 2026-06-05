import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from tensorflow_projection_qm.metrics.class_angular_distortion_index import CADI, cadi


def test_same_angles_have_zero_cadi():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 1, 1, 2])

    npt.assert_allclose(cadi(X, X, y, n_triplets=100, random_seed=7).numpy(), 0.0)


def test_cluster_with_one_point_is_valid_when_another_cluster_has_two_points():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    X_2d = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.2, 0.9],
        ]
    )
    y = np.array([1, 2, 2])

    value = cadi(X, X_2d, y, n_triplets=50, random_seed=11).numpy()

    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_one_non_singleton_cluster_and_other_clusters_singletons():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(5, 4))
    X_2d = rng.normal(size=(5, 2))
    y = np.array([0, 0, 1, 2, 3])

    value = cadi(X, X_2d, y, n_triplets=100, random_seed=17, batch_size=13).numpy()

    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_all_singleton_clusters_are_invalid():
    X = np.eye(3)
    X_2d = X[:, :2]
    y = np.array([0, 1, 2])

    with pytest.raises(tf.errors.InvalidArgumentError):
        cadi(X, X_2d, y, n_triplets=10, random_seed=0).numpy()


def test_random_seed_is_deterministic_for_integer_seed():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(100, 5))
    X_2d = rng.normal(size=(100, 2))
    y = np.repeat(np.arange(4), 25)

    first = cadi(X, X_2d, y, n_triplets=200, random_seed=99, batch_size=17).numpy()
    second = cadi(X, X_2d, y, n_triplets=200, random_seed=99, batch_size=17).numpy()

    npt.assert_allclose(first, second)


def test_random_seed_is_deterministic_for_numpy_generator():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(100, 5))
    X_2d = rng.normal(size=(100, 2))
    y = np.repeat(np.arange(4), 25)

    generator = np.random.default_rng(99)
    first = cadi(X, X_2d, y, n_triplets=200, random_seed=generator, batch_size=17).numpy()
    second = cadi(X, X_2d, y, n_triplets=200, random_seed=generator, batch_size=17).numpy()

    npt.assert_allclose(first, second)


def test_cadi_metric_class_matches_function():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 5))
    X_2d = rng.normal(size=(100, 2))
    y = np.repeat(np.arange(4), 25)

    function_value = cadi(X, X_2d, y, n_triplets=150, random_seed=12, batch_size=19).numpy()
    metric_value = CADI(n_triplets=150, random_seed=12, batch_size=19).measure(X, X_2d, y).numpy()

    npt.assert_allclose(function_value, metric_value)
    