from tensorflow_projection_qm.util import distance

import numpy as np
import numpy.testing as npt
from sklearn.metrics import pairwise_distances


def test_singleton():
    A = np.array([[1.0]])

    D = distance.psqdist(A).numpy()
    npt.assert_array_equal(D, 0.0)


def test_1d_data():
    A = np.array([[3.0], [1.0], [2.0]])

    my_res = distance.psqdist(A).numpy()

    npt.assert_(my_res.shape == (3, 3))
    npt.assert_array_equal(
        my_res,
        [
            [0.0, 4.0, 1.0],
            [4.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
    )


def test_big_matrix():
    A = np.arange(1000 * 250).reshape(1000, 250)
    my_res = distance.psqdist(A).numpy()
    sklearn_res = pairwise_distances(A, metric="sqeuclidean")

    npt.assert_(my_res.shape == sklearn_res.shape == (1000, 1000))
    npt.assert_array_equal(my_res, sklearn_res)


def test_big_random_matrix():
    A = np.random.randn(1000, 1000)
    my_res = distance.psqdist(A).numpy()
    sklearn_res = pairwise_distances(A, metric="sqeuclidean")
    npt.assert_(my_res.shape == sklearn_res.shape == (1000, 1000))
    npt.assert_array_almost_equal(my_res, sklearn_res)

    # Assert *strictly* equal fails, but the KNN should be the same
    my_nns = np.argsort(my_res)
    sklearn_nns = np.argsort(sklearn_res)

    npt.assert_array_equal(my_nns, sklearn_nns)
