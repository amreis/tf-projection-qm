import numpy as np
import numpy.testing as npt
import pytest
from sklearn.metrics import pairwise_distances

from tensorflow_projection_qm.util import distance


class TestPSQDist:
    def test_singleton(self):
        A = np.array([[1.0]])

        D = distance.psqdist(A).numpy()
        npt.assert_array_equal(D, 0.0)

    def test_1d_data(self):
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

    def test_2d_data(self):
        A = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0], [-1.0, -0.5]])

        my_res = distance.psqdist(A).numpy()

        npt.assert_(my_res.shape == (4, 4))
        npt.assert_array_equal(
            my_res,
            [
                [0.0, 2.0, 5.0, 2**2 + 2.5**2],
                [2.0, 0.0, 5.0, 3**2 + 1.5**2],
                [5.0, 5.0, 0.0, 1.25],
                [2**2 + 2.5**2, 3**2 + 1.5**2, 1.25, 0.0],
            ],
        )

    def test_big_matrix(self):
        A = np.arange(1000 * 250).reshape(1000, 250)
        my_res = distance.psqdist(A).numpy()
        sklearn_res = pairwise_distances(A, metric="sqeuclidean")

        npt.assert_(my_res.shape == sklearn_res.shape == (1000, 1000))
        npt.assert_array_equal(my_res, sklearn_res)

    def test_big_random_matrix(self):
        A = np.random.randn(1000, 1000)
        my_res = distance.psqdist(A).numpy()
        sklearn_res = pairwise_distances(A, metric="sqeuclidean")
        npt.assert_(my_res.shape == sklearn_res.shape == (1000, 1000))
        npt.assert_array_almost_equal(my_res, sklearn_res)

        # Assert *strictly* equal fails, but the KNN should be the same
        my_nns = np.argsort(my_res)
        sklearn_nns = np.argsort(sklearn_res)

        npt.assert_array_equal(my_nns, sklearn_nns)

    @pytest.mark.parametrize(
        "X",
        [
            np.array([[1.0]]),
            np.array([[1.0, 2.0, 3.0]]),
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(128, dtype=np.float32).reshape(32, 4),
        ],
    )
    def test_flat_psqdist(self, X) -> None:
        flat_D: np.ndarray = distance.flat_psqdist(X).numpy()
        D: np.ndarray = distance.psqdist(X).numpy()

        npt.assert_equal(flat_D.ndim, 1)
        npt.assert_equal(flat_D.shape[0], (X.shape[0] * (X.shape[0] - 1)) // 2)
        npt.assert_equal(flat_D, D[np.triu_indices_from(D, k=1)])


class TestCSQDist:
    def test_handles_singleton(self):
        out = distance.csqdist(np.array([[1.0]]), np.array([[2.0], [3.0], [4.0]]))

        npt.assert_allclose(out, np.array([[1.0, 4.0, 9.0]]))

    _DATA_MATRICES = [
        np.array([[3.0, -2.0, 1.0]]),
        np.array([[1.0, 2.0, -1.0], [2.0, 0.0, -3.0]]),
        np.array([[1.0, 2.0, -1.0], [2.0, 0.0, -3.0], [-2.5, 1.3, 2.6]]),
    ]

    @pytest.mark.parametrize("X", _DATA_MATRICES)
    @pytest.mark.parametrize("Y", _DATA_MATRICES)
    def test_handles_different_number_of_rows(self, X, Y):
        try:
            npt.assert_(distance.csqdist(X, Y).numpy().shape, (X.shape[0], Y.shape[0]))
        except Exception:
            pytest.fail()

    @pytest.mark.parametrize(
        "X",
        [
            np.array([[1.0]]),
            np.array([[3.0, 1.0, 1.0]]),
            np.array([[1.0, 2.0], [3.0, -4.0]]),
            np.arange(10_000).reshape(100, 100),
        ],
    )
    def test_compatible_with_psqdist(self, X):
        npt.assert_allclose(distance.csqdist(X, X), distance.psqdist(X))

    @pytest.mark.parametrize(
        "X,Y",
        [
            (np.array([[1.0]]), np.array([[2.0]])),
            (np.array([[-2.0, 0.0]]), np.array([[0.0, 0.0]])),
            (np.array([[-2.0, 0.0], [1.0, -1.0]]), np.array([[1.0, 2.0]])),
        ],
    )
    def test_compatible_with_sklearn(self, X, Y):
        npt.assert_allclose(distance.csqdist(X, Y), pairwise_distances(X, Y, metric="sqeuclidean"))
