import numpy as np
import numpy.testing as npt

from tensorflow_projection_qm.metrics.continuity import continuity


def test_single_data_point():
    X = np.array([[1.0, 2.0, 3.0]])  # a single data point in 3-D
    X_2d = np.array([[0.0, -1.0]])  # a single projected data point

    npt.assert_equal(continuity(X, X_2d, k=1).numpy(), 1.0)


def test_two_data_points():
    X = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    X_2d = np.array([[0.0, 0.0], [10.0, 10.0]])

    # In this case, one data point is always the other's nearest
    # neighbor, irrespective of the distance. Continuity should
    # be perfect (== 1.0)
    npt.assert_equal(continuity(X, X_2d, k=1).numpy(), 1.0)


def test_one_error():
    # data in a line: 0 -- 1 - 2
    X = np.c_[np.zeros(3), np.array([0.0, 1.5, 2.0]), np.zeros(3)]
    # data in a line, with one swap: 0 -- 2 - 1
    X_2d = np.c_[np.zeros(3), np.array([0.0, 2.0, 1.2])]

    # Nearest neighbor:
    #    Data Space | Proj Space |
    # 0 |     1     |     2      |  Penalty = 1
    # 1 |     2     |     2      |  Penalty = 0
    # 2 |     1     |     1      |  Penalty = 0
    npt.assert_allclose(continuity(X, X_2d, k=1).numpy(), 1 - 1.0 / 3.0)


def test_two_errors():
    # data in a line: 0 -- 1 - 2
    X = np.c_[np.zeros(3), np.array([0.0, 1.5, 2.0]), np.zeros(3)]
    # data in a line, with one swap: 0 - 2 -- 1
    X_2d = np.c_[np.zeros(3), np.array([0.0, 2.0, 0.9])]

    # Nearest neighbor:
    #    Data Space | Proj Space |
    # 0 |     1     |     2      |  Penalty = 1
    # 1 |     2     |     2      |  Penalty = 0
    # 2 |     1     |     0      |  Penalty = 1
    npt.assert_allclose(continuity(X, X_2d, k=1).numpy(), 1 - (1 + 1) / 3.0)


def test_k_larger_than_dataset_size():
    X = np.random.randn(10, 5)
    X_2d = np.random.randn(10, 2)

    # If k > n-1 (here n == 10), then we're out of neighbors to use for the
    # computation. The calculation should adapt accordingly (i.e., k shouldn't
    # have an effect on the result).
    results = [continuity(X, X_2d, k=_k).numpy() for _k in (10, 20, 1000)]

    npt.assert_allclose(results[0], results[1])
    npt.assert_allclose(results[1], results[2])


def test_no_errors_possible():
    # data in a line: 0 -- 1 - 2
    X = np.c_[np.zeros(3), np.array([0.0, 1.5, 2.0]), np.zeros(3)]
    # data in a line, with one swap: 0 - 2 -- 1
    X_2d = np.c_[np.zeros(3), np.array([0.0, 2.0, 0.9])]

    # Since K = n-1, there are no possible errors, continuity is perfect.

    # Nearest neighbors:
    #    Data Space | Proj Space |
    # 0 |    1, 2   |     2, 1   |  Penalty = 0
    # 1 |    2, 0   |     2, 0   |  Penalty = 0
    # 2 |    1, 0   |     0, 1   |  Penalty = 0
    npt.assert_allclose(continuity(X, X_2d, k=2).numpy(), 1)


def test_two_missing_neighbors():
    # data in a line: 0 -- 1 - 2 -- 3
    X = np.c_[np.zeros(4), np.array([0.0, 1.5, 2.0, 3.0]), np.zeros(4)]
    # data in a line, with one swap: 0 - 2 -- 1 - 3
    X_2d = np.c_[np.zeros(4), np.array([0.0, 2.0, 0.9, 2.1])]

    # Nearest neighbors:
    #    Data Space | Proj Space |
    # 0 |    1, 2   |     2, 1   |  Penalty = 0
    # 1 |    2, 0   |     3, 2   |  Penalty = 3 - 2 = 1 (0 is missing neighbor)
    # 2 |    1, 3   |     0, 1   |  Penalty = 3 - 2 = 1 (3 is missing neighbor)
    # 3 |    2, 1   |     1, 2   |  Penalty = 0
    npt.assert_allclose(continuity(X, X_2d, k=2).numpy(), 1 - 2.0 / 4.0)


def test_three_missing_neighbors_diff_ranks():
    # data: 0 -- 1 - 2 -- 3
    #            |
    #            4
    # proj: 0 --- 1 - 2 -- 3 ---- 4
    X = np.c_[np.zeros(5), np.array([0.0, 1.5, 2.0, 3.1, 1.5]), np.array([0, 0, 0, 0, 1.0])]
    X_2d = np.c_[np.zeros(5), np.array([0.0, 1.5, 2.0, 3.1, 10.0])]

    # Nearest neighbors:
    #    Data Space | Proj Space | Diff |
    # 0 |    1, 4   |     1, 2   |  [4] | Penalty = 4 - 2 = 2 (4 is missing)
    # 1 |    2, 4   |     2, 0   |  [4] | Penalty = 4 - 2 = 2 (4 is missing)
    # 2 |    1, 3   |     1, 3   |  []  | Penalty = 0
    # 3 |    2, 1   |     2, 1   |  []  | Penalty = 0
    # 4 |    1, 2   |     3, 2   |  [1] | Penalty = 3 - 2 = 1 (1 is missing)
    npt.assert_allclose(
        continuity(X, X_2d, k=2).numpy(),
        np.mean(1 - np.array([2.0, 2.0, 0.0, 0.0, 1.0]) / (2 * 5 - 3 * 2 - 1)),
    )
