from tensorflow_projection_qm.metrics import continuity

import numpy as np
import numpy.testing as npt


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
    npt.assert_almost_equal(continuity(X, X_2d, k=1).numpy(), 1 - 1.0 / 3.0)


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
    npt.assert_almost_equal(continuity(X, X_2d, k=1).numpy(), 1 - (1 + 1) / 3.0)