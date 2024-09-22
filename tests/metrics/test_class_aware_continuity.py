import numpy as np
import numpy.testing as npt

from tensorflow_projection_qm.metrics.continuity import class_aware_continuity


def test_single_data_point():
    X = np.array([[1.0, 2.0, 3.0]])  # a single data point in 3-D
    X_2d = np.array([[0.0, -1.0]])  # a single projected data point
    y = np.array([0])
    npt.assert_equal(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1.0)


def test_single_data_point_first_class_isnt_zero():
    X = np.array([[1.0, 2.0, 3.0]])  # a single data point in 3-D
    X_2d = np.array([[0.0, -1.0]])  # a single projected data point
    y = np.array([5])
    npt.assert_equal(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1.0)


def test_two_data_points_same_class():
    X = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    X_2d = np.array([[0.0, 0.0], [10.0, 10.0]])
    y = np.array([0, 0])
    # In this case, one data point is always the other's nearest
    # neighbor, irrespective of the distance. Continuity should
    # be perfect (== 1.0)
    npt.assert_equal(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1.0)

    # This should not care whether their classes differ or not.
    y = np.array([0, 1])
    npt.assert_equal(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1.0)

    y = np.array([3, 3])
    npt.assert_equal(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1.0)

    y = np.array([3, 5])
    npt.assert_equal(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1.0)


def test_one_error_diff_class():
    # data in a line: 0 -- 1 - 2
    X = np.c_[np.zeros(3), np.array([0.0, 1.5, 2.0]), np.zeros(3)]
    # data in a line, with one swap: 0 -- 2 - 1
    X_2d = np.c_[np.zeros(3), np.array([0.0, 2.0, 1.2])]
    y = np.array([1, 0, 0])
    # Nearest neighbor:
    #    Data Space | Proj Space |
    # 0 |     1     |     2      |  Penalty = 0 because across diff classes
    # 1 |     2     |     2      |  Penalty = 0
    # 2 |     1     |     1      |  Penalty = 0
    npt.assert_allclose(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1.0)


def test_one_error_same_class():
    # data in a line: 0 -- 1 - 2
    X = np.c_[np.zeros(3), np.array([0.0, 1.5, 2.0]), np.zeros(3)]
    # data in a line, with one swap: 0 -- 2 - 1
    X_2d = np.c_[np.zeros(3), np.array([0.0, 2.0, 1.2])]
    y = np.array([0, 0, 3])
    # Nearest neighbor:
    #    Data Space | Proj Space |
    # 0 |     1     |     2      |  Penalty = 1
    # 1 |     2     |     2      |  Penalty = 0
    # 2 |     1     |     1      |  Penalty = 0
    npt.assert_allclose(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1 - 1.0 / 3.0)


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

    # No penalty applies because all classes are different
    y = np.array([0, 1, 2])
    npt.assert_allclose(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1.0)

    # One penalty applies
    y = np.array([0, 0, 2])
    npt.assert_allclose(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1 - 1.0 / 3.0)

    # Two penalties apply
    y = np.array([0, 0, 0])
    npt.assert_allclose(class_aware_continuity(X, X_2d, y, k=1).numpy(), 1 - 2.0 / 3.0)


def test_k_larger_than_dataset_size():
    X = np.random.randn(10, 5)
    X_2d = np.random.randn(10, 2)
    y = np.zeros(10)
    # If k > n-1 (here n == 10), then we're out of neighbors to use for the
    # computation. The calculation should adapt accordingly (i.e., k shouldn't
    # have an effect on the result).
    results = [class_aware_continuity(X, X_2d, y, k=_k).numpy() for _k in (10, 20, 1000)]

    npt.assert_allclose(results[0], results[1])
    npt.assert_allclose(results[1], results[2])
