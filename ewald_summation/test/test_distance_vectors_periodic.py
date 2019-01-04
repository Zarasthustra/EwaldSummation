import numpy as np
import pytest
import ewald_summation as es


# test if all particles are in box
@pytest.mark.parametrize('x, l_box', [
    (np.array([[0, 0, 0], [1, 1, 1]]), [3, 3, 3]),
    (np.array([[0, 0], [1, 1]]), [3, 3]),
    (np.random.uniform(-10, 20, (100, 3)), (-10, 30)),
    (np.random.uniform(-10, 20, (100, 3)), (-10, 30)),
    (np.random.uniform(-10, 20, (100, 2)), (-10, 30)),
    (np.random.uniform(-20, 10, (100, 2)), (-30, 10)),
    ])
def test_distance_vectors_periodic(x, l_box):
    n_dim = x.shape[1]
    distance_vectors_periodic = es.distances.DistanceVectors(n_dim, l_box, PBC=True)
    distance_vectors_non_periodic = es.distances.DistanceVectors(n_dim, l_box, PBC=False)
    np.testing.assert_almost_equal(distance_vectors_periodic(x), distance_vectors_non_periodic(x))


# test if 1 particle is outside teh box
@pytest.mark.parametrize('x1, x2, l_box', [
    (np.array([[0, 0, 0], [1, 1, 4]]), np.array([[0, 0, 0], [1, 1, 1]]), [3, 3, 3]),
    (np.array([[0, 0, 0], [1, 3.5, 1]]), np.array([[0, 0, 0], [1, 0.5, 1]]), [3, 3, 3]),
    (np.array([[0, 0, 0], [5, 1, 1]]), np.array([[0, 0, 0], [2, 1, 1]]), [3, 3, 3]),
    (np.array([[0, 0], [1, 4]]), np.array([[0, 0], [1, 1]]), [3, 3]),
    (np.array([[0, 0], [4, 1]]), np.array([[0, 0], [1, 1]]), [3, 3]),
    ])
def test_distance_vectors_periodic(x1, x2, l_box):
    n_dim = x1.shape[1]
    distance_vectors_periodic = es.distances.DistanceVectors(n_dim, l_box, PBC=True)
    distance_vectors_non_periodic = es.distances.DistanceVectors(n_dim, l_box, PBC=False)
    np.testing.assert_almost_equal(distance_vectors_periodic(x1), distance_vectors_non_periodic(x2))
