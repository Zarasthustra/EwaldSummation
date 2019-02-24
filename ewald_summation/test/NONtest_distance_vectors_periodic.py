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
    simu_config = es.SimuConfig(n_dim=x.shape[1], l_box=l_box)
    distance_vectors_non_periodic = es.distances.DistanceVectors(simu_config)
    simu_config.PBC = True
    distance_vectors_periodic = es.distances.DistanceVectors(simu_config)
    np.testing.assert_almost_equal(distance_vectors_periodic(x, 0).distance_vectors,
                                   distance_vectors_non_periodic(x, 1).distance_vectors)


# test if 1 particle is outside teh box
@pytest.mark.parametrize('x1, x2, l_box', [
    (np.array([[0, 0, 0], [1, 1, 4]]), np.array([[0, 0, 0], [1, 1, 1]]), [3, 3, 3]),
    (np.array([[0, 0, 0], [1, 3.5, 1]]), np.array([[0, 0, 0], [1, 0.5, 1]]), [3, 3, 3]),
    # modified, previous test case is not correct ...
    (np.array([[0, 0, 0], [5, 1, 1]]), np.array([[0, 0, 0], [-1, 1, 1]]), [3, 3, 3]),
    (np.array([[0, 0], [1, 4]]), np.array([[0, 0], [1, 1]]), [3, 3]),
    (np.array([[0, 0], [4, 1]]), np.array([[0, 0], [1, 1]]), [3, 3]),
    ])
def test_distance_vectors_periodic(x1, x2, l_box):
    simu_config = es.SimuConfig(n_dim=x1.shape[1], l_box=l_box)
    distance_vectors_non_periodic = es.distances.DistanceVectors(simu_config)
    simu_config.PBC = True
    distance_vectors_periodic = es.distances.DistanceVectors(simu_config)
    np.testing.assert_almost_equal(distance_vectors_periodic(x1, 0).distance_vectors,
                                   distance_vectors_non_periodic(x2, 1).distance_vectors)
