import numpy as np
import pytest
import ewald_summation as es


# test if all particles are in box
@pytest.mark.parametrize('x, l_box, l_cell', [
    (np.random.uniform(0, 10, (3000, 2)), (20, 20), 1),
    (np.random.uniform(0, 20, (3000, 2)), (20, 20), 1),
    (np.random.uniform(0, 10, (3000, 3)), (10, 10, 10), 1),
    ])
def test_distance_vectors_non_periodic_neighbour(x, l_box, l_cell):
    n_dim = x.shape[1]
    distance_vectors = es.distances.DistanceVectors(n_dim, l_box, l_cell, PBC=False)
    distance_vectors.cell_linked_neighbour_list(x)
    max_distance = 0
    for i in range(len(x)):
        distance_vectors_arr = distance_vectors(x, i)
        distance_vectors_arr = np.linalg.norm(distance_vectors_arr, axis=-1)
        if distance_vectors_arr.max() > max_distance:
            max_distance = distance_vectors_arr.max()
    assert max_distance < np.sqrt(3 * (l_cell * 2) ** 2)
