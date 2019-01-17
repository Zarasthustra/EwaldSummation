import numpy as np
import pytest
import ewald_summation as es


# test if all particles are in box
@pytest.mark.parametrize('x, l_box, l_cell', [
    (np.random.uniform(0, 10, (500, 2)), (20, 20), 1),
    (np.random.uniform(0, 20, (500, 2)), (20, 20), 1),
    (np.random.uniform(0, 10, (500, 3)), (10, 10, 10), 1),
    ])
def test_distance_vectors_non_periodic_neighbour(x, l_box, l_cell):
    simu_config = es.SimuConfig(n_dim=x.shape[1], n_particles=x.shape[0],
    l_box=l_box, l_cell=l_cell, neighbour=True)
    distance_vectors = distance_vectors = es.distances.DistanceVectors(simu_config)
    distance_vectors.cell_linked_neighbour_list(x)
    max_distance = 0
    for i in range(len(x)):
        distance_vectors_arr = distance_vectors(x, i)[:, 1 : simu_config.n_dim + 1]
        distance_vectors_arr = np.linalg.norm(distance_vectors_arr, axis=-1)
        if distance_vectors_arr.max() > max_distance:
            max_distance = distance_vectors_arr.max()
    assert max_distance < np.sqrt(3 * (l_cell * 2) ** 2)