import numpy as np
import pytest
import ewald_summation as es


@pytest.mark.parametrize('x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj, l_box, l_cell', [
    (np.random.uniform(0, 6.9, (150, 2)), 1, 1, 2.5, 3.5, [7, 7], 3.5),
    (np.random.uniform(0, 6.9, (150, 3)), 1, 1, 2.5, 3.5, [7, 7, 7], 3.5),
    ])
def test_potential_neighbour(x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj, l_box, l_cell):
    n_dim = x.shape[1]
    epsilon = [epsilon_lj] * x.shape[0]
    sigma = [sigma_lj] * x.shape[0]
    distance_vectors = es.distances.DistanceVectors(n_dim, l_box=l_box, l_cell=l_cell, sigma=sigma, epsilon=epsilon)
    distance_vectors.cell_linked_neighbour_list(x)
    lennard_jones = es.potentials.LennardJones(n_dim, epsilon, sigma, 2.5, 3.5)
    dist = distance_vectors(x, 0)
    potential1 = lennard_jones.potential_neighbour(x, distance_vectors)
    distance_vectors.neighbour_flag = False
    potential2 = lennard_jones.potential(distance_vectors(x))
    np.testing.assert_allclose(potential1, potential2)
