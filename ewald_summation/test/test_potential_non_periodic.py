import numpy as np
import pytest
import ewald_summation as es


@pytest.mark.parametrize('x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj', [
    (np.array([[0,0,0],[1,1,1]]), 1, 1, 2.5, 3.5),
    (np.array([[0,0,0],[1,1,1],[0,1,0]]), 1, 1, 2.5, 3.5),
    (np.array([[0,0,0],[0,1,200]]), 1, 1, 2.5, 3.5),
    (np.array([[0,0,0],[0,1,3.1]]), 1, 1, 2.5, 3.5),
    (np.random.uniform(-20, 20, (100, 3)), 1, 1, 2.5, 3.5),
    ])
def test_potential(x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj):
    n_dim = x.shape[0]
    switch_width_lj = cutoff_lj - switch_start_lj
    distance_vectors = es.distances.DistanceVectors(n_dim)
    lennardjones = es.potentials.LennardJones(epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj)
    potential1 = 0.5 * np.sum(lennardjones.potential(distance_vectors(x)))
    potential2 = es.potentials.lj_potential_total(x)
    np.testing.assert_almost_equal(potential1, potential2)
