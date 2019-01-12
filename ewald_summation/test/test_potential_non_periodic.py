import numpy as np
import pytest
import ewald_summation as es


@pytest.mark.parametrize('x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj', [
    (np.array([[0,0,0],[1,1,1]]), 1, 1, 2.5, 3.5),
    (np.array([[0,0,0],[1,1,1],[0,1,0]]), 1, 1, 2.5, 3.5),
    (np.array([[0,0,0],[0,1,200]]), 1, 1, 2.5, 3.5),
    (np.array([[0,0,0],[0,1,3.1]]), 1, 1, 2.5, 3.5),
    (np.random.uniform(-20, 20, (100, 2)), 1, 1, 2.5, 3.5),
    (np.random.uniform(-20, 20, (100, 3)), 1, 1, 2.5, 3.5),
    ])
def test_potential(x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj):
    simu_config = es.SimuConfig(n_dim=x.shape[1], n_particles=x.shape[0],
    switch_start_lj=switch_start_lj, cutoff_lj=cutoff_lj)
    switch_width_lj = cutoff_lj - switch_start_lj
    distance_vectors = es.distances.DistanceVectors(simu_config)
    lennardjones = es.potentials.LennardJones(simu_config)
    potential1 = 0.5 * np.sum(lennardjones.potential(distance_vectors(x)))
    potential2 = es.potentials.lj_potential_total(x)
    np.testing.assert_allclose(potential1, potential2)
