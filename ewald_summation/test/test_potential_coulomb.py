import numpy as np
import pytest
import ewald_summation as es
from itertools import product

'''
@pytest.mark.parametrize('x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj, l_box, l_cell', [
    (np.random.uniform(0, 6.9, (150, 2)), 1, 1, 2.5, 3.5, [14, 14], 3.5),
    (np.random.uniform(0, 6.9, (150, 3)), 1, 1, 2.5, 3.5, [14, 14, 14], 3.5),
    ])
def test_potential_neighbour(x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj, l_box, l_cell):
    simu_config = es.SimuConfig(n_dim=x.shape[1], n_particles=x.shape[0], l_box=l_box, l_cell=l_cell,
    switch_start_lj=switch_start_lj, cutoff_lj=cutoff_lj, neighbour=True)
    distance_vectors = es.distances.DistanceVectors(simu_config)
    lennard_jones = es.potentials.LennardJones(simu_config)
    potential1 = lennard_jones.calc_potential(distance_vectors(x, 0))
    distance_vectors.neighbour_flag = False
    potential2 = lennard_jones.calc_potential(distance_vectors(x, 0))
    np.testing.assert_allclose(potential1, potential2)
'''

def test_potential_coulomb(x, charge_vector, l_box, pot_ref):
    simu_config = es.SimuConfig(n_dim=x.shape[1], n_particles=x.shape[0], l_box=l_box, l_cell=l_box[0], neighbour=True)
    simu_config.charges = charge_vector
    distance_vectors = es.distances.DistanceVectors(simu_config)
    coulomb = es.potentials.Coulomb(simu_config)
    pot_calc = coulomb.calc_potential(x, distance_vectors(x, 0))
    np.testing.assert_allclose(pot_calc, pot_ref)

if __name__ == "__main__":
    n_particles = 8
    n_dim = 3
    resolution = 6
    l_box=np.array([2., 2., 2.])
    grid=np.array(list(product(range(0,2), repeat=n_dim)))
    q = 1. * grid
    charge_vector = grid.sum(axis=1)%2*2-1
    Madelung = -1.74756459463
    test_potential_coulomb(q, charge_vector, l_box, Madelung * 4)