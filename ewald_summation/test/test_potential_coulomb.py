import numpy as np
import pytest
import ewald_summation as es
from itertools import product


def _intializer_NaCl():
    n_particles = 8
    n_dim = 3
    l_box=np.array([2., 2., 2.])
    grid=np.array(list(product(range(0,2), repeat=n_dim)))
    q = 1. * grid
    charge_vector = grid.sum(axis=1)%2*2-1
    return q, charge_vector, l_box


@pytest.mark.parametrize('initializer, pot_ref', [
    (_intializer_NaCl, -1.74756459463 * 4),
    ])
def test_potential_coulomb(initializer, pot_ref):
    q, charge_vector, l_box = initializer()
    simu_config = es.SimuConfig(n_dim=q.shape[1], n_particles=q.shape[0], l_box=l_box, l_cell=l_box[0], neighbour=True)
    simu_config.charges = charge_vector
    distance_vectors = es.distances.DistanceVectors(simu_config)
    coulomb = es.potentials.Coulomb(simu_config)
    pot_calc = coulomb.calc_potential(q, distance_vectors(q, 0))
    np.testing.assert_allclose(pot_calc, pot_ref)
