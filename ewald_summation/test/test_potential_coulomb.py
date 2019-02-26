import numpy as np
import pytest
import ewald_summation as es


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
    # simu_config = es.SimuConfig(n_dim=q.shape[1], n_particles=q.shape[0], l_box=l_box, l_cell=l_box[0], neighbour=True)
    simu_config = es.SimuConfig(n_dim=q.shape[1], n_particles=q.shape[0], l_box=l_box, l_cell=l_box[0], PBC=True, neighbour=False)
    simu_config.charges = charge_vector
    distance_vectors = es.distances.DistanceVectors(simu_config)
    coulomb = es.potentials.Coulomb(simu_config)
    pot_calc = coulomb.calc_potential(q, distance_vectors(q, 0))
    np.testing.assert_allclose(pot_calc, pot_ref)

'''
def particle_init_regular_grid_2_kinds_3d(inter_partilce_distance, n_particles_along_axis):
    x = np.zeros((n_particles_along_axis ** 3, 3))
    iter = 0
    for i in range(n_particles_along_axis):
        for j in range(n_particles_along_axis):
            for k in range(n_particles_along_axis):
                x[iter, :] = [i, j, k]
                iter += 1
    return x


@pytest.mark.parametrize('n_particles_along_axis', [
    (8),
    ])
def test_potential_coulomb(n_particles_along_axis):
    n_particles = n_particles_along_axis**3
    resolution = 6
    l_box = np.array([n_particles_along_axis] * 3)
    x = particle_init_regular_grid_2_kinds_3d(1, n_particles_along_axis)
    charges = x.sum(axis=1) % 2 * 2 - 1
    Madelung = -1.74756459463
    simu_config = es.SimuConfig(n_dim = x.shape[1],
                                n_particles = x.shape[0],
                                l_box = l_box,
                                l_cell = l_box[0],
                                neighbour=False,
                                lj_flag = False,
                                coulomb_flag = True,
                                cutoff = 8,
                                )
    simu_config.charges = charges
    calc_pot = es.potentials.CalcPotential(simu_config, [])
    pot_calc = calc_pot(x)
    np.testing.assert_allclose(pot_calc, Madelung * n_particles / 2)
'''
