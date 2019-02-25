import numpy as np
import pytest
import ewald_summation as es


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
    charge_vector = x.sum(axis=1) % 2 * 2 - 1
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
    simu_config.charges = charge_vector
    calc_pot = es.potentials.CalcPotential(simu_config, [])
    pot_calc = calc_pot(x)
    np.testing.assert_allclose(pot_calc, Madelung * n_particles / 2)
