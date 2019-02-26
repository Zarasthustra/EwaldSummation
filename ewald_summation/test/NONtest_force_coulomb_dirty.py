import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt
import math
from numba import njit


# Really dirty script to show that large enough systems of charged, regular particles
# give rise to the same forces with the old and new implementation. 


def particle_init_regular_grid_2_kinds_3d(inter_partilce_distance, particles_along_axis):
    x = np.zeros((particles_along_axis ** n_dim, n_dim))
    iter = 0
    for i in range(particles_along_axis):
        for j in range(particles_along_axis):
            for k in range(particles_along_axis):
                x[iter, :] = [i, j, k]
                iter += 1
    return x

n_particles_along_axis = 8
n_particles = n_particles_along_axis**3
n_dim = 3
resolution = 6
l_box = np.array([n_particles_along_axis] * 3)
# grid=np.array(list(product(range(0,3), repeat=n_dim)))
# x = 1. * grid
x = particle_init_regular_grid_2_kinds_3d(1, n_particles_along_axis)
charge_vector = x.sum(axis=1)%2*2-1
# Madelung = -1.74756459463
# pot_ref = -447.37653622528
# pot_real_ref = -159.04160026039025

simu_config = es.SimuConfig(n_dim=x.shape[1], n_particles=x.shape[0], l_box=l_box, l_cell=l_box[0], neighbour=True)
simu_config.charges = charge_vector
distance_vectors = es.distances.DistanceVectors(simu_config)
coulomb = es.potentials.Coulomb(simu_config)
force_calc = coulomb.calc_force(x, distance_vectors(x, 0))

print(force_calc)

simu_config = es.SimuConfig(n_dim = x.shape[1],
                            n_particles = x.shape[0],
                            l_box = l_box,
                            l_cell = l_box[0],
                            neighbour = False,
                            lj_flag = False,
                            coulomb_flag = True,
                            cutoff=  8,
                            )
simu_config.charges = charge_vector
calc_force = es.potentials.CalcForce(simu_config, [])
calc_forco = calc_force(x)
print(calc_forco)
print('sum', np.sum(force_calc - calc_forco))
np.testing.assert_allclose(force_calc, calc_forco)



# print(charge_vector)
# print(x)
# x = particle_init_regular_grid_2_kinds_3d(1, 2)
# charge_vector = x.sum(axis=1)%2*2-1
# print(charge_vector)


# l_box_half = tuple(np.divide(np.array(l_box), 2))
# general_params = (x.shape[1], x.shape[0], tuple(l_box), l_box_half, False, True)
# lj_params = (2.5, 3.5, tuple([1] * x.shape[0]), tuple([1] * x.shape[0]))
# coulomb_params = (charge_vector, 1)

# print(es.potentials.calc_potential_pbc(x, general_params, lj_params, coulomb_params))
