import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt
import math
from numba import njit


# run Python3 -m cProfile -0 namefile.prof speed_100_NaCl.property
# then skakeviz namefile.prof

class SodiumChlorideInit:
    def __init__(self, inter_partilce_distance, n_particles_along_axis):
        self.n_particles = n_particles_along_axis**3
        self.x_0 = self.particle_init_regular_grid_2_kinds_3d(inter_partilce_distance, n_particles_along_axis)
        self.v_0 = np.zeros(self.x_0.shape)
        self.charges = self.x_0.sum(axis=1) % 2 * 2 - 1
        self.l_box = [n_particles_along_axis] * 3
        self.sigma_lj = [4.4, 2.35] * int(self.n_particles / 2)
        self.epsilon_lj = [0.1, 0.13] * int(self.n_particles / 2)
        self.masses = np.array([35.453, 22.99] * int(self.n_particles / 2))

    def particle_init_regular_grid_2_kinds_3d(self, inter_partilce_distance, n_particles_along_axis):
        x = np.zeros((n_particles_along_axis ** 3, 3))
        iter = 0
        for i in range(n_particles_along_axis):
            for j in range(n_particles_along_axis):
                for k in range(n_particles_along_axis):
                    x[iter, :] = [i, j, k]
                    iter += 1
        return x

    def __call__(self, _1, _2):
        return self.masses, self.charges, self.x_0, self.v_0 * self.masses[:, None]


sodium_cloride_initializer = SodiumChlorideInit(1, 10)
simu_config = es.SimuConfig(n_dim = 3,
                            l_box = sodium_cloride_initializer.l_box,
                            sigma_lj = sodium_cloride_initializer.sigma_lj,
                            epsilon_lj = sodium_cloride_initializer.epsilon_lj,
                            n_particles = sodium_cloride_initializer.n_particles,
                            n_steps = 50,
                            timestep = 0.001,
                            temp = 300,
                            lj_flag = True,
                            coulomb_flag = True,
                            PBC = True,
                            cutoff = 6,
                            )
test_md = es.MD(es.PhysWorld(), simu_config, sodium_cloride_initializer, es.step_runners.Langevin(damping=0.1))
test_md.run_all()
