import numpy as np
from numba import jit, njit, cuda, prange
import math
from .lennard_jones import lj_potential_pairwise
from .lennard_jones import lj_force_pairwise
from .coulomb_new import coulomb_potential_pairwise


class CalcForce:
    def __init__(self, config, global_potentials):
        self.n_dim = config.n_dim
        self.n_particles = config.n_particles
        self.l_box = config.l_box
        self.l_box_half = np.array(self.l_box) / 2
        self.PBC = config.PBC
        self.lj_flag = config.lj_flag
        self.sigma_lj = config.sigma_lj
        self.epsilon_lj = config.epsilon_lj
        self.neighbour = config.neighbour
        self.cutoff = config.cutoff
        self.switch_start = config.switch_start
        self.switch_width = self.cutoff - self.switch_start
        self.parallel_flag = config.parallel_flag
        self.global_potentials = global_potentials

    def __call__(self, x):
        force = np.sum([pot.calc_force(x) for pot in self.global_potentials], axis=0)
        if not self.parallel_flag:
            return force + _calc_force(x, self.n_dim, self.n_particles, self.PBC, self.l_box, self.l_box_half, self.lj_flag,
                               self.switch_start, self.cutoff, self.switch_width,
                               self.sigma_lj, self.epsilon_lj)
        if self.parallel_flag:
            return force + _calc_force_parallel(x, self.n_dim, self.n_particles, self.PBC, self.l_box, self.l_box_half, self.lj_flag,
                               self.switch_start, self.cutoff, self.switch_width,
                               self.sigma_lj, self.epsilon_lj)



@njit
def _calc_force(x, n_dim, n_particles, PBC, l_box, l_box_half, lj_flag, switch_start, cutoff,
                    switch_width, sigma_lj, epsilon_lj):
        force = np.zeros(x.shape)
        distance_vector = np.zeros(n_dim)
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                # calc dist_square, dist
                # minimum image convention for pbc
                if PBC:
                    distance_squared = 0
                    for k in range(n_dim):
                        distance_vector[k] = (x[i, k] - x[j, k]) % l_box[k]
                        if distance_vector[k] > l_box_half[k]:
                            distance_vector[k] -= l_box[k]
                        distance_squared += distance_vector[k]**2
                else:
                    distance_vector = x[i, :] - x[j, :]
                    distance_squared = np.sum((distance_vector)**2)
                distance = np.sqrt(distance_squared)
                # calc lj force
                if lj_flag:
                    distance_vector *= lj_force_pairwise(distance,
                                                       distance_squared,
                                                       0.5 * (sigma_lj[i] + sigma_lj[j]),
                                                       math.sqrt(epsilon_lj[i] * epsilon_lj[j]),
                                                       switch_width,
                                                       switch_start,
                                                       cutoff,
                                                       )
                    force[i, :] += distance_vector
                    force[j, :] -= distance_vector
        return force
