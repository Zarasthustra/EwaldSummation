import numpy as np
from numba import jit, njit, cuda, prange
import math
from .lennard_jones import lj_potential_pairwise
from .coulomb_new import coulomb_potential_pairwise


class CalcPotential:
    def __init__(self, config):
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

    def __call__(self, x):
        if not self.parallel_flag:
            return _calc_potential(x, self.n_dim, self.n_particles, self.PBC, self.l_box, self.l_box_half, self.lj_flag,
                               self.switch_start, self.cutoff, self.switch_width,
                               self.sigma_lj, self.epsilon_lj)
        if self.parallel_flag:
            return _calc_potential_parallel(x, self.n_dim, self.n_particles, self.PBC, self.l_box, self.l_box_half, self.lj_flag,
                               self.switch_start, self.cutoff, self.switch_width,
                               self.sigma_lj, self.epsilon_lj)



@njit
def _calc_potential(x, n_dim, n_particles, PBC, l_box, l_box_half, lj_flag, switch_start, cutoff,
                    switch_width, sigma_lj, epsilon_lj):
        potential = 0
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                # calc dist_square, dist
                distance_squared = 0
                # minimum image convention for pbc
                if PBC:
                    for k in range(n_dim):
                        distance_temp = (x[i, k] - x[j, k]) % l_box[k]
                        if distance_temp > l_box_half[k]:
                            distance_temp -= l_box[k]
                        distance_squared += distance_temp**2
                else:
                    distance_squared += np.sum((x[i, :] - x[j, :])**2)
                distance = np.sqrt(distance_squared)
                # calc lj pot
                if lj_flag:
                    potential += lj_potential_pairwise(distance,
                                                       distance_squared,
                                                       0.5 * (sigma_lj[i] + sigma_lj[j]),
                                                       math.sqrt(epsilon_lj[i] * epsilon_lj[j]),
                                                       switch_width,
                                                       switch_start,
                                                       cutoff,
                                                       )
        return potential


@njit(parallel=True)
def _calc_potential_parallel(x, n_dim, n_particles, PBC, l_box, l_box_half, lj_flag, switch_start, cutoff,
                    switch_width, sigma_lj, epsilon_lj):
        # arrays for every parallel loop to temporarily store data
        # needs more testing
        potential = np.zeros(n_particles)
        distance_squared = np.zeros(n_particles)
        distance = np.zeros(n_particles)
        # distance_temp = np.zeros((n_particles, n_dim))
        # prange explicitely tells numba to parallelize that loop
        for i in prange(n_particles):
            for j in range(n_particles):
                # calc dist_square, dist
                distance_squared[i] = 0
                # minimum image convention for pbc
                # if PBC:
                #     for k in range(n_dim):
                #         distance_temp = (x[i, k] - x[j, k]) % l_box[k]
                #         if distance_temp > l_box_half[k]:
                #             distance_temp -= l_box[k]
                #         distance_squared[i] += distance_temp**2
                # else:
                distance_squared[i] += np.sum((x[i, :] - x[j, :])**2)
                distance[i] = np.sqrt(distance_squared[i])
                # calc lj pot
                if lj_flag and distance[i] > 0:
                    potential[i] += lj_potential_pairwise(distance[i],
                                                       distance_squared[i],
                                                       0.5 * (sigma_lj[i] + sigma_lj[j]),
                                                       math.sqrt(epsilon_lj[i] * epsilon_lj[j]),
                                                       switch_width,
                                                       switch_start,
                                                       cutoff,
                                                       )
        # print()
        return 0.5 * potential.sum()

#
#
# def calc_potential_pbc(x, general_params, lj_params, coulomb_params):
#     n_dim = general_params[0]
#     n_particles = general_params[1]
#     l_box = general_params[2]
#     l_box_half = general_params[3]
#     lj_flag = general_params[4]
#     coulomb_flag = general_params[5]
#     switch_start = lj_params[0]
#     cutoff = lj_params[1]
#     switch_width = cutoff - switch_start
#     sigma = lj_params[2]
#     epsilon = lj_params[3]
#     charges = coulomb_params[0]
#     alpha = coulomb_params[1]
#     v_rec_prefactor = 1 / math.pi / (l_box[0] * l_box[1] * l_box[2])
#
#     potential = 0
#     for i in range(n_particles):
#         for j in range(i + 1, n_particles):
#             # calc dist_square, dist
#             distance_squared = 0
#             for k in range(n_dim):
#                 distance_temp = (x[i, k] - x[j, k]) % l_box[k]
#                 if distance_temp > l_box_half[k]:
#                     distance_temp -= l_box[k]
#                 distance_squared += distance_temp**2
#             distance = math.sqrt(distance_squared)
#             # calc lj pot
#             if lj_flag:
#                 sigma_mixed = 0.5 * (sigma[i] + sigma[j])
#                 epsilon_mixed = math.sqrt(epsilon[i] * epsilon[j])
#                 potential += lj_potential_pairwise(distance,
#                                                    distance_squared,
#                                                    sigma_mixed,
#                                                    epsilon_mixed,
#                                                    switch_width,
#                                                    switch_start,
#                                                    cutoff,
#                                                    )
#             if coulomb_flag:
#                 charge_i = charges[i]
#                 charge_j = charges[j]
#                 potential += coulomb_potential_pairwise(distance,
#                                                         charge_i,
#                                                         charge_j,
#                                                         alpha,
#                                                         )
#     return potential
