import numpy as np
from numba import njit
import math
from .lj_force import calc_force_lj
# from .coulomb_new import coulomb_potential_pairwise


class CalcForce:
    def __init__(self, config, global_potentials):
        self.n_dim = config.n_dim
        self.n_particles = config.n_particles
        self.l_box = config.l_box
        self.l_box_half = np.array(self.l_box) / 2
        self.PBC = config.PBC
        self.lj_flag = config.lj_flag
        self.coulomb_flag = config.coulomb_flag
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
        if self.lj_flag and not self.coulomb_flag:
            if not self.parallel_flag:
                return force + calc_force_lj(x, self.n_dim, self.n_particles, self.PBC, self.l_box, self.l_box_half, self.lj_flag,
                                self.switch_start, self.cutoff, self.switch_width,
                                self.sigma_lj, self.epsilon_lj)
        # if self.parallel_flag:
        #     return force + _calc_force_parallel(x, self.n_dim, self.n_particles, self.PBC, self.l_box, self.l_box_half, self.lj_flag,
        #                        self.switch_start, self.cutoff, self.switch_width,
        #                        self.sigma_lj, self.epsilon_lj)
