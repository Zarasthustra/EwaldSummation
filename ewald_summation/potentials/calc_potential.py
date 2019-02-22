import numpy as np
from numba import jit, njit
import math
from .lennard_jones import lj_potential_pairwise

# @njit
def calc_potential(x, general_params, lj_params=False):
    configuration = x
    n_dim = general_params[0]
    n_particles = general_params[1]
    pbc = general_params[2]
    if pbc:
        l_box = general_params[3]
        l_box_half = general_params[4]
    if lj_params:
        switch_start = lj_params[0]
        cutoff = lj_params[1]
        switch_width = cutoff - switch_start
        sigma = lj_params[2]
        epsilon = lj_params[3]
        lj_potential = 0

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # calc dist_square, dist
            distance_squared = 0
            if pbc:
                for k in range(n_dim):
                    distance_temp = (x[i, k] - x[j, k]) % l_box[k]
                    if distance_temp > l_box_half[k]:
                        distance_temp -= l_box[k]
                    distance_squared += distance_temp**2
            else:
                for k in range(n_dim):
                    distance_squared += (x[i, k] - x[j, k])**2
            distance = math.sqrt(distance_squared)
            # calc lj pot
            if lj_params:
                sigma_mixed = 0.5 * (sigma[i] + sigma[j])
                epsilon_mixed = math.sqrt(epsilon[i] * epsilon[j])
                lj_potential += lj_potential_pairwise(distance,
                                                  distance_squared,
                                                  sigma_mixed,
                                                  epsilon_mixed,
                                                  switch_width,
                                                  switch_start,
                                                  cutoff,
                                                  )
    return lj_potential
