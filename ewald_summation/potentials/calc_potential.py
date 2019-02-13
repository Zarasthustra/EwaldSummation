import numpy as np
from numba import njit
import math

def calc_potential(x, general_params, lj_params):
    n_dim = general_params[0]
    n_particles = general_params[1]
    switch_start = lj_params[0]
    cutoff = lj_params[1]
    switch_width = cutoff - switch_start
    sigma = lj_params[2]
    epsilon = lj_params[3]
    lj_potential = 0

    for i in range(n_particles):
        for j in range(n_particles):
            distance_squared = np.sum((x[i, :] - x[j, :])**2)
            distance = np.sqrt(distance_squared)
            if distance > 0:
                lj_potential += lj_potential_pairwise(distance, distance_squared, sigma, epsilon,
                                                      switch_width, switch_start, cutoff)
    return lj_potential
