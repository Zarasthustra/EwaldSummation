import numpy as np
import math
from numba import njit
from .pairwise_template import pairwise

@pairwise
def LJ(config, switch_start=2.5, cutoff=3.5):
    # precalc
    switch_width = cutoff - switch_start
    n_dim = config.n_dim
    particle_types = config.particle_types
    n_types = len(particle_types)
    sigma = np.empty(n_types)
    epsilon = np.empty(n_types)
    sigma_matrix = np.empty((n_types, n_types))
    epsilon_matrix = np.empty((n_types, n_types))
    for i in range(n_types):
        sigma[i] = particle_types[i][3]
        epsilon[i] = particle_types[i][4]
    for i in range(n_types):
        for j in range(n_types):
            sigma_matrix[i, j] = (sigma[i] + sigma[j]) / 2
            epsilon_matrix[i, j] = math.sqrt(sigma[i] * sigma[j])

    @njit
    def pot_func(pair, dv):
        i, j = pair[0], pair[1]
        epsilon, sigma = epsilon_matrix[i, j], sigma_matrix[i, j]
        if(epsilon != 0.):
            distance = dv[1]
            distance_squared = distance ** 2
            if(distance <= switch_start):
                return (4. * epsilon * sigma**6 / distance_squared**3
                        * (sigma**6 / distance_squared**3 - 1)
                        )
            # calculate potential in switch region
            if (distance > switch_start) and (distance <= cutoff):
                return (4. * epsilon * sigma**6 / distance_squared**3 * (sigma**6 / distance_squared**3 - 1)
                    * (2 * ((distance - cutoff) / switch_width)**3
                    + 3 * ((distance - cutoff) / switch_width)**2)
                    )
        return 0.

    @njit
    def force_func(pair, dv):
        # Def: dv_ij = q_i - q_j
        i, j = pair[0], pair[1]
        epsilon, sigma = epsilon_matrix[i, j], sigma_matrix[i, j]
        if(epsilon != 0.):
            distance_vector = dv[0]
            distance = dv[1]
            distance_squared = distance ** 2
            if(distance <= switch_start):
                output = (24 * epsilon * sigma**6 / distance_squared**4
                        * (2 * sigma**6 / distance_squared**3 - 1))
                return output * distance_vector
            # calculate potential in switch region, (gradient * switch -potential * dswitch)
            if (distance > switch_start) and (distance <= cutoff):
                t = (distance - cutoff) / switch_width
                gradient = 24 * epsilon * sigma**6 / distance**8 * (2 * sigma**6 / distance**6 - 1)
                potential = 4. * epsilon * sigma**6 / distance_squared**3 * (sigma**6 / distance_squared**3 - 1)
                switch = 2 * t**3 + 3 * t**2
                dswitch = 6 / (cutoff - switch_start) / distance * (t**2 + t)
                output = gradient * switch - potential * dswitch
                return output * distance_vector
        return np.zeros(n_dim)

    return pot_func, force_func, cutoff
