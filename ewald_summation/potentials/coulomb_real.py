import numpy as np
import math
from numba import njit
from .pairwise_template import pairwise

@pairwise
def CoulombReal(config, alpha, cutoff):
    # precalc
    n_dim = config.n_dim
    assert n_dim == 3, "For other dimensions not implemented."
    assert config.PBC, "Ewald sum only meaningful for periodic systems."
    prefactor1 = config.phys_world.k_C / 2.
    prefactor2 = config.phys_world.k_C
    n_alpha_sq = -alpha * alpha
    alpha_coeff = 2 * alpha / math.sqrt(np.pi)
    particle_types = config.particle_types
    n_types = len(particle_types)
    charge = np.empty(n_types)
    charge_product = np.empty((n_types, n_types))
    for i in range(n_types):
        charge[i] = particle_types[i][2]
    for i in range(n_types):
        for j in range(n_types):
            charge_product[i, j] = charge[i] * charge[j]
    
    @njit
    def pot_func(pair, dv):
        i, j = pair[0], pair[1]
        distance = dv[1]
        return prefactor1 * charge_product[i, j] * math.erfc(alpha * distance) / distance
    
    @njit
    def force_func(pair, dv):
        i, j = pair[0], pair[1]
        distance_vector, distance = dv[0], dv[1]
        distance_squared = distance * distance
        real_part = math.erfc(alpha * distance) / distance + \
                        alpha_coeff * math.exp(n_alpha_sq * distance_squared)
        return prefactor2 * charge_product[i, j] * (real_part / distance_squared) * distance_vector
    
    return pot_func, force_func, cutoff
