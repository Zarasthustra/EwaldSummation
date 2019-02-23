import numpy as np
from numba import jit, njit
import math
from .lennard_jones import lj_potential_pairwise
from .coulomb_new import coulomb_potential_pairwise


@njit
def calc_potential(x, general_params, lj_params):
    n_dim = general_params[0]
    n_particles = general_params[1]
    l_box = general_params[2]
    l_box_half = general_params[3]
    lj_flag = general_params[4]
    switch_start = lj_params[0]
    cutoff = lj_params[1]
    switch_width = cutoff - switch_start
    sigma = lj_params[2]
    epsilon = lj_params[3]

    potential = 0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # calc dist_square, dist
            distance_squared = 0
            for k in range(n_dim):
                distance_squared += (x[i, k] - x[j, k])**2
            distance = math.sqrt(distance_squared)
            # calc lj pot
            if lj_flag:
                sigma_mixed = 0.5 * (sigma[i] + sigma[j])
                epsilon_mixed = math.sqrt(epsilon[i] * epsilon[j])
                potential += lj_potential_pairwise(distance,
                                                   distance_squared,
                                                   sigma_mixed,
                                                   epsilon_mixed,
                                                   switch_width,
                                                   switch_start,
                                                   cutoff,
                                                   )
    return potential



def calc_potential_pbc(x, general_params, lj_params, coulomb_params):
    n_dim = general_params[0]
    n_particles = general_params[1]
    l_box = general_params[2]
    l_box_half = general_params[3]
    lj_flag = general_params[4]
    coulomb_flag = general_params[5]
    switch_start = lj_params[0]
    cutoff = lj_params[1]
    switch_width = cutoff - switch_start
    sigma = lj_params[2]
    epsilon = lj_params[3]
    charges = coulomb_params[0]
    alpha = coulomb_params[1]
    v_rec_prefactor = 1 / math.pi / (l_box[0] * l_box[1] * l_box[2])

    potential = 0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # calc dist_square, dist
            distance_squared = 0
            for k in range(n_dim):
                distance_temp = (x[i, k] - x[j, k]) % l_box[k]
                if distance_temp > l_box_half[k]:
                    distance_temp -= l_box[k]
                distance_squared += distance_temp**2
            distance = math.sqrt(distance_squared)
            # calc lj pot
            if lj_flag:
                sigma_mixed = 0.5 * (sigma[i] + sigma[j])
                epsilon_mixed = math.sqrt(epsilon[i] * epsilon[j])
                potential += lj_potential_pairwise(distance,
                                                   distance_squared,
                                                   sigma_mixed,
                                                   epsilon_mixed,
                                                   switch_width,
                                                   switch_start,
                                                   cutoff,
                                                   )
            if coulomb_flag:
                charge_i = charges[i]
                charge_j = charges[j]
                potential += coulomb_potential_pairwise(distance,
                                                        charge_i,
                                                        charge_j,
                                                        alpha,
                                                        )
    return potential
