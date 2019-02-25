import numpy as np
from numba import njit
import math


# calculates lj force when only lj is present, called from calc_force.py
@njit
def calc_force_lj(x, n_dim, n_particles, PBC, l_box, l_box_half, lj_flag, switch_start, cutoff,
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


@njit()
def lj_force_pairwise(distance, distance_squared, sigma, epsilon,
                            switch_width, switch_start, cutoff):
    # calculate force below switch region, assumes all distances passed > 0
    if(distance <= switch_start) and (distance > 0):
        output = (24 * epsilon * sigma**6 / distance_squared**4
                  * (2 * sigma**6 / distance_squared**3 - 1))
        return output

    # calculate potential in switch region, (gradient * switch -potential * dswitch)
    elif (distance > switch_start) and (distance <= cutoff):
        t = (distance - cutoff) / switch_width
        gradient = 24 * epsilon * sigma**6 / distance**8 * (2 * sigma**6 / distance**6 - 1)
        potential = 4. * epsilon * sigma**6 / distance_squared**3 * (sigma**6 / distance_squared**3 - 1)
        switch = 2 * t**3 + 3 * t**2
        dswitch = 6 / (cutoff - switch_start) / distance * (t**2 + t)
        output = gradient * switch - potential * dswitch
        return output

    # set rest to 0
    else:
        return 0.
