import numpy as np
from numba import njit
import math


# calculates potential when only lj is present, called from calc_potential.py
@njit
def calc_potential_lj(x, n_dim, n_particles, PBC, l_box, l_box_half, switch_start, cutoff,
                    switch_width, sigma_lj, epsilon_lj):
        potential_lj = 0
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
                potential_lj += lj_potential_pairwise(distance,
                                                   distance_squared,
                                                   0.5 * (sigma_lj[i] + sigma_lj[j]),
                                                   math.sqrt(epsilon_lj[i] * epsilon_lj[j]),
                                                   switch_width,
                                                   switch_start,
                                                   cutoff,
                                                   )
        return potential_lj


# version of lj potential using multiple cores, does not work for pbc yet
@njit(parallel=True)
def calc_potential_lj_parallel(x, n_dim, n_particles, PBC, l_box, l_box_half, switch_start, cutoff,
                    switch_width, sigma_lj, epsilon_lj):
        # arrays for every parallel loop to temporarily store data
        # needs more testing
        potential_lj = np.zeros(n_particles)
        distance_squared = np.zeros(n_particles)
        distance = np.zeros(n_particles)
        # distance_temp = np.zeros((n_particles, n_dim))
        # prange explicitely tells numba to parallelize that loop
        for i in range(n_particles):
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
                if distance[i] > 0:
                    potential_lj[i] += lj_potential_pairwise(distance[i],
                                                       distance_squared[i],
                                                       0.5 * (sigma_lj[i] + sigma_lj[j]),
                                                       math.sqrt(epsilon_lj[i] * epsilon_lj[j]),
                                                       switch_width,
                                                       switch_start,
                                                       cutoff,
                                                       )
        # print()
        return 0.5 * potential_lj.sum()


# calculate pairwise lj potential
@njit
def lj_potential_pairwise(distance, distance_squared, sigma, epsilon,
                                switch_width, switch_start, cutoff):
    # calculate potential below switch region, assumes all distances passed > 0
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
    # set rest to 0
    else:
        return 0.


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

@njit()
def lj_force_numba(distances, distances_squared, distance_vectors, sigma, epsilon, switch_width, cutoff):
    switch_start = cutoff - switch_width
    output = np.zeros((distances.shape[0], distance_vectors.shape[2]))
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            sigma_mixed = 0.5 * (sigma[i] + sigma[j])
            epsilon_mixed = math.sqrt(epsilon[i] * epsilon[j])
            force_r_part = lj_force_pairwise_numba(distances[i, j], distances_squared[i, j], sigma_mixed,
                                                   switch_width, epsilon_mixed, switch_start, cutoff)
            force = force_r_part * distance_vectors[i, j]
            output[i, :] += force
            output[j, :] -= force
    return output

@njit()
def lj_force_numba_neighbour(distances, distances_squared, distance_vectors, array_index, sigma, epsilon, switch_width, cutoff):
    switch_start = cutoff - switch_width
    output = np.zeros((distances.shape[0], distance_vectors.shape[2]))
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            sigma_mixed = 0.5 * (sigma[i] + sigma[array_index[i, j]])
            epsilon_mixed = math.sqrt(epsilon[i] * epsilon[array_index[i, j]])
            force_r_part = lj_force_pairwise_numba(distances[i, j], distances_squared[i, j], sigma_mixed,
                                                   switch_width, epsilon_mixed, switch_start, cutoff)
            force = force_r_part * distance_vectors[i, j]
            output[i, :] += force
    return output
