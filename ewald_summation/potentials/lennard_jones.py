import numpy as np
from numba import njit
import math


class LennardJones:
    def __init__(self, config):
        self.n_dim = config.n_dim
        # calculate array for epsilon where the value of element i,j corresponds to the value
        # for particles i,j of distance_vectors array according to mixing condition
        # epsilon_ij = sqrt(epsilon_i * epsilon_j)
        # self.epsilon_arr = np.sqrt(np.array(config.epsilon_lj)[:, None] * np.array(config.epsilon_lj))
        # calculate array for sigma where the value of element i,j corresponds to the value
        # for particles i,j of distance_vectors array according to mixing condition
        # sigma_ij = (0.5 * (sigma_i + sigma_j))
        # self.sigma_arr = (0.5 * (np.array(config.sigma_lj)[:, None] + np.array(config.sigma_lj)))

        # precomputing mixing conditions disabled as they did not increase preformance
        self.sigma = config.sigma_lj
        self.epsilon = config.epsilon_lj
        self.cutoff = config.cutoff_lj
        self.switch_start = config.switch_start_lj
        self.switch_width = self.cutoff - self.switch_start

    def calc_potential(self, current_frame):
        # calls numba function outside of class namespace
        output = lj_potential_numba(current_frame.distances,
                          current_frame.distances_squared,
                          self.sigma, self.epsilon,
                          self.switch_width,
                          self.cutoff,
                          )
        return output

    def calc_force(self, current_frame):
        # calls numba function outside of class namespace
        output = lj_force_numba(current_frame.distances,
                          current_frame.distances_squared,
                          current_frame.distance_vectors,
                          self.sigma, self.epsilon,
                          self.switch_width,
                          self.cutoff,
                          )
        return output

    def potential_neighbour(self, x, distance_vectors):
        head = distance_vectors.head
        neighbour = distance_vectors.neighbour
        cell_indexes = distance_vectors.cell_indexes
        output = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            head_index = cell_indexes[i]
            temp_arr = distance_vectors(x, i)
            temp_arr = np.delete(temp_arr, np.s_[1 : self.n_dim + 1], 1)
            temp_arr[:, 0] = np.apply_along_axis(self.potential_along_axis, 1, temp_arr)
            output[i] = np.sum(temp_arr[:, 0])
        return output

    def force_neighbour(self, x, distance_vectors):
        head = distance_vectors.head
        neighbour = distance_vectors.neighbour
        cell_indexes = distance_vectors.cell_indexes
        output = np.zeros((x.shape[0], self.n_dim))
        for i in range(x.shape[0]):
            head_index = cell_indexes[i]
            temp_arr = distance_vectors(x, i)
            temp_arr[:, : self.n_dim] = np.apply_along_axis(self.force_along_axis, 1, temp_arr)
            output[i, :] = np.sum(temp_arr[:, : self.n_dim], axis=0)
        return output


# calculate potential using numba, therefor can not be class method
@njit(parallel=True)
def lj_potential_pairwise_numba(distance, distance_squared, sigma, epsilon,
                                switch_width, switch_start, cutoff):
    # calculate potential between 0 and switch region
    if(distance <= switch_start) and (distance > 0):
        output = 4. * epsilon * sigma**6 / distance_squared**3 * (sigma**6 / distance_squared**3 - 1)
        return output

    # calculate potential in switch region
    elif (distance > switch_start) and (distance <= cutoff):
        output = (4. * epsilon * sigma**6 / distance_squared**3 * (sigma**6 / distance_squared**3 - 1)
                 * (2 * ((distance - cutoff) / switch_width)**3
                 + 3 * ((distance - cutoff) / switch_width)**2)
                 )
        return output

    # set rest to 0
    else:
        return 0.

@njit()
def lj_potential_numba(distances, distances_squared, sigma, epsilon, switch_width, cutoff):
    switch_start = cutoff - switch_width
    output = np.zeros(len(distances))
    for i in range(len(distances)):
        for j in range(i, len(distances)):
            sigma_mixed = 0.5 * (sigma[i] + sigma[j])
            epsilon_mixed = math.sqrt(epsilon[i] * epsilon[j])
            pot = lj_potential_pairwise_numba(distances[i, j], distances_squared[i, j], sigma_mixed,
                                              switch_width, epsilon_mixed, switch_start, cutoff)
            output[i] += pot
            output[j] += pot
    return output


# calculate force using numba, therefor can not be class method
@njit(parallel=True)
def lj_force_pairwise_numba(distance, distance_squared, sigma, epsilon,
                            switch_width, switch_start, cutoff):
    # calculate potential between 0 and switch region
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
        for j in range(i, len(distances)):
            sigma_mixed = 0.5 * (sigma[i] + sigma[j])
            epsilon_mixed = math.sqrt(epsilon[i] * epsilon[j])
            force_r_part = lj_force_pairwise_numba(distances[i, j], distances_squared[i, j], sigma_mixed,
                                                   switch_width, epsilon_mixed, switch_start, cutoff)
            force = force_r_part * distance_vectors[i, j]
            output[i, :] += force
            output[j, :] -= force
    return output
