import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt
import math
from numba import njit

x = np.array([[0, 0], [1.1, 0]])
n_particles = 2

test_config = es.SimuConfig(n_dim=2,
                            l_box=(2., 2., 2),
                            n_particles=2,
                            n_steps=10000,
                            timestep=0.001,
                            temp=300,
                            sigma_lj = [1] * 2,
                            epsilon_lj = [1] * 2,
                            PBC = False,
                            parallel_flag = False,
                            )


test = es.potentials.CalcForce(test_config, [])
force = test(x)
print(force)

def lj_force_numba(distances, distances_squared, distance_vectors, sigma, epsilon, switch_width, cutoff):
    switch_start = cutoff - switch_width
    output = np.zeros((distances.shape[0], distance_vectors.shape[2]))
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            sigma_mixed = 0.5 * (sigma[i] + sigma[j])
            epsilon_mixed = math.sqrt(epsilon[i] * epsilon[j])
            force_r_part = lj_force_pairwise(distances[i, j], distances_squared[i, j], sigma_mixed,
                                                   switch_width, epsilon_mixed, switch_start, cutoff)
            force = force_r_part * distance_vectors[i, j]
            output[i, :] += force
            output[j, :] -= force
    return output

def lj_force_pairwise(distance, distance_squared, sigma, epsilon,
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

distance_vectors = x[:, None, :] - x[None, :, :]
distances_squared = np.sum(distance_vectors**2, axis=-1)
distances = np.sqrt(distances_squared)
print(distances)

force1 = lj_force_numba(distances, distances_squared, distance_vectors, [1] * n_particles, [1] * n_particles, 1, 3.5)
print(force1)
