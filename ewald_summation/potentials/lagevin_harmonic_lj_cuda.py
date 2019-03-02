import numpy as np
import math
from numba import cuda, float32, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64


def lagevin_harmonic_lj_cuda(q_0, p_0, masses, particle_types, lj_mixing_conditions, n_particles_tpyes,
                             n_steps, cutoff, damping=0.1, time_step=0.1, k=1, temp=30, k_B = 0.00198720360):
    # general params
    n_dim = q_0.shape[1]
    n_particles = q_0.shape[0]
    n_steps = n_steps
    dtype = float64
    beta = 1. / (k_B * temp)
    shape = (n_particles, n_dim)
    th = 0.5 * time_step
    thm = 0.5 * time_step / masses
    edt = np.exp(-damping * time_step)
    sqf = np.sqrt((1.0 - edt ** 2) / beta)
    switch_width = 1
    switch_start = cutoff - switch_width
    n_particles_tpyes = 0
    n_particles_tpyes_internal = n_particles_tpyes - 1

    # Copy the arrays to the device
    q_device = cuda.device_array((n_steps + 1, n_particles, n_dim))
    q_device[0, :, :] = q_0
    p_device = cuda.device_array((n_steps + 1, n_particles, n_dim))
    p_device[0, :, :] = p_0
    thm_device = cuda.to_device(thm)
    force_device = cuda.device_array(shape)
    particle_types_device = cuda.to_device(particle_types)
    lj_mixing_conditions_device = cuda.to_device(lj_mixing_conditions)

    # Configure the blocks, init random seed
    threadsperblock = 16
    blockspergrid_x = int(math.ceil(q_0.shape[0] / threadsperblock))
    blockspergrid = (blockspergrid_x)
    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid_x * n_dim, seed=1)

    # split lagevin exectuion in two kernels
    @cuda.jit
    def kernel_lagevin_1(q_device, p_device, force_device, thm_device, rng_states):
        i = cuda.grid(1)
        if i < n_particles:
            for k in range(n_dim):
                p_device[1, i, k] = p_device[0, i, k] + th * force_device[i, k]
            for k in range(n_dim):
                q_device[1, i, k] = q_device[0, i, k] + thm_device[i] * p_device[1, i, k]
            for k in range(n_dim):
                p_device[1, i, k] = edt * p_device[1, i, k] + sqf * xoroshiro128p_uniform_float32(rng_states, i * 3 + k)
            for k in range(n_dim):
                q_device[1, i, k] = q_device[1, i, k] + thm_device[i] * p_device[1, i, k]

    # lagevin exectuion second part, after force calculation
    @cuda.jit
    def kernel_lagevin_2(p_device, force_device, thm_device):
        i = cuda.grid(1)
        if i < p_device.shape[0]:
            for k in range(n_dim):
                p_device[i, k] = p_device[i, k] + thm_device[i] * force_device[i, k]

    # Cuda kernel for force calculattion
    @cuda.jit
    def kernel_calc_force(q, out, particle_types, lj_mixing_conditions):
        i = cuda.grid(1)
        if i < n_particles:
            pos_i = q[i, :]
            dv = cuda.local.array(n_dim, dtype=dtype)
            force = cuda.local.array(n_dim, dtype=dtype)
            for k in range(n_dim):
                out[i, k] = global_force(q[i, k])
            for j in range(n_particles):
                if i != j:
                    distance_squared = 0
                    for k in range(n_dim):
                        dv[k] = pos_i[k] - q[j, k]
                        distance_squared += dv[k]**2
                    distance = math.sqrt(distance_squared)
                    type_i = particle_types[i]
                    type_j = particle_types[j]
                    for k in range(n_dim):
                        out[i, k] += dv[k] * lj_force_pairwise(distance, distance_squared, type_i, type_j)
            # for k in range(n_dim):
            #     out[i, k] += force[k]

    @cuda.jit(device=True)
    def lj_force_pairwise(distance, distance_squared, type_i, type_j):
        # fetch miixing conditions
        sigma = lj_mixing_conditions[type_i * n_particles_tpyes_internal + type_j][0]
        sigma6 = lj_mixing_conditions[type_i * n_particles_tpyes_internal + type_j][1]
        epsilon = lj_mixing_conditions[type_i * n_particles_tpyes_internal + type_j][2]
        # calculate force below switch region, assumes all distances passed > 0
        if(distance <= switch_start) and (distance > 0):
            return (24 * epsilon * sigma6 / distance_squared**4 * (2 * sigma6 / distance_squared**3 - 1))
        # calculate potential in switch region, (gradient * switch -potential * dswitch)
        if (distance > switch_start) and (distance <= cutoff):
            t = (distance - cutoff) / switch_width
            gradient = 24 * epsilon * sigma6 / distance**8 * (2 * sigma6 / distance**6 - 1)
            potential = 4. * epsilon * sigma6 / distance_squared**3 * (sigma6 / distance_squared**3 - 1)
            switch = 2 * t**3 + 3 * t**2
            dswitch = 6 / (cutoff - switch_start) / distance * (t**2 + t)
            output = gradient * switch - potential * dswitch
            return output
        # set rest to 0
        else:
            return 0.

    @cuda.jit(device=True)
    def global_force(q):
        return -2. * k * q

    # run simulation
    kernel_calc_force[blockspergrid, threadsperblock](q_device[0, :, :], force_device, particle_types_device, lj_mixing_conditions_device)
    for n in range(n_steps):
        kernel_lagevin_1[blockspergrid, threadsperblock](q_device[n : n + 1, :, :], p_device[n : n + 1, :, :],
                                                         force_device, thm_device, rng_states)
        kernel_calc_force[blockspergrid, threadsperblock](q_device[n + 1, :, :], force_device, particle_types_device, lj_mixing_conditions_device)
        kernel_lagevin_2[blockspergrid, threadsperblock](p_device[n + 1, :, :], force_device, thm_device)

    # copy trajectory
    return q_device.copy_to_host(), p_device.copy_to_host()
    # return force_device.copy_to_host()
