import numpy as np
import ewald_summation as es
import math
import matplotlib.pyplot as plt
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64


def initializer(arg1, arg2):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.array([1., 1.])
    charges = np.array([-1., 1.])
    q_0 = np.array([[0., 1.], [1., 0.]])
    v_0 = np.array([[0.5, 0.866], [-0.8, 0.6]])
    return masses, charges, q_0, v_0 * masses[:, None]


masses, charges, q_0, p_0 = initializer(1, 1)
phy_world = es.PhysWorld()
config = es.SimuConfig(n_dim=2, l_box=(2., 2.), n_particles=2, n_steps=5000, timestep=0.01, temp=300)
config.dtype = 'float32'
config.masses = masses
config.charge = charges
damping = 0.05


def lagevin_cuda(q_0, p_0, phy_world, config, damping):
    n_dim = config.n_dim
    n_particles = config.n_particles
    n_steps = config.n_steps
    dtype = config.dtype
    time_step = config.timestep
    masses = config.masses
    beta = 1. / (phy_world.k_B * config.temp)
    shape = (config.n_particles, config.n_dim)
    th = 0.5 * time_step
    thm = 0.5 * time_step / masses
    edt = np.exp(-damping * time_step)
    sqf = np.sqrt((1.0 - edt ** 2) / beta)
    k = 1.

    # Copy the arrays to the device
    q_device = cuda.device_array((n_steps + 1, n_particles, n_dim))
    q_device[0, :, :] = q_0
    p_device = cuda.device_array((n_steps + 1, n_particles, n_dim))
    p_device[0, :, :] = p_0
    thm_device = cuda.to_device(thm)
    force_device = cuda.device_array(shape)

    # Configure the blocks, init random seed
    threadsperblock = 16
    blockspergrid_x = int(math.ceil(q_0.shape[0] / threadsperblock))
    blockspergrid = (blockspergrid_x)
    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid_x * n_dim, seed=1)

    # CUDA kernel
    @cuda.jit
    def kernel_force_pairwise(x, pot, l_box, cutoff, sigma_lj, epsilon_lj):
        switch_width = cutoff - switch_start
        i = cuda.grid(1)
        if i < x.shape[0]:
            for j in range(x.shape[0]):
                if i != j:
                    distance_squared = 0.
                    dv_x = 0
                    dv_y = 0
                    dv_z = 0
                    for k in range(n_dim):
                        distance_temp = (x[i, k] - x[j, k]) % l_box[k]
                        if distance_temp > l_box[k] / 2:
                            distance_temp -= l_box[k]
                        distance_squared += distance_temp**2
                        distance = math.sqrt(distance_squared)
                    if distance < cutoff and distance > 0:
                        sigma_mixed = 0.5 * (sigma_lj[i] + sigma_lj[j])
                        epsilon_mixed = math.sqrt(epsilon_lj[i] * epsilon_lj[j])
                        pot[i, j] += force_pairwise(distance, distance_squared, i, j)

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

    @cuda.jit(device=True)
    def global_force(q):
        return -2. * k * q

    # run simulation
    kernel_calc_force[blockspergrid, threadsperblock](q_device[0, :, :], force_device)
    for n in range(n_steps):
        kernel_lagevin_1[blockspergrid, threadsperblock](q_device[n : n + 1, :, :], p_device[n : n + 1, :, :],
                                                         force_device, thm_device, rng_states)
        kernel_calc_force[blockspergrid, threadsperblock](q_device[n + 1, :, :], force_device)
        kernel_lagevin_2[blockspergrid, threadsperblock](p_device[n + 1, :, :], force_device, thm_device)

    # copy trajectory
    return q_device.copy_to_host(), p_device.copy_to_host()

q_cuda, p_cuda = lagevin_cuda(q_0, p_0, phy_world, config, damping)

plt.plot(q_cuda[:, 0, 0], q_cuda[:, 0, 1])
plt.plot(q_cuda[:, 1, 0], q_cuda[:, 1, 1])

plt.show()
