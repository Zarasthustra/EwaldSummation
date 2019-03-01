import numpy as np
import ewald_summation as es
import math
import matplotlib.pyplot as plt
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64, float32, float64


def initializer(arg1, arg2):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.array([1., 1.])
    charges = np.array([-1., 1.])
    q_0 = np.array([[0., 1., 0], [1., 0., 0]])
    v_0 = np.array([[0.5, 0.866, 0], [-0.8, 0.6, 0]])
    return masses, charges, q_0, v_0 * masses[:, None]


masses, charges, q_0, p_0 = initializer(1, 1)
phy_world = es.PhysWorld()
config = es.SimuConfig(n_dim=3, l_box=(100, 100, 100), n_particles=2, n_steps=10000, timestep=0.01, temp=300)
config.dtype = 'float32'
config.masses = masses
config.charges = charges
damping = 0


def calc_force_coulomb_rec(q, m, charges, f_rec_prefactor, coeff_S):
    m_dot_q = np.matmul(q, np.transpose(m))
    middle0 = np.pi * 2. * m_dot_q
    S_m_cos_parts = charges[:, np.newaxis] * np.cos(middle0)
    S_m_sin_parts = charges[:, np.newaxis] * np.sin(middle0)
    S_m_cos_sum, S_m_sin_sum = S_m_cos_parts.sum(axis=0), S_m_sin_parts.sum(axis=0)
    dS_modul_sq = 2. * S_m_cos_sum * S_m_sin_parts - 2. * S_m_sin_sum * S_m_cos_parts
    f_rec = f_rec_prefactor[..., None] * (coeff_S * dS_modul_sq).dot(m)
    return f_rec


class Coulomb_PreCalc:
    def __init__(self, l_box, charges, rec_resolution, alpha):
        self.m = (1/l_box) * self.grid_points_without_center(rec_resolution, rec_resolution, rec_resolution)
        m_modul_sq = np.linalg.norm(self.m, axis = 1) ** 2
        self.coeff_S = np.exp(-(np.pi / alpha) ** 2 * m_modul_sq) / m_modul_sq
        self.v_rec_prefactor = 0.5 / np.pi / (l_box[0] * l_box[1] * l_box[2])
        self.f_rec_prefactor = -charges / (l_box[0] * l_box[1] * l_box[2]) # j and 2pi parts come here
        self.v_self = -alpha / np.sqrt(np.pi) * np.sum(charges**2)

    def grid_points_without_center(self, nx, ny, nz):
        a, b, c = np.arange(-nx, nx+1), np.arange(-ny, ny+1), np.arange(-nz, nz+1)
        xx, yy, zz = np.meshgrid(a, b, c)
        X = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T
        return np.delete(X, X.shape[0] // 2, axis=0)


def lagevin_cuda(q_0, p_0, phy_world, config, damping):
    # general params
    n_dim = config.n_dim
    n_particles = config.n_particles
    n_steps = config.n_steps
    dtype = float64
    time_step = config.timestep
    masses = config.masses
    l_box = np.array(config.l_box)
    beta = 1. / (phy_world.k_B * config.temp)
    shape = (config.n_particles, config.n_dim)
    th = 0.5 * time_step
    thm = 0.5 * time_step / masses
    edt = np.exp(-damping * time_step)
    sqf = np.sqrt((1.0 - edt ** 2) / beta)
    k = 1

    # particle params params (sigma, sigma6, epsilon, charge)
    particle_types = (0, 1)
    lj_mixing_conditions = tuple([(1, 1, 1, 1), (1, 1, 1, -1)] * int(n_particles**2 / 2))
    switch_start = 2.5
    cutoff = 3.5
    switch_width = 1
    alpha = 1.
    rec_reso = config.rec_reso
    charges = config.charges
    precalc = Coulomb_PreCalc(l_box, charges, rec_reso, alpha)
    force_0 = calc_force_coulomb_rec(q_0, precalc.m, charges, precalc.f_rec_prefactor, precalc.coeff_S)

    # Copy the arrays to the device
    q_device = cuda.device_array((n_steps + 1, n_particles, n_dim))
    q_device[0, :, :] = q_0
    p_device = cuda.device_array((n_steps + 1, n_particles, n_dim))
    p_device[0, :, :] = p_0
    thm_device = cuda.to_device(thm)
    force_0 = cuda.to_device(force_0)
    force_device = cuda.device_array(shape)
    particle_types_device = cuda.to_device(particle_types)
    lj_mixing_conditions_device = cuda.to_device(lj_mixing_conditions)
    l_box_device = cuda.to_device(l_box)

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

    # Cuda kernel for force calculattion, with minium image convention for distance vectorsh
    @cuda.jit
    def kernel_calc_force(q, out, l_box, particle_types, lj_mixing_conditions):
        i = cuda.grid(1)
        dv = cuda.local.array(n_dim, dtype=dtype)
        # force = cuda.local.array(n_dim, dtype=dtype)
        if i < n_particles:
            pos_i = q[i, :]
            for k in range(3):
                out[i, k] = 0
            for j in range(n_particles):
                if i != j:
                    distance_squared = 0
                    for k in range(n_dim):
                        dv[k] = math.fmod(pos_i[k] - q[j, k], l_box[k])
                        if dv[k] > l_box[k] / 2:
                            dv[k] -= l_box[k]
                        distance_squared += dv[k]**2
                    distance = math.sqrt(distance_squared)
                    type_i = particle_types[i]
                    type_j = particle_types[j]
                    for k in range(n_dim):
                        out[i, k] += dv[k] * lj_force_pairwise(distance, distance_squared, type_i, type_j)
                        out[i, k] += dv[k] * coulomb_force_pairwise(distance, distance_squared, type_i, type_j)
            # for k in range(n_dim):
            #     out[i, k] += force[k]

    # Cuda kernel for addition of two force arrays on kernel
    @cuda.jit
    def kernel_add_forces(force1, force2):
        i = cuda.grid(1)
        if i < n_particles:
            for k in range(3):
                force1[i, k] += force2[i, k]

    @cuda.jit(device=True)
    def lj_force_pairwise(distance, distance_squared, type_i, type_j):
        # fetch miixing conditions
        sigma = lj_mixing_conditions[type_i * n_particles + type_j][0]
        sigma6 = lj_mixing_conditions[type_i * n_particles + type_j][1]
        epsilon = lj_mixing_conditions[type_i * n_particles + type_j][2]
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
    def coulomb_force_pairwise(distance, distance_squared, type_i, type_j):
        charge_product_ij = lj_mixing_conditions[type_i * n_particles + type_j][3]
        real_part = math.erfc(alpha * distance) / distance + \
                    2 * alpha / math.sqrt(np.pi) * math.exp(-(alpha ** 2 * distance_squared))
        return charge_product_ij * (real_part / distance_squared)

    # run simulation
    kernel_calc_force[blockspergrid, threadsperblock](q_device[0, :, :],
                                                      force_device,
                                                      l_box_device,
                                                      particle_types_device,
                                                      lj_mixing_conditions_device,
                                                      )
    for n in range(n_steps):
        kernel_lagevin_1[blockspergrid, threadsperblock](q_device[n : n + 1, :, :],
                                                         p_device[n : n + 1, :, :],
                                                         force_device,
                                                         thm_device,
                                                         rng_states,
                                                         )
        cuda.synchronize()
        kernel_calc_force[blockspergrid, threadsperblock](q_device[n + 1, :, :],
                                                          force_device,
                                                          l_box_device,
                                                          particle_types_device,
                                                          lj_mixing_conditions_device,
                                                          )
        q_temp = q_device[n + 1, :, :].copy_to_host()
        coulomb_force_rec = calc_force_coulomb_rec(q_temp, precalc.m, charges, precalc.f_rec_prefactor, precalc.coeff_S)
        coulomb_force_rec_device = cuda.to_device(coulomb_force_rec)
        cuda.synchronize()
        kernel_add_forces[blockspergrid, threadsperblock](force_device, coulomb_force_rec_device)
        kernel_lagevin_2[blockspergrid, threadsperblock](p_device[n + 1, :, :],
                                                         force_device,
                                                         thm_device,
                                                         )

    # copy trajectory
    return q_device.copy_to_host(), p_device.copy_to_host()

q_cuda, p_cuda = lagevin_cuda(q_0, p_0, phy_world, config, damping)

plt.plot(q_cuda[:, 0, 0], q_cuda[:, 0, 1])
plt.plot(q_cuda[:, 1, 0], q_cuda[:, 1, 1])

plt.show()
