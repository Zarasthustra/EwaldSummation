import numpy as np
import math
from numba import cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64

# def initializer(n_particles, n_dim):
#     # need to return a tuple of four vectors
#     # masses, charges, q_0 and p_0
#     masses = np.array([22.990, 35.453] * int(n_particles / 2))
#     charges = np.array([1, -1] * int(n_particles / 2) )
#     q_0 = np.arange(n_particles * n_dim).reshape(n_particles, n_dim)
#     v_0 = np.zeros((n_particles, n_dim))
#     particle_types = [0, 1] * int(n_particles / 2)
#     lj_mixing_conditions = [(2.35, 2.35**6, 0.130, 1),
#                             (0.5 * (2.35 + 4.4), (0.5 * (2.35 + 4.4))**6, 0.13, -1),
#                             (0.5 * (2.35 + 4.4), (0.5 * (2.35 + 4.4))**6, 0.13, -1),
#                             (4.40, 4.40**6, 0.1, 1),
#                             ]
#     return q_0, v_0 * masses[:, None], masses, charges, particle_types, lj_mixing_conditions
#
# # tuple([(2.35, 2.35**6, 0.130, 1),
# #                               (0.5 * (2.35 + 4.4), (0.5 * (2.35 + 4.4))**6, 0.13, -1),
# #                               (0.5 * (2.35 + 4.4), (0.5 * (2.35 + 4.4))**6, -1),
# #                               (4.40, 4.40**6, 0.1, -1),
# #                               ])
#
# cutoff = 3.405 * 3.5
# damping = 0.01
# n_steps = 100
# n_particles = 2
# n_dim = 3
# damping = 0
# l_box = np.array([100, 100, 100])


# def initializer(arg1, arg2):
#     # need to return a tuple of four vectors
#     # masses, charges, q_0 and p_0
#     masses = np.array([1., 1.])
#     charges = np.array([-1., 1.])
#     q_0 = np.array([[0., 1., 0], [1., 0., 0]])
#     v_0 = np.array([[0.5, 0.866, 0], [-0.8, 0.6, 0]])
#     return masses, charges, q_0, v_0 * masses[:, None]
#
# def _grid(ns):
#     xx = np.meshgrid(*[np.arange(0, n) for n in ns])
#     X = np.vstack([v.reshape(-1) for v in xx]).T
#     return X
#
# def _intializer_NaCl(n):
#     n_particles = n * n * n
#     n_dim = 3
#     l_box = 4. * np.array([n, n, n])
#     grid=_grid([n, n, n])
#     q = 4. * grid
#     particle_info = grid.sum(axis=1)%2+3
#     return q, particle_info, l_box
#
#
# q_0, particle_info, l_box = _intializer_NaCl(2)
# p_0 = np.zeros(q_0.shape)
#
# phy_world = es.PhysWorld()
# config = es.SimuConfig(l_box=(100, 100, 100), n_steps=1000, timestep=0.01, temp=300, particle_info=(0, 1))
# config.accuracy = 1E-8
# # config.dtype = 'float32'
# damping = 0
# particle_index, mixing_conditions, masses, charges = calc_mixing_conditions(particle_info, config.phys_world.particle_types)
# config.masses = masses
# config.charges = charges




def lagevin_coulomb_lj_cuda(q_0, p_0, masses, charges, particle_index, mixing_conditions, l_box, n_steps, cutoff_lj,
                 accuracy=1e-8, damping=0.1, time_step=0.1, temp=300, k_B=0.00198720360, k_C=332.0637128,):

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
            self.f_rec_prefactor = -charges / (l_box[0] * l_box[1] * l_box[2]) # j and 2pi parts come here

        def grid_points_without_center(self, nx, ny, nz):
            a, b, c = np.arange(-nx, nx+1), np.arange(-ny, ny+1), np.arange(-nz, nz+1)
            xx, yy, zz = np.meshgrid(a, b, c)
            X = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T
            return np.delete(X, X.shape[0] // 2, axis=0)

    # general params
    n_dim = q_0.shape[1]
    n_particles = q_0.shape[0]
    dtype = float64

    # lagevin params
    beta = 1. / (k_B * temp)
    shape = (n_particles, n_dim)
    th = 0.5 * time_step
    thm = 0.5 * time_step / masses
    edt = np.exp(-damping * time_step)
    sqf = np.sqrt((1.0 - edt ** 2) / beta)

    # optimized for n^3 NaCl grid, 8<n<20
    ratio_real_rec = 5.5 # super parameter for balancing the calculation time of real
    # and reciprocal part. Ideally make them the same and total time achieving O(n^1.5)
    # In reality the ratio between real and reciprocal times may vary, but should be
    # within the 0.1 to 10.

    # optimal alpha and cutoff selections
    # ref: http://protomol.sourceforge.net/ewald.pdf
    V = l_box[0] * l_box[1] * l_box[2]
    alpha = ratio_real_rec * np.sqrt(np.pi) * (n_particles / V / V) ** (1/6)
    real_cutoff = np.sqrt(-np.log(accuracy)) / alpha
    rec_reso = int(np.ceil(np.sqrt(-np.log(accuracy)) * 2 * alpha))
    switch_width_lj = 1
    switch_start_lj = cutoff_lj - switch_width_lj
    precalc = Coulomb_PreCalc(l_box, charges, rec_reso, alpha)
    m = np.transpose(precalc.m)
    S_m_cos_parts = np.zeros((n_particles, m.shape[1]))
    S_m_sin_parts = np.zeros((n_particles, m.shape[1]))
    S_m_cos_sum = np.zeros(m.shape[1])
    S_m_sin_sum = np.zeros(m.shape[1])
    dS_modul_sq = np.zeros((n_particles, m.shape[1]))

    # Copy the arrays to the device
    q_device = cuda.device_array((n_steps + 1, n_particles, n_dim))
    q_device[0, :, :] = q_0
    p_device = cuda.device_array((n_steps + 1, n_particles, n_dim))
    p_device[0, :, :] = p_0
    thm_device = cuda.to_device(thm)
    force_device = cuda.device_array(shape)
    particle_index_device = cuda.to_device(particle_index)
    mixing_conditions_device = cuda.to_device(mixing_conditions)
    l_box_device = cuda.to_device(l_box)
    m_device = cuda.to_device(m)
    S_m_cos_parts_device = cuda.to_device(S_m_cos_parts)
    S_m_sin_parts_device = cuda.to_device(S_m_sin_parts)
    S_m_cos_sum_device = cuda.to_device(S_m_cos_sum)
    S_m_sin_sum_device = cuda.to_device(S_m_sin_sum)
    dS_modul_sq_device = cuda.to_device(dS_modul_sq)
    coeff_S_device  = cuda.to_device(precalc.coeff_S)
    charges_device = cuda.to_device(charges)
    f_rec_prefactor_device = cuda.to_device(precalc.f_rec_prefactor)

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
    def kernel_calc_force(q, out, l_box, particle_types):
        i = cuda.grid(1)
        dv = cuda.local.array(n_dim, dtype=dtype)
        # force = cuda.local.array(n_dim, dtype=dtype)
        if i < n_particles:
            pos_i = q[i, :]
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


    @cuda.jit(device=True)
    def lj_force_pairwise(distance, distance_squared, type_i, type_j):
        # fetch miixing conditions
        if type_i == 0 and type_j == 0:
            sigma = 2.35
            sigma6 = 168.425
            epsilon = 0.13
        if type_i == 1 and type_j == 0:
            sigma = 3.375
            sigma6 = 1477.892
            epsilon = 0.111402
        if type_i == 0 and type_j == 1:
            sigma = 3.375
            sigma6 = 1477.892
            epsilon = 0.111402
        if type_i == 0 and type_j == 1:
            sigma = 4.40
            sigma6 = 7256.314
            epsilon = 0.1
        # calculate force below switch region, assumes all distances passed > 0
        if(distance <= switch_start_lj) and (distance > 0):
            return (24 * epsilon * sigma6 / distance_squared**4 * (2 * sigma6 / distance_squared**3 - 1))
        # calculate potential in switch region, (gradient * switch -potential * dswitch)
        if (distance > switch_start_lj) and (distance <= cutoff_lj):
            t = (distance - cutoff_lj) / switch_width_lj
            gradient = 24 * epsilon * sigma6 / distance**8 * (2 * sigma6 / distance**6 - 1)
            potential = 4. * epsilon * sigma6 / distance_squared**3 * (sigma6 / distance_squared**3 - 1)
            switch = 2 * t**3 + 3 * t**2
            dswitch = 6 / (cutoff_lj - switch_start_lj) / distance * (t**2 + t)
            output = gradient * switch - potential * dswitch
            return output
        # set rest to 0
        else:
            return 0.

    @cuda.jit
    def kernel_calc_force_rec(q, out, m, S_m_cos_parts, S_m_sin_parts, S_m_cos_sum,
                              S_m_sin_sum, dS_modul_sq, charges, f_rec_prefactor, coeff_S):
        i = cuda.grid(1)
        if i < n_particles:
            for j in range(m.shape[1]):
                for k in range(n_dim):
                    S_m_cos_parts[i, j] += q[i, k] * m[k, j]
                    S_m_sin_parts[i, j] += q[i, k] * m[k, j]
                S_m_cos_parts[i, j] = math.cos(2 * math.pi * S_m_cos_parts[i, j]) * charges[i]
                S_m_sin_parts[i, j] = math.sin(2 * math.pi * S_m_sin_parts[i, j]) * charges[i]
                S_m_cos_sum[j] += S_m_cos_parts[i, j]
                S_m_sin_sum[j] += S_m_sin_parts[i, j]
            cuda.syncthreads()
            for j in range(m.shape[1]):
                dS_modul_sq[i, j] = coeff_S[j] * (2. * S_m_cos_sum[j] * S_m_sin_parts[i, j] - 2. * S_m_sin_sum[j] * S_m_cos_parts[i, j])
            for j in range(m.shape[1]):
                for k in range(n_dim):
                    out[i, k] += f_rec_prefactor[i] * dS_modul_sq[i, j] * m[k, j]

    @cuda.jit(device=True)
    def coulomb_force_pairwise(distance, distance_squared, type_i, type_j):
        if type_i == type_j:
            charge_product_ij = 1
        else:
            charge_product_ij = -1
        if distance < real_cutoff:
            real_part = math.erfc(alpha * distance) / distance + \
                        2 * alpha / math.sqrt(np.pi) * math.exp(-(alpha ** 2 * distance_squared))
            return 0.5 * k_C * charge_product_ij * (real_part / distance_squared)
        else:
            return 0

    # run simulation
    kernel_calc_force_rec[blockspergrid, threadsperblock](q_device[0, :, :],
                                                          force_device,
                                                          m_device,
                                                          S_m_cos_parts_device,
                                                          S_m_sin_parts_device,
                                                          S_m_cos_sum_device,
                                                          S_m_sin_sum_device,
                                                          dS_modul_sq_device,
                                                          charges_device,
                                                          f_rec_prefactor_device,
                                                          coeff_S_device,
                                                          )
    kernel_calc_force[blockspergrid, threadsperblock](q_device[0, :, :],
                                                      force_device,
                                                      l_box_device,
                                                      particle_index_device,
                                                      )
    for n in range(n_steps):
        kernel_lagevin_1[blockspergrid, threadsperblock](q_device[n : n + 1, :, :],
                                                         p_device[n : n + 1, :, :],
                                                         force_device,
                                                         thm_device,
                                                         rng_states,
                                                         )
        kernel_calc_force_rec[blockspergrid, threadsperblock](q_device[0, :, :],
                                                              force_device,
                                                              m_device,
                                                              S_m_cos_parts_device,
                                                              S_m_sin_parts_device,
                                                              S_m_cos_sum_device,
                                                              S_m_sin_sum_device,
                                                              dS_modul_sq_device,
                                                              charges_device,
                                                              f_rec_prefactor_device,
                                                              coeff_S_device,
                                                              )
        kernel_calc_force[blockspergrid, threadsperblock](q_device[n + 1, :, :],
                                                          force_device,
                                                          l_box_device,
                                                          particle_index_device,
                                                          )
        kernel_lagevin_2[blockspergrid, threadsperblock](p_device[n + 1, :, :],
                                                         force_device,
                                                         thm_device,
                                                         )

    # copy trajectory
    return q_device.copy_to_host(), p_device.copy_to_host()
