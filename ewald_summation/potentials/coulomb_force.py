import numpy as np
from numba import njit
import math


@njit
def calc_force_coulomb_real(x, n_dim, n_particles, charges, alpha, l_box, l_box_half, cutoff):
        force = np.zeros((n_particles, n_dim))
        for i in range(n_particles):
            for j in range(n_particles):
                if i != j:
                    # calc dist_square, dist
                    distance_vector = np.array([0, 0, 0])
                    # minimum image convention for pbc
                    for k in range(n_dim):
                        distance_vector[k] = (x[i, k] - x[j, k]) % l_box[k]
                        if distance_vector[k] > l_box_half[k]:
                            distance_vector[k] -= l_box[k]
                    distance_squared = np.sum(distance_vector**2)
                    distance = math.sqrt(distance_squared)
                    # calc coulomb force
                    if distance < cutoff:
                        force[i, :] += coulomb_force_pairwise(distance_vector,
                                                              distance,
                                                              distance_squared,
                                                              charges[j],
                                                              alpha)
        return force


def calc_force_coulomb_rec(q, m, charges, f_rec_prefactor, coeff_S):
    m_dot_q = np.matmul(q, np.transpose(m))
    # print('new', m_dot_q.sum())
    middle0 = np.pi * 2. * m_dot_q
    S_m_cos_parts = charges[:, np.newaxis] * np.cos(middle0)
    S_m_sin_parts = charges[:, np.newaxis] * np.sin(middle0)
    S_m_cos_sum, S_m_sin_sum = S_m_cos_parts.sum(axis=0), S_m_sin_parts.sum(axis=0)
    dS_modul_sq = 2. * S_m_cos_sum * S_m_sin_parts - 2. * S_m_sin_sum * S_m_cos_parts
    # print('new_sin', S_m_cos_sum.sum())
    f_rec = f_rec_prefactor[..., None] * (coeff_S * dS_modul_sq).dot(m)
    # print('new_mod', S_m_modul_sq.sum())
    # print('new', v_rec)
    return f_rec


@njit
def coulomb_force_pairwise(distance_vector, distance, distance_squared, charge_j, alpha):
            real_part = math.erfc(alpha * distance) / distance + \
                        2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha ** 2 * distance_squared))
            return charge_j * (real_part / distance_squared) * distance_vector
