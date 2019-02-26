import numpy as np
from numba import njit
import math


@njit
def calc_potential_coulomb_real(x, n_dim, n_particles, charges, alpha, l_box, l_box_half, cutoff):
        potential = 0
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                # calc dist_square, dist
                distance_squared = 0
                # minimum image convention for pbc
                for k in range(n_dim):
                    distance_temp = (x[i, k] - x[j, k]) % l_box[k]
                    if distance_temp > l_box_half[k]:
                        distance_temp -= l_box[k]
                    distance_squared += distance_temp**2
                distance = np.sqrt(distance_squared)
                # calc coulomb pot
                if distance < cutoff:
                    potential += coulomb_potential_pairwise(distance, charges[i], charges[j], alpha)
        return potential


def calc_potential_coulomb_rec(q, m, charges, prefactor, coeff_S):
    m_dot_q = np.matmul(q, np.transpose(m))
    # print('new', m_dot_q.sum())
    middle0 = np.pi * 2. * m_dot_q
    S_m_cos_sum = charges.dot(np.cos(middle0))
    S_m_sin_sum = charges.dot(np.sin(middle0))
    # print('new_sin', S_m_cos_sum.sum())
    S_m_modul_sq = np.square(S_m_cos_sum) + np.square(S_m_sin_sum)
    # print('new_mod', S_m_modul_sq.sum())
    v_rec = prefactor * np.sum(coeff_S * S_m_modul_sq)
    # print('new', v_rec)
    return v_rec


@njit
def coulomb_potential_pairwise(distance, charge_i, charge_j, alpha):
    v_real = charge_i * charge_j * math.erfc(alpha * distance) / distance
    return v_real
