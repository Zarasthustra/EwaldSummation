import numpy as np
from scipy.special import erfc
from numba import jit
from numba import njit

class Coulomb:
    def __init__(self, config):
        self.n_dim = config.n_dim
        assert self.n_dim == 3, "For other dimensions not implemented."
        self.l_box = config.l_box
        self.neighbour = config.neighbour
        # self.epsilon = config.epsilon_coulomb
        self.epsilon = 1. / (4. * np.pi)
        self.prefactor = 1. / (4. * np.pi * self.epsilon)
        self.charge_vector = config.charges
        self.alpha = 1.
        # self.REAL_CUTOFF = config.cutoff_coulomb_real_sum
        self.REAL_CUTOFF = 8
        self.REAL_PERIODIC_REPEATS = np.clip(np.ceil(self.REAL_CUTOFF / self.l_box) - 1, 0, 5).astype(int)
        # self.REC_RESO = config.cutoff_coulomb_reci_sum
        self.REC_RESO = 6
        self.precalc = self.Coulomb_PreCalc(self.l_box, self.charge_vector, self.REAL_PERIODIC_REPEATS,
                                            self.REC_RESO, self.alpha)
        self.neighbor_charge_frame_num = -1000  # For recalc detection
        self.neighbor_charge_list = None

    class Coulomb_PreCalc:
        def __init__(self, l_box, charge_vector, REAL_PERIODIC_REPEATS, rec_resolution, alpha):
            self.real_periods = l_box * _grid_points_without_center(*REAL_PERIODIC_REPEATS)
            self.m = (1/l_box) * _grid_points_without_center(rec_resolution, rec_resolution, rec_resolution)
            m_modul_sq = np.linalg.norm(self.m, axis = 1) ** 2
            self.coeff_S = np.exp(-(np.pi / alpha) ** 2 * m_modul_sq) / m_modul_sq
            self.v_rec_prefactor = 0.5 / np.pi / (l_box[0] * l_box[1] * l_box[2])
            self.f_rec_prefactor = -charge_vector / (l_box[0] * l_box[1] * l_box[2]) # j and 2pi parts come here
            self.v_self = -alpha / np.sqrt(np.pi) * np.sum(charge_vector**2)

    def calc_potential(self, q, current_frame):
        self._check_new_frame(current_frame)
        output = _ewald_pot2(q, current_frame.distances, current_frame.distance_vectors,
                             self.charge_vector, self.neighbor_charge_list, self.precalc,
                             alpha=self.alpha
                            )
        output *= self.prefactor
        return output

    def calc_force(self, q, current_frame):
        self._check_new_frame(current_frame)
        output = _ewald_force(q, current_frame.distances, current_frame.distances_squared,
                             current_frame.distance_vectors, self.charge_vector,
                             self.neighbor_charge_list, self.precalc, alpha=self.alpha, n_particles=q.shape[0]
                            )
        output *= self.prefactor
        return output

    def _check_new_frame(self, current_frame):
        if(self.neighbor_charge_frame_num != current_frame.step):
            self.neighbor_charge_list = _neighbor_charge_list_recalc(self.charge_vector, current_frame.distances,
                                                                     current_frame.array_index)
            self.neighbor_charge_frame_num = current_frame.step

@jit
def _neighbor_charge_list_recalc(charge_vector, distances, array_index):
    neighbor_charge_list = np.zeros_like(array_index, dtype=float)
    for i in range(array_index.shape[0]):
        for j in range(array_index.shape[1]):
            neighbor_charge_list[i, j] = charge_vector[array_index[i, j]]
    #print(distances)
    return neighbor_charge_list


# def _grid_points_without_center(n):
#     a = np.arange(-n, n+1)
#     xx, yy, zz = np.meshgrid(a, a, a)
#     X = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T
#     return np.delete(X, X.shape[0] // 2, axis=0)

def _grid_points_without_center(nx, ny, nz):
    a, b, c = np.arange(-nx, nx+1), np.arange(-ny, ny+1), np.arange(-nz, nz+1)
    xx, yy, zz = np.meshgrid(a, b, c)
    X = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T
    return np.delete(X, X.shape[0] // 2, axis=0)

# @jit
def _ewald_pot2(q, distances, distance_vectors, charge_vector, neighbor_charge_list,
                precalc, alpha):

    # M = rec_resolution # reciprocal periodic radius
    # v_real = v_rec = v_self = 0.
    v_real = v_rec = 0.
    # charge_product_i_j = charge_vector[:, np.newaxis] * charge_vector[np.newaxis, :]
    charge_product_i_j = charge_vector[:, np.newaxis] * neighbor_charge_list

    # real part
    # for i,j in combinations(np.arange(n_particles),2):
    #     real_part = erfc(alpha * distances[i,j]) / distances[i,j]
    #     v_real += charge_product_i_j[i, j] * real_part
    dist_masked = np.ma.masked_values(distances, 0.)
    v_real = 0.5 * np.sum(charge_product_i_j * erfc(alpha * dist_masked) / dist_masked)

    # --- only useful for cases that minimum image convention is not enough
    # e.g. when alpha is too big
    # REAL_PERIODIC_REPEATS = int(np.ceil(8. / l_box)) - 1
    # ks = np.array(list(product(range(-REAL_PERIODIC_REPEATS, REAL_PERIODIC_REPEATS+1), repeat=3)))
    # ks = l_box * np.delete(ks, ks.shape[0] // 2, 0)
    # ks = l_box * _grid_points_without_center(REAL_PERIODIC_REPEATS)
    #for k in ks:
    for k in precalc.real_periods:
        image_distances = np.linalg.norm(distance_vectors + k, axis=-1)
        v_real += 0.5 * (charge_product_i_j * (erfc(alpha * image_distances) / image_distances)).sum()
        #for i in range(n_particles):
        #    for j in range(n_particles):
        #        distance = np.linalg.norm(distance_vectors[i, j] + k)
        #        v_real += 0.5 * charge_vector[i] * charge_vector[j] * erfc(alpha * distance) / distance
    # ---

    def calc_pot_coulomb_reciprocal(x, k, prefactor):
        pot_rec = 0
        for i in range(x.shape[0]):
            for j in range(k.shape[0]):
                r_dot_k_2pi = 2 * np.pi * (x[i, 0] * k[j, 0] + x[i, 1] * k[j, 1] + x[i, 2] * k[j, 2])
                pot_rec += np.sin(r_dot_k_2pi)**2 + np.cos(r_dot_k_2pi)**2
        return pot_rec * precalc.v_rec_prefactor

    # v_rec_1 = calc_pot_coulomb_reciprocal(q, precalc.m, precalc.v_rec_prefactor)
    # def calc_pot_coulomb_reciprocal(x, k, prefactor):
    # m_dot_q = np.matmul(q, np.transpose(precalc.m))
    # middle0 = np.pi * 2. * m_dot_q
    # S_m_cos_sum = charge_vector.dot(np.cos(middle0))
    # S_m_sin_sum = charge_vector.dot(np.sin(middle0))
    # S_m_modul_sq = np.square(S_m_cos_sum) + np.square(S_m_sin_sum)
    # v_rec = precalc.v_rec_prefactor * np.sum(precalc.coeff_S * S_m_modul_sq)
    # return v_rec

    # reciprocal part
    # m = np.array(list(product(range(-M, M+1), repeat=3)))
    # m = (1/l_box) * np.delete(m, m.shape[0] // 2, 0)
    # m = (1/l_box) * _grid_points_without_center(M)
    m_dot_q = np.matmul(q, np.transpose(precalc.m))
    # print('old', m_dot_q.sum())
    ## structure factor
    middle0 = np.pi * 2. * m_dot_q
    #middle1 = np.pi * 2.j * m_dot_q
    #middle2 = np.exp(middle1)
    #middle3 = np.sin(middle0)
    #S_cos_part = np.cos(middle0)
    #S_sin_part = np.sin(middle0)
    S_m_cos_sum = charge_vector.dot(np.cos(middle0))
    S_m_sin_sum = charge_vector.dot(np.sin(middle0))
    # print('old_sin', S_m_cos_sum.sum())
    S_m_modul_sq = np.square(S_m_cos_sum) + np.square(S_m_sin_sum)
    # print('old_mod', S_m_modul_sq.sum())
    #S_m_modul_sq = np.real(np.conjugate(S_m) * S_m)
    # m_modul_sq = np.linalg.norm(m, axis = 1) ** 2
    # coeff_S = np.exp(-(np.pi / alpha) ** 2 * m_modul_sq) / m_modul_sq
    # v_rec = 0.5 / np.pi / (l_box ** 3)
    #
    v_rec = precalc.v_rec_prefactor * np.sum(precalc.coeff_S * S_m_modul_sq)

    # self part
    # v_self = -alpha / np.sqrt(np.pi) * np.sum(charge_vector**2)
    # print('v_real', v_real)
    # print('v_rec', v_rec)
    # print('v_self', v_self)
    # return v_real + v_rec + v_self
    # print('old', v_rec)
    return  v_rec + v_real + precalc.v_self

@jit
def _ewald_force(q, distances, dist_sq, distance_vectors, charge_vector, neighbor_charge_list,
                precalc, alpha, n_particles):

    # M = resolution # reciprocal periodic radius
    # f_real = f_rec = np.zeros(n_dim)
    f_real = f_rec = np.zeros((n_particles, 3))
    # real part
    # for j in np.arange(n_particles):
    #     if(j != i):
    #         real_part = erfc(alpha * distances[i,j]) / distances[i,j] + \
    #             2 * alpha / np.sqrt(np.pi) * np.exp(-((alpha * distances[i,j]) ** 2))
    #         f_real += charge_vector[j] * (real_part / (distances[i,j]) ** 2) * distance_vectors[i, j]
    dist_masked = np.ma.masked_values(distances, 0.)
    dist_sq_masked = np.ma.array(dist_sq, mask=dist_masked.mask)
    neighbor_masked = np.ma.array(neighbor_charge_list, mask=dist_masked.mask)
    real_part = erfc(alpha * dist_masked) / dist_masked + \
                2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha ** 2 * dist_sq_masked))
    f_real += ((real_part / dist_sq_masked * neighbor_masked)[..., None] * distance_vectors).sum(axis=1)
    # --- only useful for cases that minimum image convention is not enough
    # e.g. when alpha is too big
    # REAL_PERIODIC_REPEATS = int(np.ceil(8. / l_box)) - 1
    # ks = np.array(list(product(range(-REAL_PERIODIC_REPEATS, REAL_PERIODIC_REPEATS+1), repeat=3)))
    # ks = l_box * np.delete(ks, ks.shape[0] // 2, 0)
    # ks = l_box * _grid_points_without_center(REAL_PERIODIC_REPEATS)
    # for k in ks:
    for k in precalc.real_periods:
        for i in range(n_particles):
            for j in range(n_particles):
                distance_vector = distance_vectors[i, j] + k
                distance = np.linalg.norm(distance_vectors[i, j] + k)
                real_part = erfc(alpha * distance) / distance + \
                    2 * alpha / np.sqrt(np.pi) * np.exp(-((alpha * distance) ** 2))
                f_real[i] += neighbor_charge_list[i, j] * (real_part / distance ** 2) * distance_vector
    # ---
    f_real *= charge_vector[..., None]
    # reciprocal part
    # m = np.array(list(product(range(-M, M+1), repeat=3)))
    # m = (1/l_box) * np.delete(m, m.shape[0] // 2, 0)
    # m = (1/l_box) * _grid_points_without_center(M)
    m_dot_q = np.matmul(q, np.transpose(precalc.m))
    ## structure factor
    # S_m = charge_vector.dot(np.exp(np.pi * 2.j * m_dot_q))
    # dS_star_ms = np.exp(-np.pi * 2.j * m_dot_q) # *2pi*j*q_i
    middle0 = np.pi * 2. * m_dot_q
    S_m_cos_parts = charge_vector[:, np.newaxis] * np.cos(middle0)
    S_m_sin_parts = charge_vector[:, np.newaxis] * np.sin(middle0)
    S_m_cos_sum, S_m_sin_sum = S_m_cos_parts.sum(axis=0), S_m_sin_parts.sum(axis=0)
    dS_modul_sq = 2. * S_m_cos_sum * S_m_sin_parts - 2. * S_m_sin_sum * S_m_cos_parts
    #S_m_modul_sq = np.real(np.conjugate(S_m) * S_m)
    # m_modul_sq = np.linalg.norm(m, axis = 1) ** 2
    # coeff_S = np.exp(-(np.pi / alpha) ** 2 * m_modul_sq) / m_modul_sq
    # middle5 = -charge_vector / (l_box ** 3) # j and 2pi parts come here
    f_rec = precalc.f_rec_prefactor[..., None] * (precalc.coeff_S * dS_modul_sq).dot(precalc.m)
    # print('f_real', f_real)
    # print('f_rec', f_rec)
    return f_real - f_rec
