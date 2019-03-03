import numpy as np
import math
from .global_template import globalx

@globalx
def CoulombReciprocal(config, alpha, rec_reso):
    # precalc
    n_dim = config.n_dim
    assert n_dim == 3, "For other dimensions not implemented."
    assert config.PBC, "Ewald sum only meaningful for periodic systems."
    prefactor = config.phys_world.k_C
    
    l_box = np.array(config.l_box)
    charges = np.empty(config.n_particles)
    for i in range(config.n_particles):
        type_i = config.particle_info[i]
        charges[i] = config.particle_types[type_i][2]

    m = (1/l_box) * _grid_points_without_center(rec_reso, rec_reso, rec_reso)
    m_modul_sq = np.linalg.norm(m, axis = 1) ** 2
    coeff_S = np.exp(-(math.pi / alpha) ** 2 * m_modul_sq) / m_modul_sq
    v_rec_prefactor = prefactor * 0.5 / math.pi / (l_box[0] * l_box[1] * l_box[2])
    #f_rec_prefactor = -prefactor * charges / (l_box[0] * l_box[1] * l_box[2]) # j and 2pi parts come here
    f_rec_prefactor = -prefactor / (l_box[0] * l_box[1] * l_box[2]) # j and 2pi parts come here
    v_self = -alpha / math.sqrt(math.pi) * np.sum(charges**2)

    def pot_func(q):
        m_dot_q = np.matmul(q, np.transpose(m))
        middle0 = np.pi * 2. * m_dot_q
        S_m_cos_sum = charges.dot(np.cos(middle0))
        S_m_sin_sum = charges.dot(np.sin(middle0))
        S_m_modul_sq = np.square(S_m_cos_sum) + np.square(S_m_sin_sum)
        v_rec = v_rec_prefactor * np.sum(coeff_S * S_m_modul_sq)
        return v_rec + v_self

    def force_func(q):
        m_dot_q = np.matmul(q, np.transpose(m))
        middle0 = np.pi * 2. * m_dot_q
        S_m_cos_parts = charges[:, np.newaxis] * np.cos(middle0)
        S_m_sin_parts = charges[:, np.newaxis] * np.sin(middle0)
        S_m_cos_sum, S_m_sin_sum = S_m_cos_parts.sum(axis=0), S_m_sin_parts.sum(axis=0)
        dS_modul_sq = 2. * S_m_cos_sum * S_m_sin_parts - 2. * S_m_sin_sum * S_m_cos_parts
        f_rec = f_rec_prefactor[..., None] * (coeff_S * dS_modul_sq).dot(m)
        return -f_rec
    
    return pot_func, force_func

def _grid_points_without_center(nx, ny, nz):
    a, b, c = np.arange(-nx, nx+1), np.arange(-ny, ny+1), np.arange(-nz, nz+1)
    xx, yy, zz = np.meshgrid(a, b, c)
    X = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T
    return np.delete(X, X.shape[0] // 2, axis=0)
