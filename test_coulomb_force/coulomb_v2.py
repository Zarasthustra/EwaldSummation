import numpy as np
from scipy.special import erfc
from itertools import product
from itertools import permutations
from itertools import combinations
from numba import jit
import matplotlib.pyplot as plt


def distances():
    # new implementation
    distance_vectors = q[:, None, :] - q[None, :, :]
    np.mod(distance_vectors, l_box, out=distance_vectors)
    mask = distance_vectors > np.divide(l_box, 2.)
    distance_vectors += mask * - l_box
    distances = np.linalg.norm(distance_vectors, axis=2)
    return distances, distance_vectors 

@jit
def _grid_points_without_center(n):
    a = np.arange(-n, n+1)
    xx, yy, zz = np.meshgrid(a, a, a)
    X = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T
    return np.delete(X, X.shape[0] // 2, axis=0)

@jit
def ewald_pot2(distances, distance_vectors, alpha=1.):

    M = resolution # reciprocal periodic radius
    
    v_real = v_rec = v_self = 0.
    charge_product_i_j = charge_vector[:, np.newaxis] * charge_vector[np.newaxis, :]
    # real part
    '''
    for i,j in combinations(np.arange(n_particles),2):
        real_part = erfc(alpha * distances[i,j]) / distances[i,j]
        v_real += charge_product_i_j[i, j] * real_part
    '''
    dist_masked = np.ma.masked_values(distances, 0.)
    v_real = 0.5 * np.sum(charge_product_i_j * erfc(alpha * dist_masked) / dist_masked)
    # --- only useful for cases that minimum image convention is not enough
    # e.g. when alpha is too big
    REAL_PERIODIC_REPEATS = int(np.ceil(8. / l_box)) - 1
    
    # ks = np.array(list(product(range(-REAL_PERIODIC_REPEATS, REAL_PERIODIC_REPEATS+1), repeat=3)))
    # ks = l_box * np.delete(ks, ks.shape[0] // 2, 0)
    ks = l_box * _grid_points_without_center(REAL_PERIODIC_REPEATS)
    for k in ks:
        image_distances = np.linalg.norm(distance_vectors + k, axis=-1)
        v_real += 0.5 * (charge_product_i_j * (erfc(alpha * image_distances) / image_distances)).sum()
        #for i in range(n_particles):
        #    for j in range(n_particles):
        #        distance = np.linalg.norm(distance_vectors[i, j] + k)
        #        v_real += 0.5 * charge_vector[i] * charge_vector[j] * erfc(alpha * distance) / distance
    # ---
    # reciprocal part
    # m = np.array(list(product(range(-M, M+1), repeat=3)))
    # m = (1/l_box) * np.delete(m, m.shape[0] // 2, 0)
    m = (1/l_box) * _grid_points_without_center(M)
    m_dot_q = np.matmul(q, np.transpose(m))
    ## structure factor
    middle0 = np.pi * 2. * m_dot_q
    #middle1 = np.pi * 2.j * m_dot_q
    #middle2 = np.exp(middle1)
    #middle3 = np.sin(middle0)
    #S_cos_part = np.cos(middle0)
    #S_sin_part = np.sin(middle0)
    S_m_cos_part = charge_vector.dot(np.cos(middle0))
    S_m_sin_part = charge_vector.dot(np.sin(middle0))
    S_m_modul_sq = np.square(S_m_cos_part) + np.square(S_m_sin_part)
    #S_m_modul_sq = np.real(np.conjugate(S_m) * S_m)
    m_modul_sq = np.linalg.norm(m, axis = 1) ** 2
    coeff_S = np.exp(-(np.pi / alpha) ** 2 * m_modul_sq) / m_modul_sq
    v_rec = 0.5 / np.pi / (l_box ** 3)
    v_rec *= np.sum(coeff_S * S_m_modul_sq)
    # self part
    v_self = -alpha / np.sqrt(np.pi) * np.sum(charge_vector**2)
    # print('v_real', v_real)
    # print('v_rec', v_rec)
    # print('v_self', v_self)
    return v_real + v_rec + v_self

@jit
def ewald_force(distances, distance_vectors, i, alpha = 1.):

    M = resolution # reciprocal periodic radius
    # f_real = f_rec = np.zeros(n_dim)
    f_real = f_rec = np.zeros((n_particles, n_dim))
    # real part
    # for j in np.arange(n_particles):
    #     if(j != i):
    #         real_part = erfc(alpha * distances[i,j]) / distances[i,j] + \
    #             2 * alpha / np.sqrt(np.pi) * np.exp(-((alpha * distances[i,j]) ** 2))
    #         f_real += charge_vector[j] * (real_part / (distances[i,j]) ** 2) * distance_vectors[i, j]
    dist_masked = np.ma.masked_values(distances, 0.)
    real_part = erfc(alpha * dist_masked) / dist_masked + \
                2 * alpha / np.sqrt(np.pi) * np.exp(-((alpha * dist_masked) ** 2))
    f_real += ((real_part / (dist_masked) ** 2 * charge_vector)[..., None] * distance_vectors).sum(axis=1)
    # --- only useful for cases that minimum image convention is not enough
    # e.g. when alpha is too big
    REAL_PERIODIC_REPEATS = int(np.ceil(8. / l_box)) - 1
    # ks = np.array(list(product(range(-REAL_PERIODIC_REPEATS, REAL_PERIODIC_REPEATS+1), repeat=3)))
    # ks = l_box * np.delete(ks, ks.shape[0] // 2, 0)
    ks = l_box * _grid_points_without_center(REAL_PERIODIC_REPEATS)
    for k in ks:
        for i in range(n_particles):
            for j in range(n_particles):
                distance_vector = distance_vectors[i, j] + k
                distance = np.linalg.norm(distance_vectors[i, j] + k)
                real_part = erfc(alpha * distance) / distance + \
                    2 * alpha / np.sqrt(np.pi) * np.exp(-((alpha * distance) ** 2))
                f_real[i] += charge_vector[j] * (real_part / distance ** 2) * distance_vector
    # ---
    f_real *= charge_vector[..., None]
    # reciprocal part
    # m = np.array(list(product(range(-M, M+1), repeat=3)))
    # m = (1/l_box) * np.delete(m, m.shape[0] // 2, 0)
    m = (1/l_box) * _grid_points_without_center(M)
    m_dot_q = np.matmul(q, np.transpose(m))
    ## structure factor
    # S_m = charge_vector.dot(np.exp(np.pi * 2.j * m_dot_q))
    # dS_star_ms = np.exp(-np.pi * 2.j * m_dot_q) # *2pi*j*q_i
    middle0 = np.pi * 2. * m_dot_q
    S_m_cos_parts = charge_vector[:, np.newaxis] * np.cos(middle0)
    S_m_sin_parts = charge_vector[:, np.newaxis] * np.sin(middle0)
    S_m_cos_part, S_m_sin_part = S_m_cos_parts.sum(axis=0), S_m_sin_parts.sum(axis=0)
    dS_modul_sq = 2. * S_m_cos_part * S_m_sin_parts - 2. * S_m_sin_part * S_m_cos_parts
    #S_m_modul_sq = np.real(np.conjugate(S_m) * S_m)
    m_modul_sq = np.linalg.norm(m, axis = 1) ** 2
    coeff_S = np.exp(-(np.pi / alpha) ** 2 * m_modul_sq) / m_modul_sq
    middle5 = -charge_vector / (l_box ** 3) # j and 2pi parts come here
    f_rec = middle5[..., None] * (coeff_S * dS_modul_sq).dot(m)
    # print('f_real', f_real)
    # print('f_rec', f_rec)
    return f_real - f_rec

if __name__ == "__main__":
    # test salt
    n_particles = 8
    n_dim = 3
    resolution = 6
    l_box=2.
    grid=np.array(list(product(range(0,2), repeat=3)))
    q = 1. * grid
    charge_vector = grid.sum(axis=1)%2*2-1
    print("Madelung constant for salt:", -ewald_pot2(*distances()) / 4)

    l_box = 3.
    n_particles = 2
    q = np.array([[0., 0., 0.], [1., 0., 0.]])
    charge_vector = np.array([-1., 1.])
    xs = np.arange(1., 2., 0.01)
    pots = np.zeros(100)
    forces = np.zeros(100)
    force_inte = np.zeros(100)
    for i in range(100):
        q[1, 0] = xs[i]
        d, dvec = distances()
        pots[i] = ewald_pot2(d, dvec)
        forces[i] = ewald_force(d, dvec, 1)[1, 0]
    xs_prime = np.arange(1.005, 2.005, 0.01)
    force_inte[0] = pots[0] - 0.005 * forces[0]
    for i in range(1, 100):
        force_inte[i] = force_inte[i - 1] - forces[i] * 0.01
    plt.figure(dpi=150)
    plt.plot(xs, pots, label='ewald_pot')
    plt.plot(xs, forces, label='ewald_force')
    plt.plot(xs_prime, force_inte, label='integrated_pot')
    plt.ylabel('Potential/Force')
    plt.xlabel('x coordinate of particle 1')
    plt.legend()
    plt.grid()
    plt.show()
