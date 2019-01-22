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
def ewald_pot2(distances, distance_vectors, alpha=1.):

    M = resolution # reciprocal periodic radius

    v_real = v_rec = v_self = 0.
    # real part
    for i,j in combinations(np.arange(n_particles),2):
        real_part = erfc(alpha * distances[i,j]) / distances[i,j]
        v_real += charge_vector[i] * charge_vector[j] * real_part
    
    # --- only useful for cases that minimum image convention is not enough
    # e.g. when alpha is too big
    REAL_PERIODIC_REPEATS = 3
    ks = np.array(list(product(range(-REAL_PERIODIC_REPEATS, REAL_PERIODIC_REPEATS+1), repeat=3)))
    ks = l_box * np.delete(ks, ks.shape[0] // 2, 0)
    for k in ks:
        for i in range(n_particles):
            for j in range(n_particles):
                distance = np.linalg.norm(distance_vectors[i, j] + k)
                v_real += 0.5 * charge_vector[i] * charge_vector[j] * erfc(alpha * distance) / distance
    # ---
    # reciprocal part
    m = np.array(list(product(range(-M, M+1), repeat=3)))
    m = (1/l_box) * np.delete(m, m.shape[0] // 2, 0)
    m_dot_q = np.matmul(q, np.transpose(m))
    ## structure factor
    S_m = charge_vector.dot(np.exp(np.pi * 2.j * m_dot_q))
    S_m_modul_sq = np.real(np.conjugate(S_m) * S_m)
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
    f_real = f_rec = np.zeros(n_dim)
    # real part
    for j in np.arange(n_particles):
        if(j != i):
            real_part = erfc(alpha * distances[i,j]) / distances[i,j] + \
                2 * alpha / np.sqrt(np.pi) * np.exp(-((alpha * distances[i,j]) ** 2))
            f_real += charge_vector[j] * (real_part / (distances[i,j]) ** 2) * distance_vectors[i, j]
    # --- only useful for cases that minimum image convention is not enough
    # e.g. when alpha is too big
    REAL_PERIODIC_REPEATS = 3
    ks = np.array(list(product(range(-REAL_PERIODIC_REPEATS, REAL_PERIODIC_REPEATS+1), repeat=3)))
    ks = l_box * np.delete(ks, ks.shape[0] // 2, 0)
    for k in ks:
        for j in range(n_particles):
            distance_vector = distance_vectors[i, j] + k
            distance = np.linalg.norm(distance_vectors[i, j] + k)
            real_part = erfc(alpha * distance) / distance + \
                2 * alpha / np.sqrt(np.pi) * np.exp(-((alpha * distance) ** 2))
            f_real += charge_vector[j] * (real_part / distance ** 2) * distance_vector
    # ---
    f_real *= charge_vector[i]
    # reciprocal part
    m = np.array(list(product(range(-M, M+1), repeat=3)))
    m = (1/l_box) * np.delete(m, m.shape[0] // 2, 0)
    m_dot_q = np.matmul(q, np.transpose(m))
    ## structure factor
    S_m = charge_vector.dot(np.exp(np.pi * 2.j * m_dot_q))
    dS_star_m = np.exp(-np.pi * 2.j * m_dot_q[i]) # *2pi*j*q_i
    #S_m_modul_sq = np.real(np.conjugate(S_m) * S_m)
    m_modul_sq = np.linalg.norm(m, axis = 1) ** 2
    coeff_S = np.exp(-(np.pi / alpha) ** 2 * m_modul_sq) / m_modul_sq
    f_rec = -charge_vector[i] / (l_box ** 3)
    f_rec *= 2 * (coeff_S * np.imag(S_m * dS_star_m)).dot(m)
    # print('f_real', f_real)
    # print('f_rec', f_rec)
    return f_real + f_rec

if __name__ == "__main__":
    # test salt
    n_particles = 8
    n_dim = 3
    resolution = 6
    l_box=2.
    grid=np.array( list (product(range(0,2), repeat=3)))
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
        forces[i] = ewald_force(d, dvec, 1)[0]
    xs_prime = np.arange(1.005, 2.005, 0.01)
    force_inte[0] = pots[0] - 0.005 * forces[0]
    for i in range(1, 100):
        force_inte[i] = force_inte[i - 1] - forces[i] * 0.01
    plt.plot(xs, pots, label='ewald_pot')
    plt.plot(xs, forces, label='ewald_force')
    plt.plot(xs_prime, force_inte, label='integrated_pot')
    plt.legend()
    plt.grid()
    plt.show()
