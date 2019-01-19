import numpy as np
from scipy.special import erfc
from itertools import product
from itertools import permutations
from itertools import combinations
from numba import jit





n_particles = 2
n_dim = 3
vac_per = 1 / 4 / np.pi
#l_box =  l_box[0] #assume that box is quadradic
#PBC = PBC
charge_a = 1
charge_b = -1 
mole_fraction = 0.5
resolution = 10
n_box=20
step_n = 25
charge_vector_a = np.asarray([charge_a] * round(n_particles *mole_fraction))
charge_vector_b = np.asarray([charge_b] * (n_particles- round(n_particles * mole_fraction)))
charge_vector = np.append(charge_vector_a,charge_vector_b)
density = 0.8
l_box=(n_particles/density)**(1./3.)
q=np.random.rand(n_particles,3)*l_box

real_cutoff = l_box / 2
gauss_scaling = 6 / l_box


def test():
    ewald_pot = ewald_pot2(*distances())
    stp_pot = stupid_coulomb_potential(*distances_not_PBC(),n_box)   
    return ewald_pot, stp_pot
    




#print(q)



def distances():
    # new implementation
    distance_vectors = q[:, None, :] - q[None, :, :]
    np.mod(distance_vectors, l_box, out=distance_vectors)
    mask = distance_vectors > np.divide(l_box, 2.)
    distance_vectors += mask * - l_box
    distances = np.linalg.norm(distance_vectors, axis=2)
    return distances, distance_vectors


def distances_not_PBC():
    # new implementation
    distance_vectors = q[:, None, :] - q[None, :, :]
    distances = np.linalg.norm(distance_vectors, axis=2)
    return distances, distance_vectors

def ewald_pot2(distances, distance_vectors):
    # ref: https://pdfs.semanticscholar.org/9b54/fdc67c4dac6dab1044c305f40b6beb86d43e.pdf

    M = 10 # reciprocal periodic radius
    alpha = .5
    
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
    #print('v_real', v_real)
    #print('v_rec', v_rec)
    #print('v_self', v_self)
    return v_real + v_rec + v_self

@jit
def coulomb_potential(distances, distance_vectors):
    k = 2 * np.pi / l_box * np.array( list (product(range(-resolution,resolution+1), repeat=3)))
    # input are the particle positions np array with shape n_particles, n_dim
    k = np.delete(k, k.shape[0] // 2,0)
    # removes the zerovector from k
    k_dot_q = np.matmul(q,np.transpose(k))
    #print(k_dot_q, k_dot_q.shape)
    reciprocal_density_distribution = np.sum(charge_vector[:, np.newaxis] * np.exp(-1j*k_dot_q), axis=0)
    #print(reciprocal_density_distribution,reciprocal_density_distribution.shape)
    sq_absolute_of_rec_dens_func = np.absolute(reciprocal_density_distribution) ** 2   
    k_sq = np.linalg.norm(k, axis = 1) ** 2
    reciprokal_gaus = 2 * np.pi * np.exp(-k_sq / (4 * gauss_scaling**(2))) / k_sq
    # 4 is exchanged to 2 because of the factor 1/2 infront of the sum for all k vectors
    rec_part = 1 * l_box**(-3) * np.sum(reciprokal_gaus * sq_absolute_of_rec_dens_func)

    self_interaction = gauss_scaling / np.sqrt(np.pi) * np.sum(charge_vector**2)
    pot=0
    for i,j in combinations(np.arange(n_particles),2):
        real_part = erfc(gauss_scaling * distances[i,j]) / distances[i,j]
        pot += charge_vector[i] * charge_vector[j] * real_part
    return pot+rec_part

@jit
def stupid_coulomb_potential(distances,distance_vectors,n_box):
   pot, pot_inside_box = 0,0
   n = np.array( list (product(range(-n_box,n_box+1), repeat=3)))
   n = np.delete(n, n.shape[0] // 2,0)
   # n ist the lattice vector for all boxes n_box means the radius in wich boxes ar scanned
   for i,j in combinations(np.arange(n_particles),2):
       pot_inside_box += charge_vector[i]*charge_vector[j] / np.linalg.norm(distance_vectors[i,j])
   for actual_n in n:
       for i,j in product(np.arange(n_particles),repeat=2):
           pot += charge_vector[i]*charge_vector[j] / np.linalg.norm(distance_vectors[i,j] + l_box * actual_n)
   return 0.5*pot + pot_inside_box
    
#print(coulomb_potential(*distances_not_PBC()))
#print(stupid_coulomb_potential(*distances_not_PBC()))  
print('Note: theoretically stupid sum cannot give a real convergent result in this case, no matter how large n_box is.')
print('results for ewald sum and stupid sum:',test())
