import numpy as np
from scipy.special import erfc
from itertools import product
from itertools import combinations
from itertools import combinations_with_replacement





n_particles = 100
n_dim = 3
vac_per = 1 / 4 / np.pi
#l_box =  l_box[0] #assume that box is quadradic
#PBC = PBC
charge_a = 1
charge_b = -1 
mole_fraction = 0.5
resolution = 15
n_box=10

charge_vector_a = np.asarray([charge_a] * round(n_particles *mole_fraction))
charge_vector_b = np.asarray([charge_b] * (n_particles- round(n_particles * mole_fraction)))
charge_vector = np.append(charge_vector_a,charge_vector_b)



q=np.random.rand(n_particles,3)*n_particles
l_box=10
real_cutoff = l_box / 2
gauss_scaling = 6 / l_box



#print(q)
test_size = 1000


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




def coulomb_potential(distances, distance_vectors):
    k = 2 * np.pi / l_box * np.array( list (product(range(-resolution,resolution+1), repeat=n_dim)))
    # input are the particle positions np array with shape n_particles, n_dim
    k = np.delete(k, k.shape[0] // 2,0)
    # removes the zerovector from k
    k_dot_q = np.matmul(q,np.transpose(k))
    #print(k_dot_q, k_dot_q.shape)
    reciprocal_density_distribution = np.sum(charge_vector[:, np.newaxis] * np.exp(-1j*k_dot_q), axis=0)
    #print(reciprocal_density_distribution,reciprocal_density_distribution.shape)
    sq_absolute_of_rec_dens_func = np.absolute(reciprocal_density_distribution) ** 2   
    print(sq_absolute_of_rec_dens_func,sq_absolute_of_rec_dens_func.shape)
    k_sq = np.linalg.norm(k, axis = 1) ** 2
    reciprokal_gaus = 2 * np.pi * np.exp(-k_sq / (4 * gauss_scaling**(2))) / k_sq
    # 4 is exchanged to 2 because of the factor 1/2 infront of the sum for all k vectors
    rec_part = 1 * l_box**(-3) * np.sum(reciprokal_gaus * sq_absolute_of_rec_dens_func)

    self_interaction = gauss_scaling / np.sqrt(np.pi) * np.sum(charge_vector**2)
    pot=0
    for i,j in combinations(np.arange(n_particles),2):
        real_part = erfc(gauss_scaling * distances[i,j]) /distances[i,j]
        pot += charge_vector[i] * charge_vector[j] * real_part
    return pot+rec_part


def stupid_coulomb_potential(distances,distance_vectors):
    short_range_pot, long_range_pot = 0,0
    for i,j in combinations(np.arange(n_particles),2):
        short_range_pot += charge_vector[i]*charge_vector[j] / distances[i,j]
    for box in range(1,n_box):
        for i,j in combinations_with_replacement(np.arange(n_particles),2):
            long_range_pot += charge_vector[i]*charge_vector[j] / np.linalg.norm(distance_vectors[i,j] + box*l_box)
    pot = short_range_pot + long_range_pot        
    return pot
    
print(coulomb_potential(*distances_not_PBC()))
print(stupid_coulomb_potential(*distances_not_PBC()))  

