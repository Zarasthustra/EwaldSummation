# not finished version of the coulomb forces. uploadet to give a overview on the progress

# calculate the force on a particle i inspired by Allen-Tildesley 2017 p. 498
# next steps: bugfix, cutoffs , make it compatibel with the program, optimization
import numpy as np
from distances import get_distances
from particle_kind import particle_kind
from get_charges import get_charges
from scipy.special import erfc
from itertools import product


##########################global variables ############################# 
N_particles = 1000                                                        
mole_fraction = 0.5                                                     
dimension = 3                                                          
charge_a = 1                                                          
charge_b = -1                                                          
density = 0.8 
particle_kind_vector = particle_kind(N_particles, mole_fraction)        
charge_vector = get_charges(particle_kind_vector, charge_a, charge_b)   
box_length = (N_particles/density)**(1/dimension)   
x = np.random.rand(N_particles,dimension)*box_length
########################################################################


def real_space_coulomb_force(variance):
    prefactor = 2 * variance / (np.sqrt(np.pi))
    force = np.zeros([N_particles,dimension])
    some_value = 4 * np.pi / 3 * box_length ** (1/dimension) * np.sum(charge_vector*x) # possible other definition for n neq 3
    for i in range(N_particles):
        dist_and_r_ij = get_distances(x,i)
        r_ij = dist_and_r_ij[:,1:]
        dist = dist_and_r_ij[:,0]
        charge = charge_vector[i]
        charges = np.delete(charge_vector, i)
        array = 1 / (dist**3) * charges * ( prefactor * dist * np.exp(-variance*dist**2) + erfc(variance * dist))  # possible other definition for n neq 3
        array = r_ij * array[:, np.newaxis]
        force[i] = charge  * np.sum(array, axis=0) - charge * some_value
    return force


def reciprocal_space_coulomb_force(resolution, variance):
    k = 2 * np.pi / box_length * np.array( list (product(range(1,resolution+1), repeat=dimension)))
    inv_V = box_length**(1./dimension)
    k_dot_x = np.sum (np.matmul(x,np.transpose(k)), axis=0)
    reciprocal_density_distribution = np.exp(1j*k_dot_x)
    k_sq = np.linalg.norm(k, axis = 1) ** 2
    reciprokal_gaus = 4 * np.pi * np.exp(-k_sq / (4 * variance**(2)))/ k_sq
    reciprocal_force = np.zeros([N_particles,dimension])
    for i in range(N_particles):
        reciprocal_force[i] = - inv_V * charge_vector[i] * np.sum(reciprokal_gaus[:, np.newaxis] * np.imag(np.exp(1j * x[i] * k) * reciprocal_density_distribution[:, np.newaxis]),axis=0)  
    return reciprocal_force
