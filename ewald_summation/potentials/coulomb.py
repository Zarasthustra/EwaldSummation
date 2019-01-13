import numpy as np
from scipy.special import erfc
from itertools import product

class CoulombForce:
    def __init__(self, config):
        self.n_particles = config.n_particles
        self.n_dim = config.n_dim
        self.vac_per = config.vac_per
        self.l_box =  config.l_box[0] #assume that box is quadradic
        self.PBC = config.PBC
        self.charge_a = config.charge_a
        self.charge_b = config.charge_b 
        self.mole_fraction = config.mole_fraction
        
        charge_vector_a = [self.charge_a] * round(self.n_particles * self.mole_fraction)
        charge_vector_b = [self.charge_b] * (self.n_particles- round(self.n_particles * self.mole_fraction))
        self.charge_vector = np.concatenate([charge_vector_a,charge_vector_b])
        real_cutoff = self.l_box / 2
        gauss_scaling = 6 / self.l_box
        

    def real_space_coulomb_force(self, distance_vector):
        prefactor = 2 * self.gauss_scaling / (np.sqrt(np.pi))
        force = np.zeros([self.n_particles,self.n_dim])
        surrounding_of_boxes = 4 * np.pi / 3 * self.l_box ** (1/self.n_dim) \
                               * np.sum(self.charge_vector * distance_vector, axis=0)
        # sourrounding of the sphere of periodical boxes see 
        # Simulation of electrostatic systems in periodic boundary conditions 
        # I. Lattice sums and dielectric constants, de Leeuw et al., 1980.
        distances = np.linalg.norm(distances, axis=2)          
        for i in range(self.n_particles):
            indices_to_delete = np.append(np.argwhere(distances[i]>real_cutoff),i)
            r_ij = np.delete(distance_vector[i], indices_to_delete, axis=0)
            dist = np.delete(distances[i], indices_to_delete)
            charge = self.charge_vector[i]
            charges = np.delete(self.charge_vector[i], indices_to_deletew)
            force_ = 1 / (dist**3) * charges * ( prefactor * dist * np.exp(-gauss_scaling*dist**2) + erfc(gauss_scaling * dist))  # possible other definition for dim neq 3
            force_ = r_ij * array[:, np.newaxis]
            force[i] = charge  * np.sum(array, axis=0) - charge * self_interaction_therm
        return force
    def reciprocal_space_coulomb_force(self, x):
        # input are the particle positions np array with shape n_particles, n_dim
        k = 2 * np.pi / self.l_box * np.array( list (product(range(-resolution,resolution+1), repeat=self.n_dim)))
        k = np.delete(k, k.shape[0]//2,0)
        # removes the zerovector from k
        inv_V = self.l_box**(1/self.n_dim)
        k_dot_x = np.matmul(x,np.transpose(k))
        reciprocal_density_distribution = np.sum(self.charge_vector * np.exp(1j*k_dot_x), axis=0)
        k_sq = np.linalg.norm(k, axis = 1) ** 2
        reciprokal_gaus = 4 * np.pi * np.exp(-k_sq / (4 * gauss_scaling**(2)))/ k_sq
        reciprocal_force = np.zeros([self.n_particles,self.n_dim])
        for i in range(self.n_particles):
            reciprocal_force[i] = - inv_V * self.charge_vector[i] * np.sum(k * reciprokal_gaus[:, np.newaxis] * np.imag(np.exp(1j * x[i] * k) * reciprocal_density_distribution[:, np.newaxis]),axis=0)
        return reciprocal_force
    def force(q,distance_vector):
        force = reciprocal_space_coulomb_force(self,q) + real_space_coulomb_force(self,distance_vector)
        return force
        
   