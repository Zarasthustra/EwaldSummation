import numpy as np
from scipy.special import erfc
from itertools import product

class CoulombForce:
    def __init__(self, simuconfig):
        self.n_dim = simuconfig.n_dim
        self.vac_per = simuconfig.vac_per
        self.l_box =  simuconfig.l_box[0] #assume that box is quadradic
        self.coulomb_trunc_err = simuconfig.coulomb_trunc_err
        self.PBC = simuconfig.PBC

    real_cutoff = l_box / 2
    p = -np.log(coulomb_trunc_err)
    variance = real_cutoff/np.sqrt(2 * p)
    k_space_cutoff = 2*p/coulomb_trunc_err
    if PBC:
        def real_space_coulomb_force(self, x):
            prefactor = 2 * variance / (np.sqrt(np.pi))
            force = np.zeros([N_particles,dimension])
            self_interaction_therm = 4 * np.pi / 3 * box_length ** (1/dimension) * np.sum(charge_vector*x, axis=0) # possible other definition for dim neq 3
            for i in range(N_particles):
                dist_and_r_ij = get_distances(x,i)
                r_ij = dist_and_r_ij[:,1:]
                dist = dist_and_r_ij[:,0]
                charge = charge_vector[i]
                charges = np.delete(charge_vector, i)
                array = 1 / (dist**3) * charges * ( prefactor * dist * np.exp(-variance*dist**2) + erfc(variance * dist))  # possible other definition for dim neq 3
                array = r_ij * array[:, np.newaxis]
                force[i] = charge  * np.sum(array, axis=0) - charge * self_interaction_therm
            return force
        def reciprocal_space_coulomb_force(resolution, variance):
            k = 2 * np.pi / box_length * np.array( list (product(range(-resolution,resolution+1), repeat=dimension)))
            k = np.delete(k, k.shape[0]//2,0)
            # removes k= zerovector
            inv_V = box_length**(1/dimension)
            k_dot_x = np.matmul(x,np.transpose(k))
            reciprocal_density_distribution = np.sum(charge_vector * np.exp(1j*k_dot_x), axis=0)
            k_sq = np.linalg.norm(k, axis = 1) ** 2
            reciprokal_gaus = 4 * np.pi * np.exp(-k_sq / (4 * variance**(2)))/ k_sq
            reciprocal_force = np.zeros([N_particles,dimension])
            for i in range(N_particles):
                reciprocal_force[i] = - inv_V * charge_vector[i] * np.sum(k * reciprokal_gaus[:, np.newaxis] * np.imag(np.exp(1j * x[i] * k) * reciprocal_density_distribution[:, np.newaxis]),axis=0)
        return reciprocal_force
        
    