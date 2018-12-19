# calculate the force on a particle i inspired by Allen-Tildesley 2017 p. 498
# long range force will be added, could contain bugs i only testet Newton 3. Law
import numpy as np
from distances import get_distances
from particle_kind import particle_kind
from get_charges import get_charges
from scipy.special import erfc

##########################global variables ########################## 
N_particles = 100
mole_fraction = 0.5
dimension = 3
charge_a = 1.
charge_b = -1.
particle_kind_vector = particle_kind(N_particles, mole_fraction)
charge_vector = get_charges(particle_kind_vector, charge_a, charge_b)
####################################################################

#real space coulomb force without cutoff
def real_space_coulomb_force(x, variance):
    prefactor = 2 * variance / (np.sqrt(np.pi))
    force = np.zeros([N_particles,dimension])
    for i in range(N_particles):
        dist_and_r_ij = get_distances(x,i)
        r_ij = dist_and_r_ij[:,1:]
        dist = dist_and_r_ij[:,0]
        charge = charge_vector[i]
        charges = np.delete(charge_vector, i)
        array = 1 / (dist**3) * charges * ( prefactor * dist * np.exp(-variance*dist**2) + erfc(variance * dist))
        array = r_ij * array[:, np.newaxis]
        force[i] = charge  * np.sum(array, axis=0)
    return force
