import numpy as np


def particle_kind(N_particles, mole_fraction):
    """ gives back a vector of the particle kind
    wich is zero forall particle kind a and 1 for the other particle kind b
    there could be a smarter solution but i have to work with
    something.
    
    arguments:
    1) N_particles = total number of particles
    2) mole_fraction = N particles of kind a divided by N_particles
    
    """
    N_particles_a = int(N_particles * mole_fraction)
    particle_kind_vector = np.zeros([N_particles])
    particle_kind_vector[N_particles_a:] = 1
    return particle_kind_vector
  