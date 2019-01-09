import numpy as np

def get_distances(x, i):
    """ get all distances of particle i and the postion vectors for
        the corresponding particles.

        Arguments:
            x (np.ndarray(n_particels, dim)): configuration
            i (int): particle for distance calc

        Output (np.ndarray(n_particels - 1, dim + 1)): distances and
            corresponding vecotors for ith particle """

    particle = x[i,:]
    x = x - particle
    other_particles = np.delete(x, i, 0)
    output = np.zeros((other_particles.shape[0], 4))
    for j in range(other_particles.shape[0]):
        output[j,1:] = other_particles[j,:]
        output[j,0]  = np.linalg.norm(other_particles[j,:])

    return output