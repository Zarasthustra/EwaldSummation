import numpy as np

def get_distances(x, i, l):
    """ get all distances of particle i and the postion vectors for
        the corresponding particles.

        Arguments:
            x (np.ndarray(n_particels, dim)): configuration
            i (int): particle for distance calc
            l (list(float)): length of box

        Output (np.ndarray(n_particels - 1, dim + 1)): distances and
            corresponding vecotors for ith particle """

    # init relevant arrays
    particle = x[i,:]
    other_particles = np.delete(x, i, 0)
    box_lengthes = np.zeros(other_particles.shape)
    output = np.zeros((other_particles.shape[0], 4))

    # project every particle into the periodic box
    for i in range(other_particles.shape[1]):
        length = np.multiply(np.ones(other_particles.shape[0]), l[i])
        box_lengthes[:,i] = length
    other_particles = np.mod(other_particles, box_lengthes)

    # calculate distances
    for j in range(other_particles.shape[0]):
        output[j,1:] = other_particles[j,:]
        output[j,0]  = np.linalg.norm(other_particles[j,:])

    return other_particles
