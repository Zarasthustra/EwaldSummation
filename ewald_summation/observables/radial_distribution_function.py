import numpy as np
from itertools import product
from numba import jit

class RadialDistributionFunction:
    def __init__(self, config, qs):
        self.qs = qs
        self.l_box = np.array(config.l_box)
        self.n_particles = config.n_particles
        self.n_dim = config.n_dim
    def g_r(self):
        """
        writes a .xyz format trajectory
        qs = coordinates 
        component_a = string for example "Na"
        component_a = string for example "Cl"
        """
        g_r, bins = radial_dist_func(self.qs, self.n_dim, self.n_particles, self.l_box)
        return g_r,bins

def distance_vectors_periodic(x, l_box):
    # new implementation
    distance_vectors = x[:, None, :] - x[None, :, :]
    np.mod(distance_vectors, l_box, out=distance_vectors)
    mask = distance_vectors > np.divide(l_box, 2.)
    distance_vectors += mask * -l_box
    return distance_vectors

@jit(parallel=True)
def radial_dist_func(qs,n_dim,n_particles,l_box):
    density = n_particles/np.prod(l_box)
    bin_res = 50
    r_max = np.min(l_box)/2
    bin_width = r_max/bin_res
    hist,g_r,delta_V = np.zeros(bin_res-1), np.zeros(bin_res-1), np.zeros(bin_res)
    bins = r_max / bin_res * np.arange(bin_res)
    tril_vector = np.tril_indices(n_particles,-1)

    for _ in range(int(0.25*qs.shape[0]),qs.shape[0]):
        dist= np.linalg.norm(distance_vectors_periodic(qs[_],l_box), axis=2)[tril_vector]
        hist,trash = np.histogram(dist, bins)
        g_r +=  hist
    if n_dim==3:
        delta_V = 2./3. * np.pi * ((bins+bin_width)**3. -bins**3)
    if n_dim==2:
        delta_V = 0.5 * np.pi * ((bins+bin_width)**2. -bins**2)
    if n_dim==1:
        delta_V = 0.5 * bin_widht
    delta_V = np.delete(delta_V,-1)
    bins = np.delete(bins,-1) + 0.5 * bin_width
    return g_r / density / n_particles / delta_V / (qs.shape[0]-int(0.25*qs.shape[0])), bins
