import numpy as np
from itertools import product

class RadialDistributionFunction:
    def __init__(self, config):
    	self.neighbour = config.neighbour
    	self.n_steps = config.n_steps
        self.l_box = np.array(config.l_box)
        self.n_particles = config.n_particles
        self.n_dim = config.n_dim
        self.r_max = np.min(l_box) / 2
        self.bin_res = 50
        self.bin_width = self.r_max / self.bin_res
        self.density = self.n_particles / np.prod(self.l_box)
        self.func_called = 0
        self.hist, self.g_r, self.delta_V = np.zeros(bin_res-1), \
                                              np.zeros(bin_res-1), \
                                              np.zeros(bin_res)
        self.tril_vector = np.tril_indices(self.n_particles,-1)
        self.bins = self.r_max / self.bin_res * np.arange(self.bin_res)
    
    def calc_radial_dist(self, current_frame)
    	if self.neighbour:
    		output = rad_dist_func_neighbour(current_frame.distances,
                                          	 current_frame.array_index)
		else:
		    output = rad_dist_func_not_neighbour(current_frame.distances)
		return output

	def rad_dist_func_not_neighbour(self, distance_vectors):
	    dist = distances[self.tril_vector]
	    self.hist, trash = np.histogram(dist, self.bins)
	    g_r +=  hist
	    func_called += 1
	    if func_called = n_step
	    #checks if its the last time
			if n_dim==3:
	    	    self.delta_V = 2./3. * np.pi * ((self.bins+self.bin_width)**3. -self.bins**3)
	    	if n_dim==2:
	    	    self.delta_V = 0.5 * np.pi * ((self.bins+self.bin_width)**2. -self.bins**2)
	    	if n_dim==1:
	    	    self.delta_V = 0.5 * bin_widht
	    	delta_V = np.delete(delta_V,-1)
	    	real_bins = np.delete(self.bins,-1) + 0.5 * self.bin_width
	    	return g_r / self.density / self.n_particles / self.delta_V / func_called, real_bins
