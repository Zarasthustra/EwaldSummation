import numpy as np
from itertools import product

class RadialDistributionFunction:
    def __init__(self, config):
        self.neighbour = config.neighbour
        self.n_steps = config.n_steps
        self.l_box = np.array(config.l_box)
        self.n_particles = config.n_particles
        self.density = self.n_particles / np.prod(self.l_box)
        self.start_sampling = config.start_sampling
        self.sampling_rate= config.sampling_rate
        self.norm_factor = (self.n_steps - self.start_sampling) / self.sampling_rate
        self.last_call = self.n_steps - self.start_sampling
        self.n_dim = config.n_dim
        self.r_max = np.min(self.l_box) / 2
        self.bin_res = 70
        self.bin_width = self.r_max / self.bin_res
        
        self.tril_vector = np.tril_indices(self.n_particles,-1)
        self.bins = self.r_max / self.bin_res * np.arange(self.bin_res)
    
    def calc_radial_dist(self, current_frame, step):
        if self.neighbour:
        	self.rad_dist_func_neighbour(current_frame.distances, \
                                         current_frame.array_index)
        else:
            self.rad_dist_func_not_neighbour(current_frame.distances, step)


    def rad_dist_func_not_neighbour(self, distances, step):
        hist = np.zeros(self.bin_res-1)
        dist = distances[self.tril_vector]
        hist, trash = np.histogram(dist, self.bins)
        if step  == self.start_sampling + self.start_sampling % self.sampling_rate:
            print('initalize g_r')
            np.save('g_r.npy', hist)
        hist = hist + np.load('g_r.npy')
        np.save('g_r.npy', hist)
        if step == self.last_call + self.start_sampling % self.sampling_rate: 
        #checks if its the last time
            delta_V = np.zeros(self.bin_res-1)
            if self.n_dim==3:
                self.delta_V = 2./3. * np.pi * ((self.bins+self.bin_width)**3. -self.bins**3)
            if self.n_dim==2:
                self.delta_V = 0.5 * np.pi * ((self.bins+self.bin_width)**2. -self.bins**2)
            if self.n_dim==1:
                self.delta_V = [0.5*(self.bin_width)] * 2
                print(self.delta_V)
            delta_V = np.delete(self.delta_V,-1)
            real_bins = np.delete(self.bins,-1) + 0.5 * self.bin_width
            g_r = np.load('g_r.npy') / self.density / self.n_particles / delta_V / self.norm_factor
            data = np.concatenate([[g_r],[real_bins]])
            print('save g_r')
            np.savetxt("g_r.dat", np.transpose(data), delimiter=" ")

    def rad_dist_func_neighbour(self, distances, array_index):
        print('g_r with neighbourlists is not implementet yet')
