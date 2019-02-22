import numpy as np
from numba import njit
from itertools import combinations_with_replacement   
class RadialDistributionFunction:
    
    store_hist = np.zeros([6,20])
    
    #store data during simulation
    
    def __init__(self, config):
        self.neighbour = config.neighbour
        self.n_steps = config.n_steps
        self.l_box = np.array(config.l_box)
        self.n_particles = config.n_particles
        self.p_kinds = config.p_kinds
        self.density = self.n_particles / np.prod(self.l_box)
        self.start_sampling = config.start_sampling
        self.sampling_rate= config.sampling_rate
        self.norm_factor = divmod(self.n_steps - self.start_sampling, self.sampling_rate)[0]
        # times rad_func get called
        self.last_call = self.n_steps - (self.n_steps - 1) % self.sampling_rate - 1
        self.n_dim = config.n_dim
        self.r_max = np.min(self.l_box) / 2
        if self.r_max >= 7:
           self.r_max = 7
        self.bin_res = 21
        self.bin_width = self.r_max / self.bin_res
        self.bins = self.bin_width  * np.arange(self.bin_res)    
        self.p_index = []
        for _ in range(len(self.p_kinds)):
            self.p_index = np.append(self.p_index, [_] * self.p_kinds[_]).astype(int)
        
    def calc_radial_dist(self, current_frame, step):
       """
       calls rad_dist_func depending on, neighbour == True
       or not, also checks when to start averaging
       """
       print(step)
       if step == self.last_call:
           self.averaging()
       if self.neighbour:
           dist = rad_dist_func_neighbour(current_frame.distances, self.r_max, self.p_index, current_frame.array_index)
       else:
           dist = rad_dist_func_not_neighbour(current_frame.distances, self.r_max, self.p_index)
       self.store_hist += self.hist_func(dist)
            
    def averaging(self):
        """ 
        averages the sampled data, and norms to density of an ideal gas
        """
        delta_V = np.zeros(self.bin_res-1)
        if self.n_dim==3:
            self.delta_V = 2./3. * np.pi * ((self.bins+self.bin_width)**3. - self.bins**3)
        if self.n_dim==2:
            self.delta_V = 0.5 * np.pi * ((self.bins+self.bin_width)**2. - self.bins**2)
        if self.n_dim==1:
            self.delta_V = [0.5*(self.bin_width)] * 2
        delta_V = np.delete(self.delta_V,-1)
        real_bins = np.delete(self.bins,-1) + 0.5 * self.bin_width
        name_list = ["g_r_0_0.dat", "g_r_1_1.dat", "g_r_2_2.dat", "g_r_0_1.dat", "g_r_0_2.dat", "g_r_1_2.dat"]
        norm_list = [self.p_kinds[0]*self.p_kinds[0],self.p_kinds[1]*self.p_kinds[1],self.p_kinds[2]*self.p_kinds[2],
                     2*self.p_kinds[0]*self.p_kinds[1],2*self.p_kinds[0]*self.p_kinds[2],2*self.p_kinds[1]*self.p_kinds[2]]
        for _ in range(6):
            if np.sum(self.store_hist[_]) == 0:
                g_r = np.zeros(69)
            else:
                g_r = self.store_hist[_] / self.density * self.n_particles / norm_list[_] / delta_V / self.norm_factor
                data = np.concatenate([[g_r],[real_bins]])
                np.savetxt(name_list[_], np.transpose(data), delimiter=" ")

    def hist_func(self, dist):
        hist = np.zeros([6,self.bin_res-1])
        for _ in range(6):
            hist[_], trash = np.histogram(dist[_], self.bins)
        return hist

@njit
def rad_dist_func_not_neighbour(distances, r_max, p_index):
    """ 
    computes histogramm of distances, without neighbourlists
    """
    dist_0_0 = []
    dist_1_1 = []
    dist_2_2 = []
    dist_0_1 = []
    dist_0_2 = []
    dist_1_2 = []
    for i in range(len(distances)):
        for j in range(i+1,len(distances)):
            actual_dist = distances[i,j]
            if actual_dist <= r_max:
                if p_index[i] == 0 and p_index[j] == 0:
                    dist_0_0.append(actual_dist)
                if p_index[i] == 1 and p_index[j] == 1:
                    dist_1_1.append(actual_dist)
                if p_index[i] == 2 and p_index[j] == 2: 
                    dist_2_2.append(actual_dist)
                if p_index[i] == 0 and p_index[j] == 1 or p_index[i] == 1 and p_index[j] == 0:  
                    dist_0_1.append(actual_dist)       
                if p_index[i] == 0 and p_index[j] == 2 or p_index[i] == 2 and p_index[j] == 0:  
                    dist_0_2.append(actual_dist)       
                if p_index[i] == 2 and p_index[j] == 1 or p_index[i] == 1 and p_index[j] == 2:
                    dist_1_2.append(actual_dist)
    return [dist_0_0, dist_1_1, dist_2_2, dist_0_1, dist_0_2, dist_1_2]
    
@njit
def rad_dist_func_neighbour(distances, r_max, p_index, array_index):
    """ 
    computes histogramm of distances, without neighbourlists
    """
    dist_0_0 = []
    dist_1_1 = []
    dist_2_2 = []
    dist_0_1 = []
    dist_0_2 = []
    dist_1_2 = []
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            actual_dist = distances[i,j]
            #print(p_index[i],p_index[array_index[i,j]],actual_dist)
            if actual_dist <= r_max and actual_dist > 0:
                if p_index[i] == 0 and p_index[array_index[i,j]] == 0:
                    dist_0_0.append(actual_dist)
                if p_index[i] == 1 and p_index[array_index[i,j]] == 1:
                    dist_1_1.append(actual_dist)
                if p_index[i] == 2 and p_index[array_index[i,j]] == 2: 
                    dist_2_2.append(actual_dist)
                if p_index[i] == 0 and p_index[array_index[i,j]] == 1 or p_index[i] == 1 and p_index[array_index[i,j]] == 0:
                    dist_0_1.append(actual_dist)       
                if p_index[i] == 0 and p_index[array_index[i,j]] == 2 or p_index[i] == 2 and p_index[array_index[i,j]] == 0:
                    dist_0_2.append(actual_dist)       
                if p_index[i] == 2 and p_index[array_index[i,j]] == 1 or p_index[i] == 1 and p_index[array_index[i,j]] == 2:
                    dist_1_2.append(actual_dist)
    return [dist_0_0, dist_1_1, dist_2_2, dist_0_1, dist_0_2, dist_1_2]
