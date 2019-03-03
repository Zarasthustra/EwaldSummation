import numpy as np
from numba import njit

def distances_not_periodic(q):
    return np.linalg.norm(q[:, None, :] - q[None, :, :], axis = 2)

def distances_periodic(q, l_box):
    # new implementation
    distance_vectors = q[:, None, :] - q[None, :, :]
    np.mod(distance_vectors, l_box, out=distance_vectors)
    mask = distance_vectors > np.divide(l_box, 2.)
    distance_vectors += mask * -l_box
    return np.linalg.norm(distance_vectors, axis = 2)

class RadDistFunc:
    """
    input simuconfig, resolution wich means bin size of radii, and p_kinds list
    with max 3 p_kinds, for example 300 Ar, 200, Na, and 200 Cl, would be
    [300,200,200]
    """
    def __init__(self, config, bin_res, p_kinds):
        self.n_steps = config.n_steps
        self.l_box = np.array(config.l_box)
        self.n_particles = config.n_particles
        self.p_kinds = p_kinds
        self.PBC = config.PBC
        # times rad_func get called
        self.n_dim = config.n_dim
        self.r_max = np.min(self.l_box) / 2
        print(self.r_max)
        if self.r_max >= 9:
            self.r_max = 9
        self.bin_res = bin_res
        self.bin_width = self.r_max / self.bin_res
        self.bins = self.bin_width  * np.arange(self.bin_res)    
        self.p_index = []
        for _ in range(len(self.p_kinds)):
            self.p_index = np.append(self.p_index, [_] * self.p_kinds[_]).astype(int)
        
    def calc_radial_dist(self, qs, sample_start, sample_rate):
        """
        returns array of radial distribution functions, between max 3 Particle Types
        g_r, g_r_0_0, g_r_1_1, g_r_2_2, g_r_0_1, g_r_0_2, g_r_1_2
        """
        count_samples = 0
        hist_array = np.zeros([6,self.bin_res-1])
        for step in range(sample_start,self.n_steps):
            if step % sample_rate == 0 and step >= sample_start:
                count_samples += 1
                q = qs[step]
                if self.PBC == True:
                    distances = distances_periodic(q, self.l_box)
                else:
                    distances = distances_not_periodic(q)
                dist_array = get_dist_array(distances, self.l_box, self.r_max, self.p_index)
                hist_array += self.hist_func(dist_array)
        return self.averaging(hist_array, count_samples)

    def hist_func(self, dist_array):
        hist = np.zeros([6,self.bin_res-1])
        for _ in range(6):
            if len(dist_array[_]) != 0:
                hist[_], bins = np.histogram(dist_array[_], self.bins)
        return hist
    
    def averaging(self, hist_array, count_samples):
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
        norm_list = [self.p_kinds[0]**2,self.p_kinds[1]**2,self.p_kinds[2]**2, 2*self.p_kinds[0]*self.p_kinds[1]
                     ,2*self.p_kinds[0]*self.p_kinds[2],2*self.p_kinds[1]*self.p_kinds[2],np.sum(self.p_kinds)**2]
        g_r = np.zeros([7,len(real_bins)])
        hist_array = np.append(hist_array, np.sum(hist_array,axis=0)).reshape(7,self.bin_res-1)
        print(hist_array)
        for _ in range(7):
            if np.sum(hist_array[_]) != 0:
                g_r[_] =  hist_array[_] * np.prod(self.l_box)/ norm_list[_] / delta_V / count_samples
        return g_r, real_bins


@njit
def get_dist_array(distances, l_box, r_max, p_index):
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
