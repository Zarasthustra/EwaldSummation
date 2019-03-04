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
    input simuconfig, resolution wich means bin size of radii
    """
    def __init__(self, config, bin_res):
        self.n_steps = config.n_steps
        self.l_box = np.array(config.l_box)
        self.n_particles = config.n_particles
        self.PBC = config.PBC
        # times rad_func get called
        self.n_dim = config.n_dim
        self.r_max = np.min(self.l_box) / 2
        print(self.r_max)
        #if self.r_max >= 9:
        #    self.r_max = 9
        self.bin_res = bin_res
        self.bin_width = self.r_max / self.bin_res
        self.bins = self.bin_width  * np.arange(self.bin_res)
        self.particle_info = config.particle_info
        #self.p_index = []
        #for _ in range(len(self.p_kinds)):
        #    self.p_index = np.append(self.p_index, [_] * self.p_kinds[_]).astype(int)
        
    def calc_radial_dist(self, qs, sample_start, sample_end ,sample_rate):
        """
        returns array of radial distribution functions, between all combinations with replacements
        of the included particle typs Ar, H2O, Na, Cl. For H2O oxigin is choosen as the relevant
        atom.
        
        input: coordinates, start_sampling (step), end sampling (step), sampling rate (step)
        
        output: np.array of g(r) from Ar-Ar, Ar-OH2, Ar-O, Ar-Na, Ar-Cl, H2O-OH2, H2O-Na, H2O-Cl, Na-Na
        Na-Cl, Cl-Cl and all Particles together, the radii and the name list wich can be used for plotting.
        
        Note that sample all of this funcs would need a huge system and a long sampling time, the not
        used p_kinds are giving an array of zeros
        """
        count_samples = 0
        hist_array = np.zeros([10,self.bin_res-1])
        for step in range(sample_start,sample_end):
            if step % sample_rate == 0:
                count_samples += 1
                q = qs[step] + self.l_box / 2
                if self.PBC == True:
                    distances = distances_periodic(q, self.l_box)
                else:
                    distances = distances_not_periodic(q)
                dist_array = get_dist_array(distances, self.l_box, self.r_max, self.particle_info)
                hist_array += self.hist_func(dist_array)
        return self.averaging(hist_array, count_samples)

    def hist_func(self, dist_array):
        hist = np.zeros([10,self.bin_res-1])
        for _ in range(10):
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
        n_Ar, n_OW, n_HW, n_Na, n_Cl = (self.particle_info == 0).sum(), (self.particle_info == 1).sum(), (self.particle_info == 2).sum(), \
                                       (self.particle_info == 3).sum(), (self.particle_info == 4).sum()
                                        
        norm_list = [n_Ar**2, 2*n_Ar*n_OW, 2*n_Ar*n_Na, 2*n_Ar*n_Cl, n_OW**2, 2*n_OW*n_Na, 2*n_OW*n_Cl, n_Na**2, 2*n_Na*n_Cl,
                     n_Cl**2, (self.n_particles-n_HW)*+2]
                     
        g_r = np.zeros([11,len(real_bins)])
        hist_array = np.append(hist_array, np.sum(hist_array,axis=0)).reshape(11,self.bin_res-1)
        print(hist_array)
        for _ in range(11):
            if np.sum(hist_array[_]) != 0:
                g_r[_] =  hist_array[_] * np.prod(self.l_box) / norm_list[_] / delta_V / count_samples
        return g_r, real_bins, ["Ar-Ar", "Ar-OH2", "Ar-Na", "Ar-Cl", "H2O-H2O", "H2O-Na", "H2O-Cl", "Na-Na",
        "Na-Cl", "Cl-Cl", "all"]


@njit
def get_dist_array(distances, l_box, r_max, p_index):
    """ 
    computes histogramm of distances, without neighbourlists
    """
    dist_Ar_Ar = []
    dist_Ar_OW = []
    dist_Ar_Na = []
    dist_Ar_Cl = []
    dist_OW_OW = []
    dist_OW_Na = []
    dist_OW_Cl = []
    dist_Na_Na = []
    dist_Na_Cl = []
    dist_Cl_Cl = []
    # silly but very fast approach to calculate different g_r`s cooler features,
    # like auto assign would`nt work with @njit decorator
    for i in range(len(distances)):
        for j in range(i+1,len(distances)):
            actual_dist = distances[i,j]
            if actual_dist <= r_max:
                if p_index[i] == 0 and p_index[j] == 0:
                    dist_Ar_Ar.append(actual_dist)
                if p_index[i] == 0 and p_index[j] == 1 or p_index[i] == 1 and p_index[j] == 0:  
                    dist_Ar_OW.append(actual_dist)
                if p_index[i] == 0 and p_index[j] == 3 or p_index[i] == 3 and p_index[j] == 0:  
                    dist_Ar_Na.append(actual_dist)
                if p_index[i] == 0 and p_index[j] == 4 or p_index[i] == 4 and p_index[j] == 0:  
                    dist_Ar_Cl.append(actual_dist)    
                if p_index[i] == 1 and p_index[j] == 1: 
                    dist_OW_OW.append(actual_dist)
                if p_index[i] == 1 and p_index[j] == 3 or p_index[i] == 3 and p_index[j] == 1:  
                    dist_OW_Na.append(actual_dist)     
                if p_index[i] == 1 and p_index[j] == 4 or p_index[i] == 4 and p_index[j] == 1:  
                    dist_OW_Cl.append(actual_dist)
                if p_index[i] == 3 and p_index[j] == 3:
                    print("gotcha")
                    dist_Na_Na.append(actual_dist)                    
                if p_index[i] == 3 and p_index[j] == 4 or p_index[i] == 4 and p_index[j] == 3:
                   dist_Na_Cl.append(actual_dist)
                if p_index[i] == 4 and p_index[j] == 4:
                    dist_Cl_Cl.append(actual_dist)                    

    return [dist_Ar_Ar, dist_Ar_OW, dist_Ar_Na, dist_Ar_Cl, dist_OW_OW, dist_OW_Na, dist_OW_Cl,
            dist_Na_Na, dist_Na_Cl, dist_Cl_Cl]
