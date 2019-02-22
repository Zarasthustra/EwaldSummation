import numpy as np


class SimuConfig:
    def __init__(self, n_dim=3, l_box=[1.,1.,1.], PBC=True, n_particles=1, n_steps=1000,\
     timestep=1e-10, neighbour=False, l_cell=1, start_sampling=100, sampling_rate=10, temp=1,\
     mole_fraction=0.5, switch_start_lj=2.5, cutoff_lj=3.5, sigma_lj=1, epsilon_lj=1, 
     p_kinds=[200,200,200]):
     
        # TODO: sanity checks
        self.l_box = l_box
        self.PBC = PBC
        self.n_steps = n_steps
        self.n_particles = n_particles
        self.n_dim = n_dim
        # move these initializations to initializer
        self.p_kinds = p_kinds
       	self.start_sampling = start_sampling
       	self.sampling_rate = sampling_rate
        self.masses = None
        self.charges = None
        self.temp = temp
        self.timestep = timestep
        self.neighbour = neighbour
        self.l_cell = l_cell
        # Temp
        self.mole_fraction = mole_fraction       
        self.switch_start_lj = switch_start_lj
        self.cutoff_lj = cutoff_lj
        self.sigma_lj = sigma_lj
        self.epsilon_lj = epsilon_lj
