import numpy as np


class SimuConfig:
    def __init__(self, n_dim=3, l_box=[1.,1.,1.], PBC=False, n_particles=1, n_steps=1000,\
     timestep=1e-10, neighbour=False, l_cell=1, temp=300, switch_start_lj=2.5, cutoff_lj=3.5):
        # TODO: sanity checks
        self.l_box = l_box
        self.PBC = PBC
        self.n_steps = n_steps
        self.n_particles = n_particles
        self.n_dim = n_dim
        # move these initializations to initializer
        self.masses = None
        self.charges = None
        self.temp = temp
        self.timestep = timestep
        self.neighbour = neighbour
        self.l_cell = l_cell
        # Temp
        self.sigma_lj = [1] * self.n_particles
        self.epsilon_lj = [1] * self.n_particles
        self.switch_start_lj = switch_start_lj
        self.cutoff_lj = cutoff_lj
