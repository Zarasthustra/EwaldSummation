import numpy as np


class SimuConfig:
    def __init__(self, n_dim=3, l_box=[1.,1.,1.], PBC=False, n_particles=1, n_steps=1000,\
     timestep=1e-10, neighbour_flag=False, l_cell=1, temp=300):
        # TODO: sanity checks
        self.l_box = l_box
        self.PBC = PBC
        self.n_steps, self.n_particles, self.n_dim = n_steps, n_particles, n_dim
        # move these initializations to initializer
        self.masses = None
        self.charges = None
        self.temp = temp
        self.timestep = timestep
        self.neighbour_flag = neighbour_flag
        self.l_cell = l_cell
