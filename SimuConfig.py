import numpy as np
class SimuConfig:
    def __init__(self, n_dim=3, box_size=(1.,1.,1.), n_particles=1, n_steps=1000,\
     timestep=1e-10, masses=[1.], charges=[0.], temp=300):
        # TODO: sanity checks
        self.box_size = np.array(box_size)
        self.n_steps, self.n_particles, self.n_dim = n_steps, n_particles, n_dim
        self.masses = np.array(masses)
        self.charges = np.array(charges)
        self.temp = temp
        self.timestep = timestep
