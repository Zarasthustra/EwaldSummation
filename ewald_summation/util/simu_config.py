import numpy as np


class SimuConfig:
    def __init__(self, n_dim=3, l_box=[1.,1.,1.], PBC=False, n_particles=1, n_steps=1000,
     timestep=1e-10, neighbour=False, l_cell=1, temp=1, alpha=1, rec_reso=6,
     switch_start=2.5, cutoff=3.5, sigma_lj=1, epsilon_lj=1, lj_flag=True, coulomb_flag=False,
     parallel_flag=False):
        # TODO: sanity checks
        self.l_box = l_box
        self.PBC = PBC
        self.n_steps = n_steps
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.lj_flag = lj_flag
        self.coulomb_flag = coulomb_flag
        # move these initializations to initializer
        self.masses = None
        self.charges = None
        self.alpha = alpha
        self.rec_reso = rec_reso
        self.temp = temp
        self.timestep = timestep
        self.neighbour = neighbour
        self.l_cell = l_cell
        # Temp
        self.switch_start = switch_start
        self.cutoff = cutoff
        if isinstance(sigma_lj, (int, float)):
            self.sigma_lj = [sigma_lj] * self.n_particles
        else:
            self.sigma_lj = sigma_lj
        if isinstance(epsilon_lj, (int, float)):
            self.epsilon_lj = [epsilon_lj] * self.n_particles
        else:
             self.epsilon_lj = epsilon_lj
        self.parallel_flag = parallel_flag
