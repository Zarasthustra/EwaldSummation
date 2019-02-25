import numpy as np
class SanityChecks:
    def __init__(self, config, particle_initializer, masses, charges):
        self.l_box = config.l_box
        self.PBC = config.PBC
        self.n_steps = config.n_steps
        self.n_particles = config.n_particles
        self.n_dim = config.n_dim
        self.temp = config.temp
        self.timestep = config.timestep
        self.neighbour = config.neighbour
        self.l_cell = config.l_cell
        self.sigma_lj = config.sigma_lj
        self.epsilon_lj = config.epsilon_lj
        self.switch_start = config.switch_start
        self.cutoff = config.cutoff
        self.masses = masses
        self.charges = charges

    def sanity_checks(self):
        if isinstance(self.n_particles, int)==False:
            raise TypeError("n_particles has to be an integer")

        if self.n_particles <= 0:
            raise ValueError("n_particles has to be bigger than zero")

        if isinstance(self.n_dim, int) == False:
            raise TypeError("n_dim has to be an integer")

        if self.n_dim==1:
            print("Warning: not all features work in 1D")

        if self.n_dim in [1,2,3]:
            pass
        else:
            raise ValueError("n_dim has to be >=1 and <=3")

        if isinstance(self.n_steps, int)==False:
            raise TypeError("dimension should be an integer and bigger than zero")

        if self.n_steps <= 0:
            raise ValueError("n_steps has to be bigger than zero")

        if len(self.l_box) != self.n_dim:
            raise ValueError("l_box should have the right dimension")

        if any(isinstance(_, (int,float))==False for _ in self.l_box):
            raise TypeError("elements of l_box have to be int or float")

        if any(_ < 0 for _ in self.l_box):
            raise ValueError("elements of l_box have to be  >0")

        if isinstance(self.PBC, bool)==False:
            raise TypeError("PBC should be a boolean")

        if isinstance(self.temp, (int,float))==False:
            raise TypeError("Temp has to be a int or float")

        if self.temp < 0:
            raise ValueError("temp has to be bigger than zero")

        if isinstance(self.timestep, (int,float))==False:
            raise TypeError("l_box should have the right dimension")

        if self.timestep < 0:
            raise ValueError("timestep has to be bigger than zero")

        if isinstance(self.neighbour, bool)==False:
            raise TypeError("neighbour should be a boolean")

        if isinstance(self.l_cell, (int,float))==False:
            raise TypeError("l_cell has to be an int,float")

        if self.neighbour:
            if any(np.array(self.l_box) % self.l_cell != np.zeros(self.n_dim)):
                raise ValueError("l_box has to be divisible by l_cell")

        if not any(isinstance(_, (int,float)) for _ in self.sigma_lj):
            raise TypeError("sigma_lj has to be a int or float")

        if min(self.sigma_lj) < 0:
            raise ValueError("sigma_lj has to be bigger than zero")

        if not any(isinstance(_, (int,float)) for _ in self.epsilon_lj):
            raise TypeError("epsilon_lj has to be a int or float")

        if min(self.epsilon_lj) < 0:
            raise ValueError("epsilon_lj has to be bigger than zero")

        if isinstance(self.switch_start, (int,float))==False:
            raise TypeError("switch_start has to be a int or float")

        if self.switch_start < 0:
            raise ValueError("switch_start has to be bigger than zero")

        if isinstance(self.cutoff, (int,float))==False:
            raise TypeError("cutoff has to be a int or float")

        if self.cutoff <= self.switch_start:
            raise ValueError("cutoff has to be bigger than switch_start")

        if any(isinstance(_, (int,float))==False for _ in self.masses):
            raise TypeError("elements of masses have to be int or float")

        if any(_ < 0 for _ in self.masses):
            raise ValueError("elements of masses have to be  >0")

        if any(isinstance(_, (int,float))==False for _ in self.charges):
            raise TypeError("elements of charges have to be int or float")

        if np.sum(self.charges) != 0:
            raise ValueError("this program requires a neutral System")
