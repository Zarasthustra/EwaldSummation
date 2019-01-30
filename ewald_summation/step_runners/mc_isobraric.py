import numpy as np
from numpy.random import uniform

class MCIsobaric:
    def __init__(self, v_max=0.5):
        self.step = step

    def init(self, phy_world, config):
        self.beta = 1. / (phy_world.k_B * config.temp)
        self.n_particles, self.n_dim = config.n_particles, config.n_dim
        self.pressure, self.l_box = config.pressure, config.l_box
        self.potential = None # Storage for last frame potential

    def run(self, force_func, potential_func, frame, next_frame, step):
        if self.potential is None:
            self.potential = potential_func(frame.q, -1)
        proposed = frame.q.copy()
        volume = np.prod(l_box)
        proposed_ln_v = np.log(np.prod(self.l_box)) + (uniform() - 0.5) * v_max
        # random walk in log Volume
        proposed_v = np.exp(ln_v)
        proposed_l_box = proposed_v ** (1./self.n_dim) * l_box / np.linalg.norm(l_box)
        propodes = proposed_l_box * proposed
        propose_potential = potential_func(proposed, step)
        if uniform() < np.exp(self.beta * (self.potential - propose_potential   \
                       - self.pressure*(proposed_v-volume)+(self.n_particles+1) \
                       /self.beta*np.log(proposed_v/v))):
            next_frame.q[:, :] = proposed
            self.potential = propose_potentiasl
        else:
            next_frame.q[:, :] = frame.q
        return next_frame
