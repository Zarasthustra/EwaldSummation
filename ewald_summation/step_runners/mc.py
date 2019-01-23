import numpy as np
from numpy.random import uniform

class MMC:
    def __init__(self, step=0.5):
        self.step = step

    def init(self, phy_world, config):
        self.beta = 1. / (phy_world.k_B * config.temp)
        self.n_particles, self.n_dim = config.n_particles, config.n_dim
        self.potential = None # Storage for last frame potential

    def run(self, force_func, potential_func, frame, next_frame, step):
        if self.potential is None:
            self.potential = potential_func(frame.q, -1)
        pointToMove = np.random.randint(self.n_particles)
        proposed = frame.q.copy()
        proposed[pointToMove] += uniform(-self.step, self.step, self.n_dim)
        propose_potential = potential_func(proposed, step)
        if uniform() < np.exp(self.beta * (self.potential - propose_potential)):
            next_frame.q[:, :] = proposed
            self.potential = propose_potential
        else:
            next_frame.q[:, :] = frame.q
        return next_frame
