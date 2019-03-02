import numpy as np
from numpy.random import uniform

class MMC:
    def __init__(self, step=0.5):
        self.step = step

    def init(self, config):
        self.beta = 1. / (config.phys_world.k_B * config.temp)
        self.n_particles, self.n_dim = config.n_particles, config.n_dim
        self.potential = None # Storage for last frame potential

    def run(self, sum_force, sum_potential, frame, next_frame, step):
        if self.potential is None:
            self.potential = sum_potential(frame.q, step=None)
        pointToMove = np.random.randint(self.n_particles)
        proposed = frame.q.copy()
        proposed[pointToMove] += uniform(-self.step, self.step, self.n_dim)
        propose_potential = sum_potential(proposed, step=None)
        if uniform() < np.exp(self.beta * (self.potential - propose_potential)):
            next_frame.q[:, :] = proposed
            self.potential = propose_potential
        else:
            next_frame.q[:, :] = frame.q
        return next_frame
