import numpy as np
from numpy.random import uniform

class MMC:
    def __init__(self, step=0.5):
        self.step = step

    def init(self, config):
        self.beta = 1. / (config.phys_world.k_B * config.temp)
        self.n_particles, self.n_dim = config.n_particles, config.n_dim
        self.mc_step = self.step
        #inital maximal moving particle step
        self.mc_accepted = 0
        # counter for accepted moves
        self.potential = None # Storage for last frame potential

    def run(self, sum_force, sum_potential, frame, next_frame, step):
        if self.potential is None:
            self.potential = sum_potential(frame.q, step=None)
            
        if step % 100 == 0 and step != 0:
            acc_rate = self.mc_accepted / 100
            if acc_rate >= 0.6:
                self.mc_step *= 1.05
            if acc_rate <= 0.4:
                self.mc_step *= 0.95
            self.mc_accepted = 0
        
        # dynamically change the acceptance rate to make shure that the phase space 
        # is accurately sampled. The interval 40-60 % is taken form Allen Tildesley 1989   
        pointToMove = np.random.randint(self.n_particles)
        proposed = frame.q.copy()
        proposed[pointToMove] += uniform(-self.mc_step, self.mc_step, self.n_dim)
        propose_potential = sum_potential(proposed, step=None)
        if uniform() < np.exp(self.beta * (self.potential - propose_potential)):
            next_frame.q[:, :] = proposed
            self.potential = propose_potential
            self.mc_accepted += 1
        else:
            next_frame.q[:, :] = frame.q
        return next_frame
