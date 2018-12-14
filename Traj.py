import numpy as np

class Traj:

    def __init__(self, box_size, n_particles, n_steps, timestep):
        # TODO: sanity checks
        self.box_size = box_size
        self.n_dim = np.shape(box_size)[0]
        self.qs = np.zeros((self.n_steps, self.n_particles, self.n_dim))
        self.ps = np.zeros((self.n_steps, self.n_particles, self.n_dim))
        self.timestep = timestep
        if timestep is not None:
            self.ts = np.arange(n_steps) * timestep

    def fetch_q(i_frame):
        return qs[i_frame]

    def fetch_p(i_frame):
        return ps[i_frame]
