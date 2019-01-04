import numpy as np

class Traj:

    def __init__(self, config):
        # TODO: sanity checks
        self.l_box = config.l_box
        self.n_dim = config.n_dim
        self.timestep = config.timestep
        self.n_particles = config.n_particles
        self.n_steps = config.n_steps

        self._frames = [self.make_new_frame()]
        self.current_frame_num = 0
        self.current_time = 0.
        if self.timestep is not None:
            self.ts = np.arange(self.n_steps + 1) * self.timestep

    def make_new_frame(self):
        return TrajFrame(self.n_dim, self.n_particles)

    def get_current_frame(self):
        return self._frames[self.current_frame_num]

    def set_new_frame(self, new_frame):
        # check
        self._frames.append(new_frame)
        self.current_frame_num += 1
        self.current_time += self.timestep

    def get_ps(self):
        return np.array([frame.p for frame in self._frames])

    def get_qs(self):
        return np.array([frame.q for frame in self._frames])

class TrajFrame:
    def __init__(self, n_dim, n_particles):
        self.q = np.zeros((n_particles, n_dim))
        self.p = np.zeros((n_particles, n_dim))
