import numpy as np

class GradientDecent:
    def init(self, phy_world, config):
        self.timestep = config.timestep
        self.l_box = config.l_box

    def run(self, force_func, potential_func, frame, next_frame, step):     
        next_frame.q = frame.q + self.timestep * force_func(frame.q, step)
        return next_frame