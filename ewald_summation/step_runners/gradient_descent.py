import numpy as np

class GradientDescent:
    def __init__(self, rate=1.):
        self.rate = rate

    def init(self, config):
        self.timestep = config.timestep
        self.l_box = config.l_box

    def run(self, force_func, potential_func, frame, next_frame, step):
        next_frame.q = frame.q + self.timestep * force_func(frame.q, step) * self.rate
        return next_frame
