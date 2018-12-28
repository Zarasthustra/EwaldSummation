import numpy as np


# new distance implementation, giving all distance vectors at once
class DistanceVectors:
    def __init__(self, periodicity=False, l_box=1, l_cell=1):
        self.periodicity = periodicity
        self.l_box = l_box
        self.l_cell = l_cell

    def distances_non_periodic(self, x):
        return x[:, None, :] - x[None, :, :]

    __call__ = distances_non_periodic
