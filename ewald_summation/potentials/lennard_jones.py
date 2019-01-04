import numpy as np


class LennardJones:
    def __init__(self, epsilon, sigma, switch_start, cutoff):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.switch_start = switch_start

    def potential(self, x):
        distances = np.linalg.norm(x, axis=-1)

        # potential_pairwise
        def p1(d):
            sigma6 = self.sigma ** 6
            potential = 4 * self.epsilon * sigma6 * (sigma6 / d ** 12 - 1 / d ** 6)
            return potential

        # potential_pairwise with switch function smoothstep S1
        def p2(d):
            t = (d - self.cutoff) / (self.cutoff - self.switch_start)
            switch_function = t * t * (3. + 2. * t)
            sigma6 = self.sigma ** 6
            potential = 4 * self.epsilon * sigma6 * (sigma6 / d ** 12 - 1 / d ** 6)
            return potential * switch_function

        # piecewise function for Lennard Jones Potential
        def p12(d):
            output = np.piecewise(d, [d <= 0,
                                 (0 < d) & (d < self.switch_start),
                                 (self.switch_start <= d) & (d < self.cutoff),
                                 self.cutoff <= d],
                                 [0, p1, p2,0]
                                 )
            return output

        # sum potentials for every particle
        potential = np.sum(p12(distances), axis=-1)
        return potential

    def force(self, x):
        # initialize output with distances and distance vectors
        n_dim = x.shape[-1]
        shape = list(x.shape)
        shape[-1] += 1
        output = np.zeros(shape)
        output[:,:,1:] = x
        output[:,:,0] = np.linalg.norm(x, axis=-1)

        # force pairwise
        def f1(d):
            sigma6 = self.sigma ** 6
            gradient = 24 * self.epsilon * sigma6 * (2 * sigma6 / d ** 14 - 1 / d ** 8)
            return gradient

        # force pairwise with switch_function
        def f2(d):
            sigma6 = self.sigma ** 6
            t = (d - self.cutoff) / (self.cutoff - self.switch_start)
            switch = 2 * t ** 3 + 3 * t ** 2
            potential = 4 * self.epsilon * sigma6 * (sigma6 / d ** 12 - 1 / d ** 6)
            dswitch = 6 / (self.cutoff - self.switch_start) / d * (t ** 2 + t)
            gradient = 24 * self.epsilon * sigma6 * (2 * sigma6 / d ** 14 - 1 / d ** 8)
            return (-potential * dswitch + gradient * switch)

        # piecewise function for lennard jones forces
        def f12(d):
            output = np.piecewise(d, [d <= 0,
                                 (0 < d) & (d < self.switch_start),
                                 (self.switch_start <= d) & (d < self.cutoff),
                                 self.cutoff <= d],
                                 [0, f1, f2,0]
                                 )
            return output

        # apply piecewise function to distances and multiply with vectors
        output[:, :, 0] = f12(output[:, :, 0])
        for i in range(n_dim):
            output[:, :, i+1] = np.multiply(output[:, :, i+1], output[:, :, 0])
        output = np.sum(output, axis=-2)[:, 1:]
        return output
