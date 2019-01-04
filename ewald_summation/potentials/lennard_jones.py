import numpy as np


class LennardJones:
    def __init__(self, n_dim, epsilon, sigma, switch_start, cutoff):
        self.n_dim = n_dim
        # calculate array for epsilon where the value of element i,j corresponds to the value
        # for particles i,j of distance_vectors array according to mixing condition
        # epsilon_ij = sqrt(epsilon_i * epsilon_j)
        self.epsilon_arr = np.sqrt(np.array(epsilon)[:, None] * np.array(epsilon))
        # calculate array for sigma where the value of element i,j corresponds to the value
        # for particles i,j of distance_vectors array according to mixing condition
        # sigma_ij = (0.5 * (sigma_i + sigma_j))
        self.sigma_arr = (0.5 * (np.array(sigma)[:, None] + np.array(sigma)))
        self.cutoff = cutoff
        self.switch_start = switch_start

    def potential(self, x):
        # write distances into array with corresponding sigma, epsilon along axis=2
        output = np.zeros((x.shape[0], x.shape[1], 3))
        output[:, :, 0] = np.linalg.norm(x, axis=-1)
        output[:, :, 1] = self.sigma_arr
        output[:, :, 2] = self.epsilon_arr

        def potential(x):
            # potenital w/o switch
            if x[0] > 0 and x[0] <= self.switch_start * x[1]:
                x[0] = 4 * x[2] * x[1]**6 * (x[1]**6 / x[0]**12 - 1 / x[0]**6)
            # potential with switch
            elif x[0] > self.switch_start * x[1] and x[0] <= self.cutoff * x[1]:
                t = (x[0] - self.cutoff * x[1]) / (self.cutoff * x[1] - self.switch_start * x[1])
                switch = 2 * t ** 3 + 3 * t ** 2
                x[0] = switch * (4 * x[2] * x[1]**6 * (x[1]**6 / x[0]**12 - 1 / x[0]**6))
            # potential after cutoff
            elif x[0] > self.cutoff * x[1]:
                x[0] = 0
            return x

        # calculate potentials
        output = np.apply_along_axis(potential, 2, output)
        output = np.sum(output[:, :, 0], axis=-1)
        return output

    def force(self, x):
        # initialize output with distances and distance vectors
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
            dswitch = 6 / (self.cutoff - self.switch_start) * (t ** 2 + t)
            gradient = 24 * self.epsilon * sigma6 * (2 * sigma6 / d ** 14 - 1 / d ** 8)
            return (potential * dswitch + gradient * switch)

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
        for i in range(self.n_dim):
            output[:, :, i+1] = np.multiply(output[:, :, i+1], output[:, :, 0])
        output = np.sum(output, axis=-2)[:, 1:]
        return output
