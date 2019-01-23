import numpy as np


class LennardJones:
    def __init__(self, config):
        self.n_dim = config.n_dim
        # calculate array for epsilon where the value of element i,j corresponds to the value
        # for particles i,j of distance_vectors array according to mixing condition
        # epsilon_ij = sqrt(epsilon_i * epsilon_j)
        self.epsilon_arr = np.sqrt(np.array(config.epsilon_lj)[:, None] * np.array(config.epsilon_lj))
        # calculate array for sigma where the value of element i,j corresponds to the value
        # for particles i,j of distance_vectors array according to mixing condition
        # sigma_ij = (0.5 * (sigma_i + sigma_j))
        self.sigma_arr = (0.5 * (np.array(config.sigma_lj)[:, None] + np.array(config.sigma_lj)))
        self.cutoff = config.cutoff_lj
        self.switch_start = config.switch_start_lj

    def potential_along_axis(self, x):
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
        return x[0]

    def calc_potential(self, frame):
        # initialize output as array with distances and corresponding sigma, epsilon along axis=2
        x = frame.distance_vectors
        output = np.zeros((x.shape[0], x.shape[1], 3))
        output[:, :, 0] = np.linalg.norm(x, axis=-1)
        output[:, :, 1] = self.sigma_arr
        output[:, :, 2] = self.epsilon_arr
        # calculate potentials
        output[:, :, 0] = np.apply_along_axis(self.potential_along_axis, 2, output)
        output = np.sum(output[:, :, 0], axis=-1)
        return output

    def force_along_axis(self, x):
        # force w/o switch
        if x[0] > 0 and x[0] <= self.switch_start * x[-2]:
            x[: self.n_dim] = 24 * x[-1] * x[-2]**6 * (2 * x[-2]**6 / x[0] ** 14 - 1 / x[0] ** 8) * x[1 : self.n_dim + 1]
        # force with switch
        elif x[0] > self.switch_start * x[-2] and x[0] <= self.cutoff * x[-2]:
                t = (x[0] - self.cutoff * x[-2]) / (self.cutoff * x[-2] - self.switch_start * x[-2])
                switch = 2 * t ** 3 + 3 * t ** 2
                potential = 4 * x[-1] * x[-2]**6 * (x[-2]**6 / x[0]**12 - 1 / x[0]**6)
                dswitch = 6 / (self.cutoff * x[-2] - self.switch_start * x[-2]) / x[0] * (t ** 2 + t)
                gradient = 24 * x[-1] * x[-2]**6 * (2 * x[-2]**6 / x[0] ** 14 - 1 / x[0] ** 8)
                x[: self.n_dim] = (-potential * dswitch + gradient * switch) * x[1 : self.n_dim + 1]
        # force after cutoff
        elif x[0] > self.cutoff * x[-2]:
                x[: self.n_dim] = 0
        return x[: self.n_dim]

    def calc_force(self, frame):
        # initialize output as array with distances  and corresponding
        # distanvce vecotors, sigma, epsilon along axis=2
        x = frame.distance_vectors
        output = np.zeros((x.shape[0], x.shape[1], 3 + self.n_dim))
        output[:, :, 0] = np.linalg.norm(x, axis=-1)
        output[:, :, 1 : self.n_dim + 1] = x
        output[:, :, -2] = self.sigma_arr
        output[:, :, -1] = self.epsilon_arr
        # calculate forces
        output[:, :, : self.n_dim] = np.apply_along_axis(self.force_along_axis, 2, output)
        output = np.sum(output[:, :, : self.n_dim], axis=-2)
        return output

    def potential_neighbour(self, x, distance_vectors):
        head = distance_vectors.head
        neighbour = distance_vectors.neighbour
        cell_indexes = distance_vectors.cell_indexes
        output = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            head_index = cell_indexes[i]
            temp_arr = distance_vectors(x, i)
            temp_arr = np.delete(temp_arr, np.s_[1 : self.n_dim + 1], 1)
            temp_arr[:, 0] = np.apply_along_axis(self.potential_along_axis, 1, temp_arr)
            output[i] = np.sum(temp_arr[:, 0])
        return output

    def force_neighbour(self, x, distance_vectors):
        head = distance_vectors.head
        neighbour = distance_vectors.neighbour
        cell_indexes = distance_vectors.cell_indexes
        output = np.zeros((x.shape[0], self.n_dim))
        for i in range(x.shape[0]):
            head_index = cell_indexes[i]
            temp_arr = distance_vectors(x, i)
            temp_arr[:, : self.n_dim] = np.apply_along_axis(self.force_along_axis, 1, temp_arr)
            output[i, :] = np.sum(temp_arr[:, : self.n_dim], axis=0)
        return output
