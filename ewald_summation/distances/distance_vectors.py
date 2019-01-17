import numpy as np


# new distance implementation, giving all distance vectors at once
# works for n_dim=2,3
class DistanceVectors:
    def __init__(self, config):
        self.n_dim =  config.n_dim
        self.PBC = config.PBC
        self.l_box = np.array(config.l_box)
        self.l_cell = config.l_cell
        self.neighbour_flag = False
        self.sigma = config.sigma_lj
        self.epsilon = config.epsilon_lj
        self.neighbour = config.neighbour
        # make array with cell indexes and pbc neighbours or -1 entries on the border for
        # self.distance_vectors_neighbour_list
        if self.neighbour:
            self.n_cells = [int(self.l_box[i] / self.l_cell) for i in range(self.n_dim)]
            self.cell_indexes_arr = -1 * np.ones(np.array(self.n_cells) + 2)
            if self.n_dim == 2:
                self.cell_indexes_arr[1:-1, 1:-1] = np.transpose(np.arange(np.prod(self.n_cells)).reshape(self.n_cells))
                if self.PBC:
                    self.cell_indexes_arr[:, 0] = self.cell_indexes_arr[:, -2]
                    self.cell_indexes_arr[:, -1] = self.cell_indexes_arr[:, 1]
                    self.cell_indexes_arr[0, :] = self.cell_indexes_arr[-2, :]
                    self.cell_indexes_arr[-1, :] = self.cell_indexes_arr[1, :]


            if self.n_dim == 3:
                self.cell_indexes_arr[1:-1, 1:-1, 1:-1] = np.transpose(np.arange(np.prod(self.n_cells)).reshape(self.n_cells))
                if self.PBC:
                    self.cell_indexes_arr[:, :, 0] = self.cell_indexes_arr[:, :, -2]
                    self.cell_indexes_arr[:, :, -1] = self.cell_indexes_arr[:, :, 1]
                    self.cell_indexes_arr[:, 0, :] = self.cell_indexes_arr[:, -2, :]
                    self.cell_indexes_arr[:, -1, :] = self.cell_indexes_arr[:, 1, :]
                    self.cell_indexes_arr[0, :, :] = self.cell_indexes_arr[-2, :, :]
                    self.cell_indexes_arr[-1, :, :] = self.cell_indexes_arr[1, :, :]



    def distance_vectors_non_periodic(self, x):
        return x[:, None, :] - x[None, :, :]

    def distance_vectors_periodic(self, x):
        # new implementation
        distance_vectors = x[:, None, :] - x[None, :, :]
        np.mod(distance_vectors, self.l_box, out=distance_vectors)
        mask = distance_vectors > np.divide(self.l_box, 2.)
        distance_vectors += mask * -self.l_box
        return distance_vectors
        '''
        # create divisor array containing corresponding box length for x
        divisor = np.zeros(x.shape)
        for i in range(self.n_dim):
            divisor[:,i] = self.l_box[i]
        # project all particles into box
        projection = np.mod(x, divisor)
        # return distance vector tensor
        output = projection[:, None, :] - projection[None, :, :]
        return output
        '''

    def cell_linked_neighbour_list(self, x):
        self.neighbour_flag = True
        n_particles = x.shape[0]
        n_cells = self.n_cells

        # create array containing cell index for every particle
        cell_indexes = np.floor(x / self.l_cell)
        cell_indexes[:, 1] = np.multiply(cell_indexes[:, 1], n_cells[0])
        if self.n_dim == 3:
            cell_indexes[:, 2] = np.multiply(cell_indexes[:, 2], n_cells[0] * n_cells[1])
        cell_indexes = np.sum(cell_indexes, axis=-1, dtype=np.int16)

        # create lists
        head = [-1] * int(np.prod(n_cells))
        neighbour = [-1] * n_particles
        for i in range(0, n_particles):
            cell_index = cell_indexes[i]
            # the current particle points
            # to the old head of the cell
            neighbour[i] = head[cell_index]
            # the new head of the cell
            # is the current particle
            head[cell_index] = i
        # define lists as class objects
        self.head = head
        self.neighbour = neighbour
        self.cell_indexes = cell_indexes
        self.n_cells = n_cells

    def distance_vectors_neighbour_list(self, x, i):
        n_cells = np.array(self.n_cells)
        cell_index = self.cell_indexes[i]
        index_ijk = [0] * self.n_dim

        # calculate cell index for i as i,j,k
        index_i = int(cell_index % n_cells[0])
        if self.n_dim == 2:
            index_j = int(cell_index / n_cells[0])
        if self.n_dim == 3:
            index_j = int((cell_index % (n_cells[0] * n_cells[1])) / n_cells[0])
            index_k = int(cell_index / (n_cells[0] * n_cells[1]))

        # create array with all cell indexes off all neighbouring cells
        if self.n_dim == 2:
            head_indexes = self.cell_indexes_arr[index_i : index_i + 3,
                                                 index_j : index_j + 3]
        if self.n_dim == 3:
            head_indexes = self.cell_indexes_arr[index_i : index_i + 3,
                                                 index_j : index_j + 3,
                                                 index_k : index_k + 3]
        head_indexes = head_indexes[head_indexes >= 0].astype(int)

        # get distance vectors for particle i with all particles in box and neighbour boxes
        # get lists for corresponding sigma and epsilon with mixing condition
        distance_vectors = np.array([])
        sigma_list = []
        epsilon_list = []
        for j in range(len(head_indexes)):
            list_index = self.head[head_indexes[j]]
            while list_index != -1:
                distance_vector_ij = x[i, :] - x[list_index, :]
                distance_vectors = np.append(distance_vectors, distance_vector_ij)
                sigma_ij = 0.5 * (self.sigma[i] + self.sigma[list_index])
                sigma_list.append(sigma_ij)
                epsilon_ij = np.sqrt(self.epsilon[i] * self.epsilon[list_index])
                epsilon_list.append(epsilon_ij)
                list_index = self.neighbour[list_index]
        distance_vectors = distance_vectors.reshape(int(distance_vectors.size / self.n_dim), self.n_dim)
        # init output as array with distances, dist vect, sigma and epsilon along axis=2
        output = np.zeros((distance_vectors.shape[0], distance_vectors.shape[1] + 3))
        output[:, 1 : self.n_dim + 1] = distance_vectors
        output[:, 0] = np.linalg.norm(output[:, 1 : self.n_dim + 1], axis=-1)
        output[:, -2] = sigma_list
        output[:, -1] = epsilon_list
        return output

    def call_function(self, x, i=0):
        if self.neighbour_flag:
            return self.distance_vectors_neighbour_list(x, i)
        elif self.PBC:
            return self.distance_vectors_periodic(x)
        else:
            return self.distance_vectors_non_periodic(x)

    __call__ = call_function