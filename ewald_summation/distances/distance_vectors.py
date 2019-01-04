import numpy as np


# new distance implementation, giving all distance vectors at once
class DistanceVectors:
    def __init__(self, n_dim, l_box=[], l_cell=1, PBC=False):
        self.n_dim =  n_dim
        self.PBC = PBC
        self.l_box = l_box
        self.l_cell = l_cell
        self.neighbour_flag = False

    def distance_vectors_non_periodic(self, x):
        return x[:, None, :] - x[None, :, :]

    def distance_vectors_periodic(self, x):
        # create divisor array containing corresponding box length for x
        divisor = np.zeros(x.shape)
        for i in range(self.n_dim):
            divisor[:,i] = self.l_box[i]
        # project all particles into box
        projection = np.mod(x, divisor)
        # return distance vector tensor
        output = projection[:, None, :] - projection[None, :, :]
        return output

    def cell_linked_neighbour_list(self, x):
        self.neighbour_flag = True
        n_particles = x.shape[0]
        n_cells = [self.l_box[i] / self.l_cell for i in range(self.n_dim)]

        # create array containing cell index for every particle
        cell_indexes = np.floor(np.divide(x, self.l_cell))
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
        cell_index = self.cell_indexes[i]
        index_i = int(cell_index % self.n_cells[0])
        if self.n_dim == 2:
            index_j = int(cell_index / self.n_cells[0])
        if self.n_dim == 3:
            index_j = int((cell_index % (self.n_cells[0] * self.n_cells[1])) / self.n_cells[0])
            index_k = int(cell_index / (self.n_cells[0] * self.n_cells[1]))

        # create list with head indices for cell and neighbourcells
        head_indexes = [cell_index]
        # append head_indexes with neighbours along axis0
        if index_i == 0:
            head_indexes.append(int(cell_index + 1))
            if self.PBC:
                head_indexes.append(int(cell_index + self.n_cells[0] - 1))
        else:
            if index_i == self.n_cells[0] - 1:
                head_indexes.append(int(cell_index - 1))
                if self.PBC:
                    head_indexes.append(int(cell_index - self.n_cells[0] + 1))
            else:
                head_indexes.append(int(cell_index - 1))
                head_indexes.append(int(cell_index + 1))
        # append head_indexes with neighbours along axis1
        if index_j == 0:
            head_indexes.append(int(cell_index + self.n_cells[0]))
            if self.PBC:
                head_indexes.append(int(cell_index + self.n_cells[0] * self.n_cells[1] - self.n_cells[0]))
        else:
            if index_j == self.n_cells[1] - 1:
                head_indexes.append(int(cell_index - self.n_cells[0]))
                if self.PBC:
                    head_indexes.append(int(cell_index - self.n_cells[0] * self.n_cells[1] + self.n_cells[0]))
            else:
                head_indexes.append(int(cell_index - self.n_cells[0]))
                head_indexes.append(int(cell_index + self.n_cells[0]))
        # append head_indexes with neighbours along axis2
        if self.n_dim == 3:
            if index_k == 0:
                head_indexes.append(int(cell_index + self.n_cells[0] * self.n_cells[1]))
                if self.PBC:
                    head_indexes.append(int(cell_index + self.n_cells[0] * self.n_cells[1] * (self.n_cells[2] - 1)))
            else:
                if index_k == self.n_cells[2] - 1:
                    head_indexes.append(int(cell_index - self.n_cells[0] * self.n_cells[1]))
                    if self.PBC:
                        head_indexes.append(int(cell_index - self.n_cells[0] * self.n_cells[1] * (self.n_cells[1] - 1)))
                else:
                    head_indexes.append(int(cell_index - self.n_cells[0] * self.n_cells[1]))
                    head_indexes.append(int(cell_index + self.n_cells[0] * self.n_cells[1]))

        # get distance vectors for particle i with all particles in box and neighbour boxes
        distance_vectors = np.array([])
        for j in range(len(head_indexes)):
            list_index = self.head[head_indexes[j]]
            while list_index != -1:
                distance_vector = x[i, :] - x[list_index, :]
                distance_vectors = np.append(distance_vectors, distance_vector)
                list_index = self.neighbour[list_index]
        distance_vectors = distance_vectors.reshape(int(distance_vectors.size / self.n_dim), self.n_dim)
        return distance_vectors

    def call_function(self, x, i=0):
        if self.neighbour_flag:
            return self.distance_vectors_neighbour_list(x, i)
        else:
            if self.PBC:
                return self.distance_vectors_periodic(x)
            else:
                return self.distance_vectors_non_periodic(x)

    __call__ = call_function
