import numpy as np


# new distance implementation, giving all distance vectors at once
# works for n_dim=2,3
class DistanceVectors:
    def __init__(self, config):
        self.n_dim =  config.n_dim
        self.n_particles = config.n_particles
        self.PBC = config.PBC
        self.l_box = np.array(config.l_box)
        self.l_cell = config.l_cell
        self.neighbour_flag = False
        self.sigma = config.sigma_lj
        self.epsilon = config.epsilon_lj
        self.neighbour = config.neighbour
        self.current_frame = self.CurrentFrame(self.n_particles, self.n_dim)

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

    # class to store frame so it can be reused for different potentials
    class CurrentFrame:
        def __init__(self, n_particles, n_dim):
            # set step to -2 so integrator can set it to -1 when initialised
            self.step = -2

            # init storage arrays
            self.distance_vectors = np.zeros((n_particles, n_particles, n_dim))
            self.distances_squared = np.zeros((n_particles, n_particles))
            self.distances = np.zeros((n_particles, n_particles))

        def store_frame(self, distance_vectors):
            self.distance_vectors = distance_vectors
            self.distances_squared = np.sum(self.distance_vectors**2, axis=-1)
            self.distances = np.sqrt(self.distances_squared)

    def distance_vectors_non_periodic(self, x):
        return x[:, None, :] - x[None, :, :]

    def distance_vectors_periodic(self, x):
        # new implementation
        distance_vectors = x[:, None, :] - x[None, :, :]
        np.mod(distance_vectors, self.l_box, out=distance_vectors)
        mask = distance_vectors > np.divide(self.l_box, 2.)
        distance_vectors += mask * -self.l_box
        return distance_vectors

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
        n_particles_cell = [0] * np.prod(n_cells)
        for i in range(0, n_particles):
            cell_index = cell_indexes[i]
            # count how many particles are in each cell
            n_particles_cell[cell_index] += 1
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
        self.n_particles_cell = n.n_particles_cell

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
        array_index_list = []
        for j in range(len(head_indexes)):
            list_index = self.head[head_indexes[j]]
            while list_index != -1:

                # append distance_vectors list
                distance_vector_ij = x[i, :] - x[list_index, :]
                distance_vectors = np.append(distance_vectors, distance_vector_ij)

                # append list index list, gvining information for which particle the
                # distance vector was calculated
                array_index_list.append(list_index)

                # update list index variable for calculation to jump to next particle
                list_index = self.neighbour[list_index]

        #reshape distance_vectors array to propter shape, make entire output ndarray
        distance_vectors = distance_vectors.reshape(int(distance_vectors.size / self.n_dim), self.n_dim)
        array_index_list = np.array(array_index_list)
        return output, array_index_list

    def call_function(self, x, step):
        # compute frame obj, if not stored for this step in iteration already.
        if self.current_frame.step == step:

        # return stored frame obj
            return self.current_frame
        else:

            # set set variable in frame
            self.current_frame.step = step

            # update current_frame
            if self.neighbour_flag:
                self.distance_vectors = self.distance_vectors_neighbour_list(x, i)
            elif self.PBC:
                self.current_frame.store_frame(self.distance_vectors_periodic(x))
            else:
                self.current_frame.store_frame(self.distance_vectors_non_periodic(x))

            # return updated current_frame
            return self.current_frame

    __call__ = call_function
