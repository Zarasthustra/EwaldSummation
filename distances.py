import numpy as np


def cell_linked_neighbour_list(x, l, m):
    """ Creates neighbour list where
        position of head is number of cell
        position in list is index of particle
        entry in list points to next particle
        -1 signals end of list

        Arguments:
            x (np.ndarray(n_particles, dim)): configuration
            l (list of floats len(dim)): length of simulation box
            m (float): box length

        Output:
            Neighbour list """

    n_particles = x.shape[0]
    dimensions = x.shape[1]
    l_cells = np.divide(l, m)
    n_cells = np.round(np.divide(l, l_cells))
    n_cells_total = int(np.product(n_cells))

    # get cell indexes as array, index for last axis moves fastest
    cell_indexes = np.zeros((n_particles, dimensions))
    for i in range(dimensions):
        cell_indexes[:,i] = np.divide(x[:,i], l_cells[i])
    cell_indexes = np.floor(cell_indexes)
    for i in range(1, dimensions):
        cell_indexes[:,i] = np.multiply(cell_indexes[:,i], np.product(n_cells[i]))
    cell_indexes = np.sum(cell_indexes, axis=1)

    # create head list and neighbour list
    head = [-1] * n_cells_total
    neighbour = [-1] * n_particles
    for i in range(0, n_particles):
        cell_index = int(cell_indexes[i])
        neighbour[i] = head[cell_index]
        head[cell_index] = i

    return head, neighbour

test_array = np.array([[0,0,0],[1,1,1],[3,1.5,0],[2,2,2]])
print(test_array)
a = cell_linked_neighbour_list(test_array, [4, 4, 4], 2)
print(a)
