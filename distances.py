import numpy as np


def cell_linked_neighbour_list(x, l, m):
    """ Creates neighbour list

        Arguments:
            x (np.ndarray(n_particles, dim)): configuration
            l (list of floats len(dim)): length of simulation box
            m (list of floats len(dim)): number of box

        Output:
            Neighbour list """

    n_particles = x.shape[0]
    dim = x.shape[1]
    l_box = [l[i] / m[i] for i in range(dim)]
    n_cells = np.prod(m)

    def get_cell_index(n):
        """ get index for nth particle """

        i = int(x[n,0] / l_box[0])
        j = int(x[n,1] / l_box[1])
        k = int(x[n,2] / l_box[2])
        index = i + j * m[0] + k * m[0] * m[1]

        return index

    head_array = [-1] * n_cells
    neighbour_list = [-1] * n_particles
    for i in range(0, n_particles):
        cell_index = get_cell_index(i)
        # the current particle points
        # to the old head of the cell
        neighbour_list[i] = head_array[cell_index]
        # the new head of the cell
        # is the current particle
        head_array[cell_index] = i

    return head_array, neighbour_list
