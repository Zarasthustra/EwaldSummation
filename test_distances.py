import numpy as np
import pytest
from distances import cell_linked_neighbour_list

def get_test_configuration(n, d):
    """ generates evenly distributed particles with distance d
        works for 2 and 3 dim

            Arguments:
                n (list of int len(dim)): number of particles along axis
                d (list of float len(dim)): distance of particles along axis

            Output:
                np.ndarray(n, d):  configuration """

    dim = len(n)
    n_particels = np.product(n)
    configuration = np.zeros((n_particels, dim))

    # loop generating configuration, there is prob better solution
    k = 0
    if dim == 2:
        for i in range(n[0]):
            for j in range(n[1]):
                configuration[k,:] = [i * d[0], j * d[1]]
                k += 1

    if dim == 3:
        for i in range(n[0]):
            for j in range(n[1]):
                for l in range(n[2]):
                    configuration[k,:] = [i * d[0], j * d[1] ,l * d[2]]
                    k += 1

    return configuration

test_configuration = get_test_configuration([3,3,3], [1,1,1])

print(test_configuration)

print(cell_linked_neighbour_list(test_configuration, [2,2,2], [2.1,2.1,2.1]))
