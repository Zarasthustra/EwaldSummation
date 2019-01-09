import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

### I use this for development, so its always dirty ###
### right now it computes potentials with and without neighbour lists and compares the results

x = np.random.uniform(0, 10.5, (10, 2))

x = np.array([[ 0.7981622,   4.36161408],
 # [ 8.0316976 ,  2.82936701],
 # [ 5.73925569,  0.38718973],
 # [ 8.09399424,  6.32034004],
 [ 2.99969741,  6.81350401],
 # [ 9.37189372,  6.59517841],
 # [ 7.57672347, 10.01023115],
 [ 3.6382473 ,  7.89578095],
 [ 0.52676375,  5.6192579 ],
 # [ 8.1107442,   3.16067917],
                       ])

epsilon = [1] * x.shape[0]
sigma = [1] * x.shape[0]
n_dim = x.shape[1]

distance_vectors = es.distances.DistanceVectors(n_dim, l_box=[10.5] * n_dim, l_cell=3.5,
sigma=sigma, epsilon=epsilon, neighbour=True, PBC=False)
# print(distance_vectors.cell_indexes_arr)
distance_vectors.cell_linked_neighbour_list(x)
# print(distance_vectors.distance_vectors_neighbour_list(x, 0))
# # print(len(distance_vectors.distance_vectors_neighbour_list(x, 0)))
lennard_jones = es.potentials.LennardJones(2, epsilon, sigma, 2.5, 3.5)
dist = distance_vectors(x, 1)[:, :3]
print(dist)
potential1 = lennard_jones.potential_neighbour(x, distance_vectors)
distance_vectors.neighbour_flag = False
potential2 = lennard_jones.potential(distance_vectors(x))
a = distance_vectors(x)[1, :, :]
b = np.zeros((a.shape[0], a.shape[1] + 1))
b[:, 0] = np.linalg.norm(a, axis=-1)
b[:, 1:] = a
# print('nieghbor')
# print(b - dist)
print(potential2 - potential1)
print(b)
b.sort(axis=0)
dist.sort(axis=0)
print(b - dist)
# print('cell_index', distance_vectors.cell_indexes)
# print(dist)
