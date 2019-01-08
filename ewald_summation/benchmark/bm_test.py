import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

### I use this for development, so its always dirty ###
### right now it computes potentials with and without neighbour lists and compares the results

x = np.random.uniform(0, 10.5, (10, 2))

# x = np.random.uniform([[0.1, 0.1],
#                        [1.03191961, 0.21524854],
#                        [3.22118963, 0.54861028],
#                        [4.27407734, 7.95771957],
#                        [5.32091109, 5.34034925],
#                        [4.68244577, 5.72148385],
#                        [6.97563888, 8.41157314],
#                        [3.50278161, 9.11960907],
#                        [2.42106713, 5.68641472],
#                        [0.94817014, 7.4468208 ],
#                        ])

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
dist = distance_vectors(x, 0)
# print(dist)
potential1 = lennard_jones.potential_neighbour(x, distance_vectors)
distance_vectors.neighbour_flag = False
potential2 = lennard_jones.potential(distance_vectors(x))
a = distance_vectors(x)[0, :, :]
b = np.zeros((a.shape[0], a.shape[1] + 1))
b[:, 0] = np.linalg.norm(a, axis=-1)
b[:, 1:] = a
# print(b)
print(potential2 - potential1)
# print(x)
# print('cell_index', distance_vectors.cell_indexes)
# print(dist)
