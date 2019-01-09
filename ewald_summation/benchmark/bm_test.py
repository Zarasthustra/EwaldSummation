import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

### I use this for development, so its always dirty ###
### right now it computes potentials with and without neighbour lists and compares the results

x = np.random.uniform(0, 10.5, (100, 3))

# x = np.array([[ 0,   0, 0],
#                [1, 1, 1],
#                        ])

epsilon = [1] * x.shape[0]
sigma = [1] * x.shape[0]
n_dim = x.shape[1]

distance_vectors = es.distances.DistanceVectors(n_dim, l_box=[10.5] * n_dim, l_cell=3.5,
sigma=sigma, epsilon=epsilon, neighbour=True, PBC=False)
distance_vectors.cell_linked_neighbour_list(x)
lennard_jones = es.potentials.LennardJones(n_dim, epsilon, sigma, 2.5, 3.5)
# print(distance_vectors(x, 0))
dist = distance_vectors(x, 0)
# print(dist)
force1 = lennard_jones.force_neighbour(x, distance_vectors)
distance_vectors.neighbour_flag = False
force = lennard_jones.force(distance_vectors(x))
print(force1 -  force)
# print(b)
# potential1.sort(axis=0)
# potential2.sort(axis=0)
# print(potential1 - potential2)
# print('cell_index', distance_vectors.cell_indexes)
# print(dist)
