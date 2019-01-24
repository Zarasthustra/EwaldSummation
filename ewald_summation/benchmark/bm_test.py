import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

### I use this for development, so its always dirty ###

x = np.array([[0, 0],
              [.5, .5],
              [.5, .7],
              ])

test_config = es.SimuConfig(n_dim=x.shape[1],
                            l_box=[3.3] * x.shape[1],
                            l_cell=1.1,
                            n_particles=x.shape[0],
                            n_steps=3000,
                            timestep=0.001,
                            temp=100,
                            PBC=False,
                            neighbour=True,
                            )
lennard_jones = es.potentials.LennardJones(test_config)
distance_vectors = es.distances.DistanceVectors(test_config)
print(lennard_jones.calc_potential(distance_vectors(x, 0)))
# print(distance_vectors.head)
# print(distance_vectors.neighbour)
# print(distance_vectors.cell_indexes)
# print(distance_vectors.n_particles_cell)
# print('############')
# print(distance_vectors.distance_vectors_neighbour_list(x, 0)[0])
# print(distance_vectors.distance_vectors_neighbour_list(x, 0)[1])
# distance_vectors.distance_vectors_neighbour_list(x)
