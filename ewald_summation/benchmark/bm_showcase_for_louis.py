import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt



test_config = es.SimuConfig(n_dim=3, l_box=(2., 2., 2), n_particles=4,
                            n_steps=10000, timestep=0.001, temp=300)

x = np.random.random((test_config.n_particles, test_config.n_dim))

distance_vectors = es.distances.DistanceVectors(test_config)

distance_vectors_array = distance_vectors(x)
print(distance_vectors_array, distance_vectors_array.shape)
print(np.linalg.norm(distance_vectors_array,axis=2),np.linalg.norm(distance_vectors_array,axis=-1).shape)
#print(np.sum(distance_vectors_array, axis=3))
