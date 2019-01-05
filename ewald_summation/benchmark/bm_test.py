import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

epsilon = [1, 1, 1, 1]
sigma = [1, 1, 1, 1]
x = np.array([[0, 0],
              [1, 1],
              [2, 2],
              [3, 3],
              ])

distance_vectors = es.distances.DistanceVectors(2)
lennard_jones = es.potentials.LennardJones(2, epsilon, sigma, 2.5, 3.5)

force = lennard_jones.force(distance_vectors(x))
print(force)
