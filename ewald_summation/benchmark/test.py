import numpy as np
import ewald_summation as es
x=[[1,1,1],[2,2,2]]

x = np.random.uniform(0, 10.5, (100, 3))

epsilon = [1] * x.shape[0]
sigma = [1] * x.shape[0]
n_dim = x.shape[1]

#distance_vectors = es.distances.DistanceVectors(n_dim, l_box=[1,1,1] * n_dim, l_cell=3.5,
#sigma=sigma, epsilon=epsilon, neighbour=True, PBC=False)


dist = es.distances.DistanceVectors(config)
distvecarr=dist(x)
print(distvecarr)
