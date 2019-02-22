import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt
import math

### I use this for development, so its always dirty ###


# old implementation for lj pot with pbc
def distance_vectors_periodic(x, l_box):
    l_box = np.array(l_box)
    distance_vectors = x[:, None, :] - x[None, :, :]
    np.mod(distance_vectors, l_box, out=distance_vectors)
    mask = distance_vectors > np.divide(l_box, 2.)
    distance_vectors += mask * -l_box
    return np.linalg.norm(distance_vectors, axis=-1)

x = np.array([[4.1, 0], [1, 0]])
n_particles = x.shape[0]
n_dim = x.shape[1]
pbc = True
l_box = (2, 2)
l_box_half = (1, 1)
max_length = np.linalg.norm(l_box) / 2

for i in range(n_particles):
    for j in range(i + 1, n_particles):
        distance_temp = 0
        distance_squared = 0
        if pbc:
            for k in range(n_dim):
                distance_temp = (x[i, k] - x[j, k]) % l_box_half[k]
                if distance_temp < l_box_half[k]:
                    distance_temp -= l_box[k]
            distance_squared += distance_temp**2
        distance = math.sqrt(distance_squared)
        print(distance)

print(distance_vectors_periodic(x, l_box))
