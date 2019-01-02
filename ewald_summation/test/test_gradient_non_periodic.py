import numpy as np
import matplotlib.pyplot as plt
import pytest
import ewald_summation as es


# test lennard jones force by comparing the output of force function from potentials
# to the negative gradient of the potential from potentials
@pytest.mark.parametrize('n_points, start, end, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj', [
    (int(1E5), 1, 2, 1, 1, 2.5, 3.5),
    ])
def test_gradient(n_points, start, end, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj):
    # initiate classes
    distance_vectors = es.distances.DistanceVectors()
    lennardjones = es.potentials.LennardJones(epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj)

    # define distances array
    distances = np.linspace(start, end, n_points)

    # calculate lj potential corresponding to one particle at origin and one at
    # d for every element in distances
    potential = np.zeros(len(distances))
    for i in range(len(distances)):
        potential[i] = lennardjones.potential(np.array([distances[i]]))

    # calculate the corresponding gradient by interpolation, mult by -1
    gradient_np = - 1 * np.gradient(potential, distances)
    
    # caculate the force for one particle at origin and one particle at d acting
    # on second particle with force methond from lennard_jones.py
    gradient_calc = np.zeros(len(distances))
    for i in range(len(distances)):
        gradient_calc[i] = lennardjones.force(distance_vectors(np.array([[0, 0],[distances[i], 0]])))[1,0]
    return distances, gradient_np, gradient_calc


# plot result from sigma = 1 to sigma = 3.5
a1,b1,c1 = test_gradient(int(1E4), 1, 3.5, 1, 1, 2.5, 3.5)
plt.subplot(2,1,1)
plt.plot(a1,b1)
plt.plot(a1,c1)

# plot result form sigma = 2.4 to sigma = 3.6
a2,b2,c2 = test_gradient(int(1E4), 2.4, 3.6, 1, 1, 2.5, 3.5)
plt.subplot(2,1,2)
plt.plot(a2,b2)
plt.plot(a2,c2)
plt.show()
