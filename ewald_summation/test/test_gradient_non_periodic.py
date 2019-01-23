import numpy as np
import matplotlib.pyplot as plt
import pytest
import ewald_summation as es

# test lennard jones force by comparing the output of force function from potentials
# to the negative gradient of the potential from potentials
@pytest.mark.parametrize('n_points, start, end, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj', [
    (int(3E3), 1, 2.5, [1, 1], [1, 1], 2.5, 3.5),
    (int(3E3), 2.5, 3.5, [1, 1], [1, 1], 2.5, 3.5),
    ])
def test_gradient(n_points, start, end, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj):
    # initiate classes
    simu_config = es.SimuConfig(n_dim=2, n_particles=2,
    cutoff_lj=cutoff_lj, switch_start_lj=switch_start_lj)
    distance_vectors = es.distances.DistanceVectors(simu_config)
    lennardjones = es.potentials.LennardJones(simu_config)

    # define distances array
    distances = np.linspace(start, end, n_points)

    # calculate lj potential corresponding to one particle at origin and one at
    # d for every element in distances
    potential = np.zeros(len(distances))
    for i in range(len(distances)):
        potential[i] = lennardjones.calc_potential(
                distance_vectors(np.array([[distances[i], 0],[0, 0]]), i))[0]

    # calculate the corresponding gradient by interpolation, mult by -1
    gradient_np = - 1 * np.gradient(potential, distances)

    # caculate the force for one particle at origin and one particle at d acting
    # on second particle with force methond from lennard_jones.py
    gradient_calc = np.zeros(len(distances))
    for i in range(len(distances)):
        gradient_calc[i] = lennardjones.calc_force(
                distance_vectors(np.array([[0, 0],[distances[i], 0]]), i))[1,0]
                
    # test both gradients
    np.testing.assert_allclose(gradient_np, gradient_calc, rtol=1E-2, atol=2E-5)
