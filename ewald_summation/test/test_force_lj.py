import numpy as np
import matplotlib.pyplot as plt
import pytest
import ewald_summation as es

# test lennard jones force by comparing the output of force function from potentials
# to the negative gradient of the potential from potentials
@pytest.mark.parametrize('n_points, start, end, epsilon_lj, sigma_lj, switch_start, cutoff', [
    (int(3E3), 1, 2.5, [1, 1], [1, 1], 2.5, 3.5),
    (int(3E3), 2.5, 3.5, [1, 1], [1, 1], 2.5, 3.5),
    ])
def test_gradient(n_points, start, end, epsilon_lj, sigma_lj, switch_start, cutoff):
    # initiate classes
    simu_config = es.SimuConfig(n_dim = 2,
                                n_particles = 2,
                                cutoff = cutoff,
                                switch_start = switch_start,
                                lj_flag = True,
                                PBC = False,
                                )
    calc_potential = es.potentials.CalcPotential(simu_config, [])
    calc_force = es.potentials.CalcForce(simu_config, [])

    # define distances array
    distances = np.linspace(start, end, n_points)

    # calculate lj potential corresponding to one particle at origin and one at
    # d for every element in distances
    potential = np.zeros(len(distances))
    for i in range(len(distances)):
        potential[i] = calc_potential(np.array([[distances[i], 0],[0, 0]]))

    # calculate the corresponding gradient by interpolation, mult by -1
    gradient_np = - 1 * np.gradient(potential, distances)

    # caculate the force for one particle at origin and one particle at d acting
    # on second particle with force methond from lennard_jones.py
    gradient_calc = np.zeros(len(distances))
    for i in range(len(distances)):
        gradient_calc[i] = calc_force(np.array([[0, 0],[distances[i], 0]]))[1,0]

    # test both gradients
    np.testing.assert_allclose(gradient_np, gradient_calc, rtol=1E-2, atol=1E-2)
