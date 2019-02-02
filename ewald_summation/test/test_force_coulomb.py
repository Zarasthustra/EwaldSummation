import numpy as np
import pytest
import ewald_summation as es
import matplotlib.pyplot as plt

'''
@pytest.mark.parametrize('x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj, l_box, l_cell', [
    (np.random.uniform(0, 6.9, (150, 2)), 1, 1, 2.5, 3.5, [14, 14], 3.5),
    (np.random.uniform(0, 6.9, (150, 3)), 1, 1, 2.5, 3.5, [14, 14, 14], 3.5),
    ])
def test_potential_neighbour(x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj, l_box, l_cell):
    simu_config = es.SimuConfig(n_dim=x.shape[1], n_particles=x.shape[0], l_box=l_box, l_cell=l_cell,
    switch_start_lj=switch_start_lj, cutoff_lj=cutoff_lj, neighbour=True)
    distance_vectors = es.distances.DistanceVectors(simu_config)
    lennard_jones = es.potentials.LennardJones(simu_config)
    potential1 = lennard_jones.calc_potential(distance_vectors(x, 0))
    distance_vectors.neighbour_flag = False
    potential2 = lennard_jones.calc_potential(distance_vectors(x, 0))
    np.testing.assert_allclose(potential1, potential2)
'''

def test_force_coulomb():
    '''Ensure the correctness of Coulomb force calculation by the testing if the potentials from direct calculation and
    from numerical integration of force are the same.
    '''
    n_dim = 3
    l_box = np.array([3., 3., 3.])
    n_particles = 2
    q = np.array([[0., 0., 0.], [1., 0., 0.]])
    charge_vector = np.array([-1., 1.])
    simu_config = es.SimuConfig(n_dim=q.shape[1], n_particles=q.shape[0], l_box=l_box, l_cell=l_box[0], neighbour=True)
    simu_config.charges = charge_vector
    distance_vectors = es.distances.DistanceVectors(simu_config)
    coulomb = es.potentials.Coulomb(simu_config)

    xs = np.arange(1., 2., 0.01)
    pots = np.zeros(100)
    forces = np.zeros(100)
    force_inte = np.zeros(100)
    step = 0
    for i in range(100):
        q[1, 0] = xs[i]
        step += 1
        pots[i] = coulomb.calc_potential(q, distance_vectors(q, step))
        forces[i] = coulomb.calc_force(q, distance_vectors(q, step))[1, 0]
    force_inte[0] = pots[0]
    for i in range(1, 100):
        force_inte[i] = force_inte[i - 1] + 0.01 * -forces[i - 1]
    np.testing.assert_allclose(pots, force_inte, rtol=1e-02)

