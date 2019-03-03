import numpy as np
import pytest
import ewald_summation as es

dummy_world = es.PhysWorld()
dummy_world.k_B = 1.
dummy_world.k_C = 1.
dummy_world.particle_types = [
            ('dummy_Ar', 1., 1., 1., 1.), #0
            ('dummy_+', 1., 1., 1., 1.), #1
            ('dummy_-', 1., -1., 1., 1.) #2
            ]
dummy_world.molecule_types = []

# old implementation for lj pot with pbc
def distance_vectors_periodic(x, l_box):
    l_box = np.array(l_box)
    distance_vectors = x[:, None, :] - x[None, :, :]
    np.mod(distance_vectors, l_box, out=distance_vectors)
    mask = distance_vectors > np.divide(l_box, 2.)
    distance_vectors += mask * -l_box
    return distance_vectors


def lj_potential_total(distance_vectors):
    distances = np.linalg.norm(distance_vectors, axis=-1)
    n_particles = distances.shape[0]
    potential = 0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # print('distances_old', distances[i, j])
            potential += lj_potential_pairwise(distances[i, j])
    return potential


def lj_potential_pairwise(distance):
    epsilon_lj = 1.
    sigma_lj = 1
    cutoff_lj = 3.5 * sigma_lj
    switch_width_lj = 1
    ndim = 2
    if(distance <= 0 or distance > cutoff_lj):
        return 0.
    else:
        inv_dist = sigma_lj / distance
        inv_dist2 = inv_dist * inv_dist
        inv_dist4 = inv_dist2 * inv_dist2
        inv_dist6 = inv_dist2 * inv_dist4
        phi_LJ = 4. * epsilon_lj * inv_dist6 * (inv_dist6 - 1.)
        if(distance <= cutoff_lj - switch_width_lj):
            return phi_LJ
        else:
            t = (distance - cutoff_lj) / switch_width_lj
            switch = t * t * (3. + 2. * t)
            return phi_LJ * switch

@pytest.mark.parametrize('x', [
    (np.array([[0, 0, 0], [1, 0, 0]])),
    (np.array([[0, 0, 1],[1, 0, 0], [0, 1, 0]])),
    (np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]])),
    # (np.array([[0,0,0],[0,1,3.1]]), 1, 1, 2.5, 3.5),
    (np.random.uniform(-20, 20, (100, 2))),
    # (np.random.uniform(-20, 20, (100, 3))),
    ])
def test_potential(x):
    test_config = es.SimuConfig(l_box=[10.] * x.shape[1],
                                PBC=True,
                                particle_info=[0] * x.shape[0],
                                n_steps=2000,
                                timestep=0.001,
                                temp=30,
                                phys_world=dummy_world,
                                )
    pot = es.potentials.LJ(test_config, 2.5, 3.5)
    pot.set_positions(x)
    potential1 = pot.pot
    # legacy potential implementaton, requires sigma, epsilon = 1
    # and a switch region of 2.5 to 3.5
    potential2 = es.potentials.lj_potential_total(x)
    np.testing.assert_allclose(potential1, potential2)
