import numpy as np
import pytest
import ewald_summation as es


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

@pytest.mark.parametrize('x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj, l_box', [
    (np.array([[0, 0], [0, 1.1]]), 1, 1, 2.5, 3.5, (2, 2)),
    (np.array([[0, 0, 0], [0, 0, 1]]), 1, 1, 2.5, 3.5, (5, 5, 5)),
    (np.array([[0, 0, 0], [0, 0, 7]]), 1, 1, 2.5, 3.5, (5, 5, 5)),
    (np.array([[1, 0, 0], [0, 1, -7]]), 1, 1, 2.5, 3.5, (5, 5, 5)),
    (np.random.uniform(-2, 10, (100, 2)), 1, 1, 2.5, 3.5, (5, 5)),
    (np.random.uniform(-2, 10, (100, 3)), 1, 1, 2.5, 3.5, (5, 5, 5)),
    ])
def test_potential(x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj, l_box):
    l_box_half = tuple(np.divide(np.array(l_box), 2))
    general_params = (x.shape[1], x.shape[0], l_box, l_box_half, True)
    lj_params = (2.5, 3.5, tuple([1] * x.shape[0]), tuple([1] * x.shape[0]))
    potential1 = es.potentials.calc_potential_pbc(x, general_params, lj_params)
    # legacy potential implementaton, requires sigma, epsilon = 1
    # and a switch region of 2.5 to 3.5
    potential2 = lj_potential_total(distance_vectors_periodic(x, l_box))
    np.testing.assert_allclose(potential1, potential2)
