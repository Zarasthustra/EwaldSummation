import numpy as np
import pytest
import ewald_summation as es


@pytest.mark.parametrize('x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj', [
    (np.array([[0,0,0],[1,1,1]]), 1, 1, 2.5, 3.5),
    (np.array([[0,0,0],[1,1,1],[0,1,0]]), 1, 1, 2.5, 3.5),
    (np.array([[0,0,0],[0,1,200]]), 1, 1, 2.5, 3.5),
    (np.array([[0,0,0],[0,1,3.1]]), 1, 1, 2.5, 3.5),
    (np.random.uniform(-20, 20, (100, 2)), 1, 1, 2.5, 3.5),
    (np.random.uniform(-20, 20, (100, 3)), 1, 1, 2.5, 3.5),
    ])
def test_potential(x, epsilon_lj, sigma_lj, switch_start_lj, cutoff_lj):
    general_params = (x.shape[1], x.shape[0], 0, 0, True)
    lj_params = (2.5, 3.5, tuple([1] * x.shape[0]), tuple([1] * x.shape[0]))
    potential1 = es.potentials.calc_potential(x, general_params, lj_params)
    # legacy potential implementaton, requires sigma, epsilon = 1
    # and a switch region of 2.5 to 3.5
    potential2 = es.potentials.lj_potential_total(x)
    np.testing.assert_allclose(potential1, potential2)
