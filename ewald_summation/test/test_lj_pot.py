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

@pytest.mark.parametrize('x', [
    (np.array([[0, 0, 0], [1, 0, 0]])),
    (np.array([[0, 0, 1],[1, 0, 0], [0, 1, 0]])),
    # (np.array([[0,0,0],[0,1,200]]), 1, 1, 2.5, 3.5),
    # (np.array([[0,0,0],[0,1,3.1]]), 1, 1, 2.5, 3.5),
    (np.random.uniform(-20, 20, (100, 2))),
    # (np.random.uniform(-20, 20, (100, 3))),
    ])
def test_potential(x):
    test_config = es.SimuConfig(l_box=(8., 8., 8.),
                                PBC=False,
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
