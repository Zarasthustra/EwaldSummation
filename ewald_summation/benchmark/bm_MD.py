import numpy as np
import ewald_summation as es

"""
from MD import MD
from SimuConfig import SimuConfig
from PhysicalWorld import PhysicalWorld
from Integrators import Langevin
"""


class NullPotential:
    def calc_force(self, q, sys_config):
        return np.zeros((sys_config.n_particles, sys_config.n_dim))
    # TODO: calc_potential(q, sys_config)


def StupidInitializer(l_box, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.array([1., 1.])
    charges = np.array([0., 0.])
    q_0 = np.array([0., 1.])[:, None]
    v_0 = np.array([1., -0.5])[:, None]
    return masses, charges, q_0, v_0 * masses[:, None]


test_config = es.SimuConfig(n_dim=1, l_box=[1.], n_particles=2, n_steps=1000, timestep=0.001, temp=300)
test_md = es.MD(es.PhysWorld(), test_config, StupidInitializer, es.step_runners.Langevin(damping=0.))
test_md.add_global_potential(NullPotential())
test_md.run_all()
print(test_md.traj.get_qs())
