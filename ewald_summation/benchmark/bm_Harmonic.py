import numpy as np
import ewald_summation as es
"""from MD import MD
from SimuConfig import SimuConfig
from PhysicalWorld import PhysicalWorld
from Integrators import Langevin
"""
import matplotlib.pyplot as plt

class HarmonicPotential:
    def __init__(self, k):
        self.k = k

    def calc_force(self, q, sys_config):
        return -2. * self.k * q
    # TODO: calc_potential(q, sys_config)

def StupidInitializer2(l_box, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.array([1., 1.])
    charges = np.array([0., 0.])
    q_0 = np.array([0., 1.])[:, None]
    v_0 = np.array([1., -0.5])[:, None]
    return masses, charges, q_0, v_0 * masses[:, None]

test_config = es.SimuConfig(n_dim=1, l_box=[1.], n_particles=2, n_steps=10000, timestep=0.001, temp=300)
test_md = es.MD(es.PhysWorld(), test_config, StupidInitializer2, es.step_runners.Langevin(damping=0.))
test_md.add_global_potential(HarmonicPotential(1.))
test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()
plt.plot(test_md.traj.ts, qs[:, 0, :])
plt.plot(test_md.traj.ts, qs[:, 1, :])
plt.show()
