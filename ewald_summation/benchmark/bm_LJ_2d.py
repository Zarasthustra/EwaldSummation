import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt
from .dummies import dummy_world


def random_initiliazer(box_size, n_particles):
    n_dim = len(box_size)
    q_0 = np.random.uniform(0.1, 7, (n_particles, n_dim))
    v_0 = np.zeros((n_particles, n_dim))
    return q_0, v_0


test_config = es.SimuConfig(l_box=(8., 8.), PBC=True, particle_info=[0] * 40, n_steps=2000, timestep=0.001, temp=30, phys_world=dummy_world)
test_md = es.MD(test_config, random_initiliazer, es.step_runners.Langevin(damping=0.01))
test_md.add_potential(es.potentials.LJ(test_config, switch_start=2.5, cutoff=3.5))
test_md.run_all()


qs = test_md.traj.get_qs() % 8.
plt.plot(qs[:, 0, 0], qs[:, 0, 1])
plt.plot(qs[:, 1, 0], qs[:, 1, 1])
plt.show()
