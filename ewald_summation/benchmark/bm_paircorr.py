import numpy as np
import ewald_summation as es
from .dummies import dummy_world


def random_initiliazer(box_size, n_particles):
    n_dim = len(box_size)
    q_0 = np.random.rand(n_particles, n_dim) * box_size
    v_0 = np.zeros((n_particles, n_dim))
    return q_0, v_0
    
q_opt= np.load("optimized.npy")
def langevin_init_N_108_l_box_6(box_size, n_particles):
    n_dim = len(box_size)
    q_0 = q_opt
    v_0 = np.zeros((n_particles, n_dim))
    return q_0, v_0

test_config = es.SimuConfig(l_box=(6.,6.,6.), PBC=True, neighbor_list=False, particle_info=[0] * 108, n_steps=8000, timestep=0.001, temp=0.5, phys_world=dummy_world)
test_md = es.MD(test_config, random_initiliazer, es.step_runners.MMC(step=0.1))
test_md.add_potential(es.potentials.LJ(test_config, switch_start=2.5, cutoff=3.5))

test_md.run_all()
print("simu_done")
qs = test_md.traj.get_qs()
#np.save("optimized.npy",qs[-1])

Gr = es.observables.RadDistFunc(test_config, 40, [108,0,0])
g_r, radii = Gr.calc_radial_dist(qs, 500, 1)

import matplotlib.pyplot as plt
for _ in range(7):
    plt.plot(radii, g_r[_])
plt.show()
