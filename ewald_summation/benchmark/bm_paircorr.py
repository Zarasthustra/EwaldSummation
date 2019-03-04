import numpy as np
import ewald_summation as es
from .dummies import dummy_world


def random_initiliazer(box_size, n_particles):
    n_dim = len(box_size)
    q_0 = np.random.rand(n_particles, n_dim) * box_size
    v_0 = np.zeros((n_particles, n_dim))
    return q_0, v_0


q_opt= np.load("optimized.npy")
def langevin_init(box_size, n_particles):
    n_dim = len(box_size)
    q_0 = q_opt
    v_0 = np.zeros((n_particles, n_dim))
    return q_0, v_0

test_config = es.SimuConfig(l_box=(40.,40.,40.), PBC=True, neighbor_list=False, particle_info=np.asarray([0]*100), n_steps=50000, timestep=0.001, temp=1000)
test_md = es.MD(test_config, random_initiliazer, es.step_runners.MMC(step=0.5))
test_md.add_potential(es.potentials.LJ(test_config, switch_start=5, cutoff=6))


test_md.run_all()
print("simu_done")
qs = test_md.traj.get_qs()
np.save("optimized.npy",qs[-1])

steps =50000
pdb = es.observables.PdbWriter(test_config, 'Ar_Ar.pdb')
for i in range(int(steps/20)):
    pdb.write_frame(qs[i * 20])

Gr = es.observables.RadDistFunc(test_config, 150)
g_r, radii, name_list = Gr.calc_radial_dist(qs, 100, 50000, 5)

                                                                                                                                              

import matplotlib.pyplot as plt
plt.plot(radii, g_r[0], label=name_list[0])
plt.legend()
plt.show()
