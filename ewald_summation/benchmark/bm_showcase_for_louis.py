import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt


n_particles=20
n_dim = 3
density = 0.8
l_box = [3,3,3]


def StupidInitializer2(box_size, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.ones(n_particles)
    charges = np.ones(n_particles)
    charges[len(charges)//2:] = -1
    q_0 = np.random.rand(n_particles,n_dim)
    v_0 = np.zeros(n_particles)
    return masses, charges, q_0, v_0 * masses[:, None]

test_config = es.SimuConfig(n_dim=n_dim, l_box=[3,3,3], n_particles=n_particles,
                            n_steps=10000, timestep=0.001, temp=2, PBC=True, l_cell=1., neighbour=False)
test_md = es.MD(es.PhysWorld(), test_config, StupidInitializer2,
                es.step_runners.MMC())

test_md.add_lennard_jones_potential()
test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()

clas =es.observables.WriteXyz(test_config,qs, "Na", "Cl")
clas.write_xyz()

clas =es.observables.RadialDistributionFunction(test_config, qs)
g,b = clas.g_r()

import matplotlib.pyplot as plt

plt.plot(b,g)
plt.show()



