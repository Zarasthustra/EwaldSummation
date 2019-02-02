import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt


n_particles=20
n_dim = 3
density = 0.8
l_box = (n_particles/density)**(1/n_dim) * np.array([1,1,1])


def StupidInitializer2(box_size, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.ones(n_particles)
    charges = np.zeros(n_particles)
    q_0 = np.random.rand(n_particles,n_dim)
    v_0 = np.zeros(n_particles)
    return masses, charges, q_0, v_0 * masses[:, None]

test_config = es.SimuConfig(n_dim=n_dim, l_box=l_box, n_particles=n_particles,
                            n_steps=10000, timestep=0.001, temp=2, PBC=True, l_cell=1, neighbour=False)
test_md = es.MD(es.PhysWorld(), test_config, StupidInitializer2,
                es.step_runners.MMC())

test_md.add_lennard_jones_potential()
test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()
print('trajectory shape', qs.shape)
plt.plot(qs[:, 0, 0], qs[:, 0, 1])
plt.plot(qs[:, 1, 0], qs[:, 1, 1])
plt.show()
