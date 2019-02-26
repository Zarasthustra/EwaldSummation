import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt
from itertools import product
                             
n_particles = 200
n_dim = 3
l_box = [7] * n_dim

def StupidInitializer2(box_size, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.ones(n_particles)
    charges = np.append(np.ones(30),-np.ones(30))
    charges = np.append(charges, np.zeros(40))
    q_0 = np.random.rand(n_particles,n_dim)*l_box
    v_0 = np.zeros([n_particles,n_dim])
    return masses, charges, q_0, v_0 * masses[:, None]

test_config = es.SimuConfig(n_dim=n_dim, l_box=l_box, n_particles=n_particles,
                            n_steps= 100, p_kinds= [200,0,0], timestep=10e-3, temp=0.8, start_sampling=1, sampling_rate=40, PBC=True, neighbour = False, l_cell = 2.5)
test_md = es.MD(es.PhysWorld(), test_config, StupidInitializer2,
                es.step_runners.MMC())

test_md.add_lennard_jones_potential()
test_md.run_all()
qs = test_md.traj.get_qs()


MakeXYZ = es.observables.WriteXyz(test_config, qs, "Ar", "Ar")
MakeXYZ.write_xyz()
