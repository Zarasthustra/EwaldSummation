import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

n_particles = 200
n_dim = 3
l_box = [7] * n_dim

def StupidInitializer2(box_size, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.ones(n_particles)
    charges = np.zeros(n_particles)
    q_0 = np.random.rand(n_particles,n_dim)*l_box
    v_0 = np.zeros([n_particles,n_dim])
    return masses, charges, q_0, v_0 * masses[:, None]

test_config = es.SimuConfig(n_dim=n_dim, l_box=l_box, n_particles=n_particles,
                            n_steps=10000, timestep=0.001, temp=0.6, start_sampling=10, sampling_rate=4, PBC=True)
test_md = es.MD(es.PhysWorld(), test_config, StupidInitializer2,
                es.step_runners.MMC())

test_md.add_lennard_jones_potential()
test_md.run_all()






