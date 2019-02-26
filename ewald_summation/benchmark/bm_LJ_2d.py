import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

class HarmonicPotential:
    def __init__(self, k):
        self.k = k
        self.iter = 0

    def calc_force(self, q):
        self.iter += 1
        # if self.iter % 1 == 0:
        #     print(-2. * self.k * q)
        return -2. * self.k * q
    # TODO: calc_potential(q, sys_config)

def StupidInitializer2(box_size, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.array([1., 1.])
    charges = np.array([0., 0.])
    q_0 = np.array([[0., 0.], [0., 1.]])
    v_0 = np.array([[0.5, 0.866], [-0.8, 0.6]])
    return masses, charges, q_0, v_0 * masses[:, None]

test_config = es.SimuConfig(n_dim = 2,
                            l_box = (2., 2.),
                             n_particles = 2,
                             n_steps = 20000,
                             timestep = 0.001,
                             temp = 300,
                             lj_flag = True,
                             PBC = False,
                             )

test_md = es.MD(es.PhysWorld(), test_config, StupidInitializer2, es.step_runners.Langevin(damping=0.1))
test_md.add_global_potential(HarmonicPotential(1))
test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()
plt.plot(qs[:, 0, 0], qs[:, 0, 1])
plt.plot(qs[:, 1, 0], qs[:, 1, 1])
plt.show()
