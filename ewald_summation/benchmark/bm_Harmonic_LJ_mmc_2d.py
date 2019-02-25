import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

class HarmonicPotential:
    def __init__(self, k):
        self.k = k

    def calc_force(self, q):
        return -2. * self.k * q

    def calc_potential(self, q):
        return self.k * np.square(q).sum()

def StupidInitializer2(box_size, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.array([1., 1.])
    charges = np.array([0., 0.])
    q_0 = np.array([[0., 0.], [0., 1.]])
    v_0 = np.array([[0.5, 0.866], [-0.8, 0.6]])
    return masses, charges, q_0, v_0 * masses[:, None]

test_config = es.SimuConfig(n_dim = 2,
                            l_box = [1., 1.],
                            n_particles = 2,
                            n_steps = 10000,
                            timestep = 0.001,
                            temp = 100,
                            lj_flag = True,
                            )

test_md = es.MD(es.PhysWorld(), test_config, StupidInitializer2, es.step_runners.MMC(step=0.05))
test_md.add_global_potential(HarmonicPotential(10000.))
test_md.run_all()
qs = test_md.traj.get_qs()
plt.plot(qs[:, 0, 0], qs[:, 0, 1])
plt.plot(qs[:, 1, 0], qs[:, 1, 1])
plt.show()