import numpy as np
import ewald_summation as es
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class HarmonicPotential:
    def __init__(self, k):
        self.k = k

    def calc_force(self, q, sys_config):
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

def grid_initializer_2d(l_box, n_particles):
    x = np.linspace(0., l_box[0], 3, endpoint=False)
    y = np.linspace(0., l_box[1], 3, endpoint=False)
    masses = np.array([1.] * 9)
    charges = np.array([0.] * 9)
    q_0 = np.array(np.meshgrid(x, x)).T.reshape(9,2)
    v_0 = np.array([[0., 0.]] * 9)
    return masses, charges, q_0, v_0 * masses[:, None]

N_particles = 9
l_box = (8., 8.)
test_config = es.SimuConfig(n_dim=2, l_box=l_box, n_particles=N_particles, n_steps=3000, timestep=0.001, temp=100, PBC=True, neighbour=True)
test_md = es.MD(es.PhysWorld(), test_config, grid_initializer_2d, es.step_runners.Langevin(damping=0.1))
#test_md.add_global_potential(HarmonicPotential(1.))
test_md.add_lennard_jones_potential()
test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()
#plt.plot(qs[:, 0, 0]%3., qs[:, 0, 1]%3.)
#plt.plot(qs[:, 1, 0]%3., qs[:, 1, 1]%3.)
#plt.show()

import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=20, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'ro')
plt.setp(l, markersize=int(256/l_box[0]))
plt.setp(l, markerfacecolor='C0') 


plt.xlim(0., l_box[0])
plt.ylim(0., l_box[1])

x, y = np.zeros(N_particles), np.zeros(N_particles)

with writer.saving(fig, "movie_of_lj.mp4", 100):
    for _ in range(300):
        coord = qs[_ * 10] % l_box[0]
        for i in range(N_particles):
            x[i] = coord[i, 0]
            y[i] = coord[i, 1]
        l.set_data(x, y)
        writer.grab_frame()
