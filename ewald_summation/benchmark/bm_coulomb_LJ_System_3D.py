import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

def initializer_3d(l_box, n_particles):
    masses = np.array([1.] * N_particles)
    charges = np.array([0.] * N_particles)
    q_0 = l_box[0] * np.random.rand(n_particles,3)
    v_0 = np.array([[0., 0.,0.]] * n_particles)
    return masses, charges, q_0, v_0 * masses[:, None]

N_particles = 6
l_box = (20.,20.,20)
test_config = es.SimuConfig(n_dim=3, l_box=l_box, n_particles=N_particles, n_steps=3000, timestep=0.001, temp=100, PBC=True, neighbour=True)
test_md = es.MD(es.PhysWorld(), test_config, initializer_3d, es.step_runners.Langevin(damping=0.1))
#test_md.add_global_potential(HarmonicPotential(1.))
test_md.add_lennard_jones_potential()
#test_md.add_coulumb_potential()
test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()
#plt.plot(qs[:, 0, 0]%3., qs[:, 0, 1]%3.)
#plt.plot(qs[:, 1, 0]%3., qs[:, 1, 1]%3.)
#plt.show()



################## create Movie of configs #####################
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator

matplotlib.use("Agg")



def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)










  

import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=3, metadata=metadata)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
l, = plt.plot([], [], 'ro')
#plt.setp(l, markersize=30)
#plt.setp(l, markerfacecolor='C0') 

  

#x, y, z = np.zeros(10), np.zeros(10), np.zeros(10)





with writer.saving(fig, "movie_of_configs.mp4", 100):
    for _ in range(300):
        print(_/300 * 100, '%')
        ax.cla()
        q = qs[_*10] % l_box[0]
        for i in range(N_particles):
            x = q[i, 0]
            y = q[i, 1]
            z = q[i,2]
            r=1.
            (xs,ys,zs) = drawSphere(x,y,z,r)
            #print(xs,ys,zs)
            if i <= N_particles // 2 - 1:
                ax.plot_surface(xs, ys, zs, color="r")
            else:
                ax.plot_surface(xs, ys, zs, color="b")
            #l.set_data(x, y)
        ax.set_zlim(0,l_box[0])
        ax.set_ylim(0,l_box[1])
        ax.set_xlim(0,l_box[2])
        writer.grab_frame()


'''
=========================
3D surface (checkerboard)
=========================

Demonstrates plotting a 3D surface colored in a checkerboard pattern.
'''










                                     














