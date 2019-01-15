import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

def StupidInitializer2(box_size, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.ones(10)
    charge_vector_a = [1] * round(10 * 0.5)
    charge_vector_b = [-1] * (10- round(10 * 0.5))
    charges = np.concatenate([charge_vector_a,charge_vector_b])
    q_0 = np.random.rand(10,3)
    v_0 = np.random.rand(10,3)
    return masses, charges, q_0, v_0 * masses[:, None]
    
test_config = es.SimuConfig(n_dim=3, l_box=(10.,10.,10.), PBC=True,n_particles=10, n_steps=10000, timestep=0.001, temp=300)
test_md = es.MD(es.PhysWorld(), test_config, StupidInitializer2, es.step_runners.Langevin(damping=0.))
test_md.add_coulumb_potential()
test_md.add_lennard_jones_potential()
test_md.run_all()
print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()
print('trajectory shape', qs.shape)
#plt.plot(qs[:, 0, 0], qs[:, 0, 1])
#plt.plot(qs[:, 1, 0], qs[:, 1, 1])
#plt.show()

################### create Movie of configs #####################
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=20, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'ro')
plt.setp(l, markersize=30)
plt.setp(l, markerfacecolor='C0') 















