import numpy as np
import ewald_summation as es
from itertools import product
from numba import jit

def initializer_3d(l_box, n_particles):
    grid=np.array( list (product(range(0,3), repeat=n_dim)))
    q_0 = l_box[0] * np.random.rand(N_particles,n_dim)
    masses = np.array([1.] * N_particles)
    charges = np.array([0.] * N_particles)
    v_0 = l_box[0] * np.zeros([N_particles,n_dim])
    return masses, charges, q_0, v_0 * masses[:, None]

def distances(q):
    # new implementation
    distance_vectors = q[:, None, :] - q[None, :, :]
    np.mod(distance_vectors, l_box, out=distance_vectors)
    mask = distance_vectors > np.divide(l_box, 2.)
    distance_vectors += mask * - l_box[0]
    return distance_vectors
    
    

N_particles = 30
density = 0.2
N_steps=1600
n_dim=3
Mole_fraction = 1
l_box = (N_particles/density) ** (1/n_dim) * np.ones(n_dim)
test_config = es.SimuConfig(n_dim=n_dim, l_box=l_box, n_particles=N_particles, n_steps=N_steps, timestep=0.000001, temp=0.1, PBC=True, neighbour=False)
test_md = es.MD(es.PhysWorld(), test_config, initializer_3d, es.step_runners.Langevin(damping=0.2))
#test_md.add_global_potential(HarmonicPotential(1.))
test_md.add_lennard_jones_potential()

test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()




bin_res = 100
r_max = np.min(l_box)
#bins = r_max/bin_res * np.arange(bin_res)
bin_width = r_max/bin_res
hist=np.zeros(bin_res-1)

#ideal_gas_qs=l_box[0] * np.random.rand(N_steps,N_particles,3)

 


clas=es.observables.WriteXyz(test_config,qs, "Na", "Cl")
clas.write_xyz()
clas2=es.observables.RadialDistributionFunction(test_config,qs)

g_r,bins = clas2.g_r()

import matplotlib.pyplot as plt

plt.plot(bins,g_r)
#plt.plot(bins_id,g_r_id)

plt.show()






