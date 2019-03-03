import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt
from .dummies import dummy_world
from ewald_summation.observables.pdb_writer import PdbWriter

@es.potentials.globalx
def HarmonicTrap(config, k, center):
    center = np.asarray(center)
    def pot_func(q):
        return k * np.square(q - center).sum()
    def force_func(q):
        return -2. * k * (q - center)
    return pot_func, force_func

def StupidInitializer3(box_size, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    q_0 = np.array([[0., 0., 0.], [0., 1., 0.]]) + box_size / 2
    #v_0 = np.array([[0.5, 0.866], [-0.8, 0.6]])
    v_0 = np.array([[0., 0., 0.], [0., 0., 0.]])
    return q_0, v_0

test_config = es.SimuConfig(l_box=(8., 8., 8.), PBC=True, particle_info=[0, 0], n_steps=10000, timestep=0.001, temp=30, phys_world=dummy_world)
q, _ = StupidInitializer3(np.array([8., 8., 8.]), 2)
pdb = PdbWriter(test_config, 'test.pdb')
pdb.write_frame(q)
#test_md = es.MD(test_config, StupidInitializer3, es.step_runners.Langevin(damping=0.01))
#test_md.add_potential(HarmonicTrap(test_config, 100., [4., 4.]))
#test_md.add_potential(es.potentials.LJ(test_config, switch_start=2.5, cutoff=3.5))
#test_md.run_all()
#print(test_md.traj.get_qs())
#qs = test_md.traj.get_qs() % 8.
#plt.plot(qs[:, 0, 0], qs[:, 0, 1])
#plt.plot(qs[:, 1, 0], qs[:, 1, 1])
#plt.show()