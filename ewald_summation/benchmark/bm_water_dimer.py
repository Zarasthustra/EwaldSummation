import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

def StupidInitializerWater(box_size, n_particles):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    q_0 = np.array([[0., 0., -0.064609], [0., -0.81649, 0.51275], [0., 0.81649, 0.51275]])
    q = np.vstack((q_0, q_0 + box_size / 5 * 2))
    #v_0 = np.array([[0.5, 0.866], [-0.8, 0.6]])
    v_0 = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    v = np.vstack((v_0, v_0))
    return q, v

particle_info=[1, 2, 2, 1, 2, 2]
mol_list = [(0, [0, 1, 2]), (0, [3, 4, 5])]

test_config = es.SimuConfig(l_box=(8., 8., 8.), PBC=True, particle_info=particle_info, mol_list=mol_list, n_steps=1000, timestep=0.01, temp=500)
test_md = es.MD(test_config, StupidInitializerWater, es.step_runners.Langevin(damping=0.05))
test_md.add_potential(es.potentials.Water(test_config))
test_md.add_potential(es.potentials.LJ(test_config, switch_start=5., cutoff=7.))
test_md.add_potential(es.potentials.Coulomb(test_config))
#test_md.add_potential(HarmonicTrap(test_config, 100., [4., 4.]))
#test_md.add_potential(es.potentials.LJ(test_config, switch_start=2.5, cutoff=3.5))
test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()
pdb = es.observables.PdbWriter(test_config, 'water_dimer.pdb')
for i in range(100):
    pdb.write_frame(qs[i * 10])
