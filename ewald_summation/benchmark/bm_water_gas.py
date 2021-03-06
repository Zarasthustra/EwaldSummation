import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

def _grid(ns):
    xx = np.meshgrid(*[np.arange(0, n) for n in ns])
    X = np.vstack([v.reshape(-1) for v in xx]).T
    return X

water_shape = np.array([[0., 0., -0.064609], [0., -0.81649, 0.51275], [0., 0.81649, 0.51275]])

def _intializer_WaterBox(n):
    n_mol = n * n * n
    n_dim = 3
    # 3.104 is for correct liquid water density at 300 K
    l_box = 7. * np.array([n, n, n])
    grid=_grid([n, n, n])
    q_0 = 7. * grid
    #init_directions = []
    q = (q_0[:, None, :] + water_shape[None, :, :]).reshape(-1, 3)
    v = np.zeros((3 * n_mol, n_dim))
    mol_list = []
    for i in range(n_mol):
        mol_list.append((0, [3 * i, 3 * i + 1, 3 * i + 2]))
    particle_info = np.tile([1, 2, 2], n_mol)
    return q, v, particle_info, l_box, mol_list

q, v, particle_info, l_box, mol_list = _intializer_WaterBox(6)
steps = 2000
test_config = es.SimuConfig(l_box=l_box, PBC=True, particle_info=particle_info, mol_list=mol_list, n_steps=steps, timestep=0.01, temp=500)
init = lambda x,y: (q, v)
test_md = es.MD(test_config, init, es.step_runners.Langevin(damping=0.05))
test_md.add_potential(es.potentials.Water(test_config))
test_md.add_potential(es.potentials.LJ(test_config, switch_start=5., cutoff=7.))
test_md.add_potential(es.potentials.Coulomb(test_config))
#test_md.add_potential(HarmonicTrap(test_config, 100., [4., 4.]))
#test_md.add_potential(es.potentials.LJ(test_config, switch_start=2.5, cutoff=3.5))
test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()
pdb = es.observables.PdbWriter(test_config, 'water_gas.pdb')
for i in range(int(steps/10)):
    pdb.write_frame(qs[i * 10])
