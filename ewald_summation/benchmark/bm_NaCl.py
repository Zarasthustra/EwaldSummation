import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt

def _grid(ns):
    xx = np.meshgrid(*[np.arange(0, n) for n in ns])
    X = np.vstack([v.reshape(-1) for v in xx]).T
    return X

def _intializer_NaCl(n):
    n_particles = n * n * n
    n_dim = 3
    l_box = 10. * np.array([2 * n, n, n])
    grid=_grid([n, n, n])
    q = 10. * grid
    v = np.zeros((n_particles, n_dim))
    particle_info = grid.sum(axis=1)%2+3
    return q, v, particle_info, l_box

q, v, particle_info, l_box = _intializer_NaCl(4)
steps = 1000
test_config = es.SimuConfig(l_box=l_box, PBC=True, particle_info=particle_info[1:-1], n_steps=steps, timestep=200, temp=2000)
init = lambda x,y: (q[1:-1], v[1:-1])
test_md = es.MD(test_config, init, es.step_runners.Langevin(damping=0.1))
#test_md.add_potential(es.potentials.Water(test_config))
test_md.add_potential(es.potentials.LJ(test_config, switch_start=5., cutoff=7.))
test_md.add_potential(es.potentials.Coulomb(test_config))
#test_md.add_potential(HarmonicTrap(test_config, 100., [4., 4.]))
#test_md.add_potential(es.potentials.LJ(test_config, switch_start=2.5, cutoff=3.5))
test_md.run_all()
#print(test_md.traj.get_qs())
qs = test_md.traj.get_qs()
pdb = es.observables.PdbWriter(test_config, 'NaCl7.pdb', put_in_box=True)
for i in range(steps//8):
    pdb.write_frame(qs[i*8])
