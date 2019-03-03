import numpy as np
#from .coulomb_real import CoulombReal
#from .coulomb_reciprocal import CoulombReciprocal
from .coulomb_combined import Coulomb
from timeit import default_timer as timer
import matplotlib.pyplot as plt

class FakeWorld:
    def __init__(self):
        self.k_C = 1.

class FakeConfig:
    def __init__(self, n_dim, l_box, n_particles, particle_info, mol_list):
        self.n_dim = n_dim
        self.PBC = True
        self.l_box = l_box
        self.n_particles = n_particles
        self.particle_info = particle_info
        self.phys_world = FakeWorld()
        self.particle_types = [
            # Argon parameter from Rowley, Nicholson and Parsonage, 1975
            ('Ar', 39.948, 0., 3.405, 0.238), #0
            # data below are from software MDynaMix
            # http://www.fos.su.se/~sasha/mdynamix/Examples/nacl.html
            # water parameter finally from SPC/F model
            # K TOUKAN AND A.RAHMAN, PHYS. REV. B Vol. 31(2) 2643 (1985)
            ('OW', 15.999, -0.82, 3.166, 0.155), #1
            ('HW', 1.0079, 0.41, 0., 0.), #2
            # NaCl ori ref: https://doi.org/10.1063/1.466363
            ('Na+', 22.990, 1., 2.35, 0.130), #3
            ('Cl-', 35.453, -1., 4.40, 0.100) #4
            ]
        _water_bonds = [
            # (bond_type, index of par1, index of par2, EqnLen r_0, Bond k, Morse D, Morse rho)
            # bond_type = 0 (harmonic) or 1 (Morse)
            # units: r_0 (Angstrom), k (kcal/mol/A^2), D (kcal/mol), rho (A^{-1})
            (1, 0, 1, 1.000,     0., 101.90, 2.566),
            (1, 0, 2, 1.000,     0., 101.90, 2.566),
            (0, 1, 2, 1.633, 164.30,     0.,    0.)
            ]

        self.molecule_types = [
            # (name, list of particles, initial positions, bonds)
            ('water', [1, 2, 2], np.array([[0., 0., -0.064609], [0., -0.81649, 0.51275], [0., 0.81649, 0.51275]]), _water_bonds)
            ]
        self.mol_list = mol_list

'''
a = lj_pairwise('config*', 2, 3)
print(a)

a = lj_pairwise(FakeConfig(), 6., 8.)
print(a.pot_func((1, 0), (np.array([3., 4., 0.]), 5.)))
print(a.force_func((1, 0), (np.array([3., 0., 0.]), 3.)))
'''
def _grid(ns):
    xx = np.meshgrid(*[np.arange(0, n) for n in ns])
    X = np.vstack([v.reshape(-1) for v in xx]).T
    return X

q, particle_info, l_box = np.array([[0., 0., 0.], [0.7, 0., 0.]]), [3, 4], np.array([3., 3., 3.])
config = FakeConfig(q.shape[1], l_box, q.shape[0], particle_info, [])

#accuracy = 1e-8
## ratio_real_rec = 5.3 #for 1e-6 accuracy
#ratio_real_rec = 5.5 # for 1e-8 accuracy
#V = config.l_box[0] * config.l_box[1] * config.l_box[2]
#alpha = ratio_real_rec * np.sqrt(np.pi) * (config.n_particles / V / V) ** (1/6)
#REAL_CUTOFF = np.sqrt(-np.log(accuracy)) / alpha
#REC_RESO = int(np.ceil(np.sqrt(-np.log(accuracy)) * 2 * alpha))

#a = CoulombReal(config, alpha, REAL_CUTOFF)
#b = CoulombReciprocal(config, alpha, REC_RESO)
a, b, c = Coulomb(config, accuracy=1e-8)
a.set_positions(q)
b.set_positions(q)
c.set_positions(q)
#for pair in a.pairs:
#    print(pair)
# to show the results and finish the jit compiling
print('MULTI:', a.MULTI)
#print('real cutoff:', REAL_CUTOFF)
#print('reciprocal reso:', REC_RESO)
print(a.forces)
print(b.forces)
print(c.forces)
xs = np.arange(1., 2., 0.01)
pots = np.zeros(100)
forces = np.zeros(100)
force_inte = np.zeros(100)
step = 0
for i in range(100):
    q[1, 0] = xs[i]
    step += 1
    a.set_positions(q)
    b.set_positions(q)
    pots[i] = a.pot + b.pot
    forces[i] = (a.forces + b.forces)[1, 0]
force_inte[0] = pots[0]
for i in range(1, 100):
    force_inte[i] = force_inte[i - 1] + 0.01 * -forces[i - 1]

plt.plot(xs, pots, label='pot')
plt.plot(xs, force_inte, label='force_inte')
plt.legend()
plt.show()
