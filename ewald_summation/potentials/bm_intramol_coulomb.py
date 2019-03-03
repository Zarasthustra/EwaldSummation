import numpy as np
from .coulomb_real import CoulombReal
from .coulomb_correction import CoulombCorrection
from .water import Water
from timeit import default_timer as timer
#from intramol_template import _get_dv_mol_nPBC, _get_dv_mol_PBC

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
            ('HOH', [1, 2, 2], np.array([[0., 0., -0.064609], [0., -0.81649, 0.51275], [0., 0.81649, 0.51275]]), _water_bonds)
            ]
        self.mol_list = mol_list


#positions = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
positions = np.array([[0., 0., -0.064609], [0., -0.81649, 0.51275], [0., 0.81649, 0.51275]])
mol_list = [(0, [0, 1, 2])]
l_box = np.array([4, 4, 4])

q, particle_info = positions, np.asarray([1, 2, 2], dtype=np.uint8)
config = FakeConfig(q.shape[1], l_box, q.shape[0], particle_info, mol_list)
w = Water(config)
w.set_positions(q)
print(w.pot)
print(w.forces)
'''
a = CoulombReal(config, 1., 2.)
b = CoulombCorrection(config, 1.)
a.set_positions(q)
b.set_positions(q)
print("MULTI:", a.MULTI)
#for pair in a.pairs:
#    print(pair)
# to show the results and finish the jit compiling
print(a.pot)
print(b.pot)
print(a.forces)
print(b.forces)
'''
