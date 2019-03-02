import numpy as np
import math
from numba import njit
from .intramol_template import intramol
from .bond import bond_pot, bond_forces

@intramol
def Water(config):
    # precalc
    n_dim = config.n_dim

    @njit
    #def pot_func(pair, dv):
    def pot_func(mol_descr, dv):
        # assert mol_descr[0] == 'water', 'only water molecule supported.'
        pot = 0.
        # to simplify, assume only have water molecules
        for bond in mol_descr[3]:
            pot += bond_pot(bond, dv)
        return pot

    @njit
    #def force_func(pair, dv):
    def force_func(mol_descr, dv):
        forces = np.zeros((len(mol_descr[1]), n_dim))
        for bond in mol_descr[3]:
            i, j = bond[1], bond[2]
            force_i, force_j = bond_forces(bond, dv)
            forces[i] += force_i
            forces[j] += force_j
        return forces

    return pot_func, force_func 
