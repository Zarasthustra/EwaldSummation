import numpy as np
import math
from numba import njit
from .intramol_template import intramol

@intramol
def CoulombCorrection(config, alpha):
    # precalc
    n_dim = config.n_dim
    assert n_dim == 3, "For other dimensions not implemented."
    assert config.PBC, "Ewald sum only meaningful for periodic systems."
    # exactly the opposite of real part, to cancel the intramolecular 
    # Coulomb calc.
    prefactor = -config.phys_world.k_C / 2.
    n_alpha_sq = -alpha * alpha
    alpha_coeff = 2 * alpha / math.sqrt(np.pi)
    particle_types = config.particle_types
    n_types = len(particle_types)
    charge = np.empty(n_types)
    charge_product = np.empty((n_types, n_types))
    for i in range(n_types):
        charge[i] = particle_types[i][2]
    for i in range(n_types):
        for j in range(n_types):
            charge_product[i, j] = charge[i] * charge[j]
    
    @njit
    #def pot_func(pair, dv):
    def pot_func(mol_descr, dv):
        pot = 0.
        # to simplify, assume the bond list consist of all intramolecular pairs.
        # as now we only have water molecule, it's OK.
        for bond in mol_descr[3]:
            i, j = bond[1], bond[2] # here i, j are the atom indeces in the mol
            distance = dv[1][i, j]
            pot += 2 * prefactor * charge_product[mol_descr[1][i], mol_descr[1][j]] *\
                   math.erfc(alpha * distance) / distance
        return pot
    
    @njit
    #def force_func(pair, dv):
    def force_func(mol_descr, dv):
        forces = np.zeros((len(mol_descr[1]), n_dim))
        for bond in mol_descr[3]:
            i, j = bond[1], bond[2]
            distance_vector, distance = dv[0][i, j], dv[1][i, j]
            distance_squared = distance * distance
            real_part1 = math.erfc(alpha * distance) / distance + \
                            alpha_coeff * math.exp(n_alpha_sq * distance_squared)
            real_part = prefactor * charge_product[mol_descr[1][i], mol_descr[1][j]] *\
                        (real_part1 / distance_squared) * distance_vector
            forces[i] += real_part
            forces[j] -= real_part
        return forces
    
    return pot_func, force_func 
