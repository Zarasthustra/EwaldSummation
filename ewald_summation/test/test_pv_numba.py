import numpy as np
import pytest
from itertools import product
import ewald_summation as es
from numba import njit
from ewald_summation.distances.pw_numba import PWNumba
from math import exp

def _intializer_NaCl(n):
    n_particles = n * n * n
    n_dim = 3
    l_box=np.array([n, n, n])
    grid=np.array(list(product(range(0,n), repeat=n_dim)))
    q = 1. * grid
    charge_vector = grid.sum(axis=1)%2*2-1
    return q, charge_vector, l_box

'''
# test for multi images
q, charge_vector, l_box = _intializer_NaCl(2)
simu_config = es.SimuConfig(n_dim=q.shape[1], n_particles=q.shape[0], l_box=l_box, PBC=True)
# simu_config.charges = charge_vector
distance_vectors = es.distances.DVNumba(simu_config, 2.)
distance_vectors.set_positions(q)
# print(distance_vectors.get_particles_in_cutoff(0))
print("MULTI:", distance_vectors.MULTI)
for pair in distance_vectors.pairs:
    print(pair)
'''

'''
@njit
def func(pair, dv):
    return dv[1]
'''
def func(Coeff):
    @njit
    def pot_func(pair, dv):
        return Coeff * exp(-dv[1])
    @njit
    def force_func(pair, dv):
        return dv[0] * Coeff
    return pot_func, force_func

q, charge_vector, l_box = _intializer_NaCl(3)
simu_config = es.SimuConfig(n_dim=q.shape[1], n_particles=q.shape[0], l_box=l_box, PBC=True)
# simu_config.charges = charge_vector
distance_vectors = PWNumba(simu_config, func(2.), 1.)
distance_vectors.set_positions(q)
# print(distance_vectors.get_particles_in_cutoff(0))
print("MULTI:", distance_vectors.MULTI)
#for pair in distance_vectors.pairs:
#    print(pair)
print(distance_vectors.pot)
print(distance_vectors.forces)
print()
print(distance_vectors.pots)
print(distance_vectors.forces)
print()
distance_vectors.set_positions(q)
print(distance_vectors.pots)
print(distance_vectors.forces)
#TODO: this file is NOT really a test!
