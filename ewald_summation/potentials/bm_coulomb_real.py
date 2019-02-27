import numpy as np
#from coulomb_real import CoulombFake as CoulombReal
from coulomb_real import CoulombReal
from timeit import default_timer as timer

class FakeWorld:
    def __init__(self):
        self.k_C = 1.

class FakeConfig:
    def __init__(self, n_dim, l_box, n_particles, particle_info):
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

def _intializer_NaCl(n):
    n_particles = n * n * n
    n_dim = 3
    l_box = 4. * np.array([n, n, n])
    grid=_grid([n, n, n])
    q = 4. * grid
    particle_info = grid.sum(axis=1)%2+3
    return q, particle_info, l_box

q, particle_info, l_box = _intializer_NaCl(12) # 12*12*12 = 1728 particles
config = FakeConfig(q.shape[1], l_box, q.shape[0], particle_info)
a = CoulombReal(config, 1., 8.)
a.set_positions(q)
print("MULTI:", a.MULTI)
#for pair in a.pairs:
#    print(pair)
# to show the results and finish the jit compiling
#print(a.pot)
#print(len(a.forces))
print()
# benchmark
pasofdp = None
start = timer()
for i in range(20):
    a.set_positions(q)
    #print(a.pot)
    pasofdp = a.pot
stop = timer()
print('\nper step:', (stop - start)/20)
