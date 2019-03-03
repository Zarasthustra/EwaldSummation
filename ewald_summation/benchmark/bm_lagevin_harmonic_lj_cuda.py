import ewald_summation as es
import matplotlib.pyplot as plt
import numpy as np

def initializer(n_particles, n_dim):
    # need to return a tuple of four vectors
    # masses, charges, q_0 and p_0
    masses = np.array([39.948] * n_particles)
    q_0 = np.arange(n_particles * n_dim).reshape(n_particles, n_dim)
    v_0 = np.zeros((n_particles, n_dim))
    particle_types = [0] * n_particles
    lj_mixing_conditions = tuple([(3.405, 3.405**6, 0.238)])
    n_particles_tpyes = 1
    return q_0, v_0 * masses[:, None], masses, particle_types, lj_mixing_conditions, n_particles_tpyes

cutoff = 3.405 * 3.5
damping = 0.01
n_steps = 10000
n_particles = 4
n_dim = 2

q_cuda, p_cuda = es.potentials.lagevin_harmonic_lj_cuda(*initializer(n_particles, n_dim), n_steps, cutoff, damping=damping)
plt.plot(q_cuda[:, 0, 0], q_cuda[:, 0, 1])
plt.plot(q_cuda[:, 1, 0], q_cuda[:, 1, 1])
plt.show()
