import numpy as np
import ewald_summation as es
import matplotlib.pyplot as plt
import math
from numba import njit

x = np.random.random((2, 3))

test_config = es.SimuConfig(n_dim=3,
                            l_box=(2., 2., 2),
                            n_particles=2,
                            n_steps=10000,
                            timestep=0.001,
                            temp=300,
                            sigma_lj = [1] * 2,
                            epsilon_lj = [1] * 2,
                            PBC = True,
                            )


test = es.potentials.CalcPotentialClass(test_config)
pot = test(x)
print(pot)
