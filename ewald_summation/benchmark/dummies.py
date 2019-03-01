import numpy as np
import ewald_summation as es

dummy_world = es.PhysWorld()
dummy_world.k_B = 1.
dummy_world.k_C = 1.
dummy_world.particle_types = [
            ('dummy_Ar', 1., 0., 1., 1.), #0
            ('dummy_+', 1., 1., 1., 1.), #1
            ('dummy_-', 1., -1., 1., 1.) #2
            ]
dummy_world.molecule_types = []
