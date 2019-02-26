import numpy as np

class SimuConfig:
    def __init__(self, n_dim=3, l_box=[1.,1.,1.], PBC=False, n_particles=1, n_steps=1000,\
     timestep=1e-10, neighbour=False, l_cell=1, temp=1, mole_fraction=0.5, \
     switch_start_lj=2.5, cutoff_lj=3.5, sigma_lj=1, epsilon_lj=1, phys_world=None):
        # TODO: sanity checks
        self.l_box = l_box
        self.PBC = PBC
        self.n_steps = n_steps
        self.n_particles = n_particles
        self.n_dim = n_dim
        # move these initializations to initializer
        self.masses = None
        self.charges = None
        self.temp = temp
        self.timestep = timestep
        self.neighbour = neighbour
        self.l_cell = l_cell
        # Temp
        self.mole_fraction = mole_fraction       
        self.switch_start_lj = switch_start_lj
        self.cutoff_lj = cutoff_lj
        self.sigma_lj = sigma_lj
        self.epsilon_lj = epsilon_lj
        
        # physical world
        if phys_world is None:
            self.phys_world = PhysWorld()
        else:
            self.phys_world = phys_world
        self.particle_types = self.phys_world.particle_types
        self.molecule_types = self.phys_world.molecule_types

class PhysWorld:
    def __init__(self):
        # physical constants
        self.N_A = 6.02214086e23
        # k_B = 1.38065e-23
        self.k_B = 0.00198720360
        self.k_C = 332.0637128

        # (mass, charge, lj_sigma, lj_epsilon)
        # in unit g/mol, e, Angstrom, kcal/mol
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

        # here we use a simplied version from MDynaMix
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
