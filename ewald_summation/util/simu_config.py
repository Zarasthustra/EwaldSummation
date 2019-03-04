import numpy as np

class SimuConfig:
    def __init__(self, l_box=[1.,1.,1.], PBC=True, phys_world=None, particle_info=[0],
                 mol_list=[], neighbor_list=True, n_steps=1000, timestep=1e-10, temp=300.):
        # TODO: sanity checks
        self.l_box = np.asarray(l_box)
        self.n_dim = self.l_box.shape[0]
        self.neighbor_list = neighbor_list
        self.PBC = PBC
        if not PBC:
            self.neighbor_list = False
        # physical world
        if phys_world is None:
            self.phys_world = PhysWorld()
        else:
            self.phys_world = phys_world
        self.particle_types = self.phys_world.particle_types
        self.molecule_types = self.phys_world.molecule_types
        # particles
        self.particle_info = np.asarray(particle_info, dtype=np.uint8)
        self.mol_list = mol_list
        self.n_particles = self.particle_info.shape[0]
        self.masses = np.empty(self.n_particles)
        for i in range(self.n_particles):
            self.masses[i] = self.particle_types[self.particle_info[i]][1]
        self.n_steps = n_steps
        self.timestep = timestep# / 2390.057 # to fix the problem of kcal unit
        self.temp = temp #* 2390.057

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
            #(0, 1, 2, 1.633, 164.30,     0.,    0.)
            (0, 1, 2, 1.633, 354.04,     0.,    0.)
            ]

        self.molecule_types = [
            # (name, list of particles, initial positions, bonds)
            ('HOH', [1, 2, 2], np.array([[0., 0., -0.064609], [0., -0.81649, 0.51275], [0., 0.81649, 0.51275]]), _water_bonds)
]