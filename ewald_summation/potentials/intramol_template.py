import numpy as np
from numba import jit, njit

def intramol(intramol_pot):
    """Decorator to transform a intramolecular potenial/force function into 
    a well-defined (requires only particle positions) potenial class. With auto
    detection of neighbor list or multiple image requirement.
    
    Usage:
    @intramol
    def Blahblah(config, <parameters>...):
        # define intramol potential function and intramol 
        # force function under config and parameters.
        # Both functions should accept input as 
        # molecule_descr, (dvs, dists)
        # molecule_descr: one entry defined in phys_world.molecule_types
        # dvs: distance vectors q_i - q_j for all intramol i, j pair
        # dists: distances |q_i - q_j| for all intramol i, j pair
        return intramol_pot, intramol_force
    
    The resulting class will have:
    1. set_positions(q): method
    2. pot: property
    3. forces: property
    """
    def mk_cls(config, *args, **kwargs):
        return IntraMol(config, intramol_pot(config, *args, **kwargs))
    return mk_cls

class IntraMol:
    """Class to define a pair-wise potential/force, with auto selction between
    nearest-neighbor and multi-image conventions.
    
    Init:
    config: MD system configuration
    func: a tuple of pair-wise potential and force functions (and cutoff(optional))
    cutoff (optional): if cutoff is not provided above, then put it here
    
    Usage:
    First set new positions of system particle q.
    Then potentials and forces can be retrieved from properties pot, pots and forces.
    """

    def __init__(self, config, func):
        self.PBC = config.PBC
        self.l_box = config.l_box
        self.n_particles = config.n_particles
        self.n_dim = config.n_dim
        self.molecule_types = config.molecule_types
        self.mol_list = config.mol_list
        self.positions = np.zeros((self.n_particles, self.n_dim))
        self.pot_func = func[0]
        self.force_func = func[1]

        self._pot = None
        self._forces = None

    def set_positions(self, new_positions):
        self.positions[:, :] = new_positions
        if(self.PBC):
            self.positions %= self.l_box
        self._pot_recalc = True
        self._force_recalc = True
    
    @property
    def pot(self):
        if(self._pot_recalc):
            if(self.PBC):
                self._pot = _pot_mol(self.mol_list, self.molecule_types,
                                     self.positions, self.l_box, self.pot_func,
                                     _get_dv_mol_PBC)
            else:
                self._pot = _pot_mol(self.mol_list, self.molecule_types,
                                     self.positions, self.l_box, self.pot_func,
                                     _get_dv_mol_nPBC)
            self._pot_recalc = False
        return self._pot
    
    @property
    def forces(self):
        if(self._force_recalc):
            if(self.PBC):
                self._forces = _forces_mol(self.mol_list, self.molecule_types,
                                     self.positions, self.l_box, self.force_func,
                                     _get_dv_mol_PBC)
            else:
                self._forces = _forces_mol(self.mol_list, self.molecule_types,
                                     self.positions, self.l_box, self.force_func,
                                     _get_dv_mol_nPBC)
            self._force_recalc = False
        return self._forces

@jit
def _pot_mol(mol_list, molecule_types, positions, l_box, pot_func, get_dv_mol):
    pot_sum = 0.
    for mol in mol_list:
        # mol = (mol_type, [indeces of particles])
        pot_sum += pot_func(molecule_types[mol[0]], get_dv_mol(positions, mol, l_box))
    return pot_sum

@jit
def _forces_mol(mol_list, molecule_types, positions, l_box, force_func, get_dv_mol):
    n_particles = np.shape(positions)[0]
    n_dim = np.shape(positions)[1]
    forces_sum = np.zeros((n_particles, n_dim))
    for mol in mol_list:
        # mol = (mol_type, [indeces of particles])
        mol_forces = force_func(molecule_types[mol[0]], get_dv_mol(positions, mol, l_box))
        n_par = np.shape(mol[1])[0]
        for i in range(n_par):
            forces_sum[mol[1][i]] += mol_forces[i]
    return forces_sum

@jit
def _get_dv_mol_nPBC(positions, mol, l_box):
    n_dim = np.shape(positions)[1]
    n_par = np.shape(mol[1])[0]
    mol_posi = np.empty((n_par, n_dim))
    for i in range(n_par):
        mol_posi[i] = positions[mol[1][i]]
    dvs = mol_posi[:, None, :] - mol_posi[None, :, :]
    dists = np.sqrt(np.square(dvs).sum(axis=-1))
    return dvs, dists

@jit
def _get_dv_mol_PBC(positions, mol, l_box):
    n_dim = positions.shape[1]
    n_par = len(mol[1])
    mol_posi = np.empty((n_par, n_dim))
    for i in range(n_par):
        mol_posi[i] = positions[mol[1][i]]
    dvs = mol_posi[:, None, :] - mol_posi[None, :, :]
    np.mod(dvs, l_box, out=dvs)
    mask = dvs > np.divide(l_box, 2.)
    dvs += mask * -l_box
    dists = np.sqrt(np.square(dvs).sum(axis=-1))
    return dvs, dists
