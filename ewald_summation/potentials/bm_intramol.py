import numpy as np
from intramol_template import _get_dv_mol_nPBC, _get_dv_mol_PBC

positions = np.array([[7, 8], [0, 0], [-1, 1], [1, 0], [3, 3]])
mol = (None, [1, 2, 3])
l_box = np.array([2, 2])

print(_get_dv_mol_PBC(positions, mol, l_box))
