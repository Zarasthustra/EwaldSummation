import numpy as np
from numba import njit

def deco(pairwise_pot):
    def mk_cls(config, *args, **kwargs):
        # pw = PWNumba(config)
        # pw.pot_func, pw.force_func = pairwise_pot(config, *args, **kwargs)
        return PWNumba(config, pairwise_pot(config, *args, **kwargs))
    return mk_cls

_HEAD = -1
_None = -1

class PWNumba:
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

    def __init__(self, config, func, cutoff=None):
        assert config.PBC, 'Only implemented for periodic case.'
        self.l_box = config.l_box
        self.n_particles = config.n_particles
        self.n_dim = config.n_dim
        self.particle_info = config.particle_info
        self.positions = np.zeros((self.n_particles, self.n_dim))
        if(len(func) == 3):
            self.CUTOFF = func[2]
        else:
            self.CUTOFF = cutoff
        assert self.CUTOFF is not None, 'invalid cutoff.'
        self.pot_func = func[0]
        self.force_func = func[1]
        # determine if multiple images are needed (cutoff ?> l_box / 2)
        # otherwise nearest image convention is used
        self.image_radii = (np.ceil(2 * self.CUTOFF / self.l_box) - 1).astype(int)
        if(np.all(self.image_radii == 0)):
            # nearest image convention + neighbor list
            self.MULTI = False
            # gridding the box
            self.n_cell = self.l_box // self.CUTOFF
            self.l_cell = self.l_box / self.n_cell
            self.n_cells = int(np.cumprod(self.n_cell)[-1])
            self.convert = np.ones(self.n_dim, np.int32)
            for i in range(self.n_dim - 1, 0, -1):
                self.convert[i - 1] = self.convert[i] * self.n_cell[i]
            self.new_raw_index = -np.ones((self.n_particles, self.n_dim), np.int32)
            # in_which_cell, every line (curr_box_index)
            self.in_which_cell = -np.ones(self.n_particles, np.int32)
            # neighbor list, a double linked list, every line (pred_, succ)
            self.neighbor_list = -np.ones((self.n_particles, 2), np.int32)
            self.neighbor_list_head = -np.ones(self.n_cells, np.int32)
            # neighboring_cells, every line (neighboring cells)
            self.neighboring_cells = np.empty((self.n_cells, 3 ** self.n_dim), np.int32)
            _make_neighboring_cells(self.neighboring_cells, self.n_cell, self.convert)
        else:
            # multiple images within cutoff
            self.MULTI = True
            self.image_grid = self.l_box * _grid(self.image_radii)
            self.n_image = len(self.image_grid)
            # raise NotImplementedError
        # core generator for functionality
        self._pots = None
        self._forces = None

    def set_positions(self, new_positions):
        self.positions[:, :] = new_positions % self.l_box
        if not self.MULTI:
            # update neighbor list
            self.new_raw_index[:, :] = self.positions // self.l_cell
            _update_neighbor_list(self.new_raw_index, self.convert, self.neighbor_list,
                                self.neighbor_list_head, self.in_which_cell)
        self._pot_recalc = True
        self._force_recalc = True
    
    @property
    def pots(self):
        if(self._pot_recalc):
            if(self.MULTI):
                self._pots = _pots_multi(self.n_particles, self.positions, self.l_box,
                                        self.n_image, self.image_grid, self.pot_func, self.particle_info)
            else:
                self._pots = _pots_neighbor(self.n_cells, self.n_particles, self.neighbor_list_head,
                     self.neighboring_cells, self.neighbor_list, self.positions, self.l_box, self.pot_func, self.particle_info)
            self._pot_recalc = False
        return self._pots
    
    @property
    def pot(self):
        return np.sum(self.pots)
    
    @property
    def forces(self):
        if(self._force_recalc):
            if(self.MULTI):
                self._forces = _forces_multi(self.n_particles, self.positions, self.l_box,
                                        self.n_image, self.image_grid, self.force_func, self.particle_info)
            else:
                self._forces = _forces_neighbor(self.n_cells, self.n_particles, self.neighbor_list_head,
                     self.neighboring_cells, self.neighbor_list, self.positions, self.l_box, self.force_func, self.particle_info)
            self._force_recalc = False
        return self._forces
'''
    def recalc(self):
        if(self.MULTI):
            self.pairs = _pairs_multi(self.n_particles, self.positions, self.l_box,
                                      self.n_image, self.image_grid)
        else:
            self.pairs = _pairs_neighbor(self.n_cells, self.n_particles, self.neighbor_list_head,
                     self.neighboring_cells, self.neighbor_list, self.positions, self.l_box, self.pot_func)

    def get_particles_in_cutoff(self, particle_index):
        """Currently only for test use.
        """
        my_cell_index = self.in_which_cell[particle_index]
        temp = np.empty(self.n_particles, np.int32)
        count =  _get_particles_in_neighboring_cells(temp, self.neighbor_list,
                     self.neighbor_list_head, self.neighboring_cells[my_cell_index])
        return temp[:count]
'''

def _grid(n_cell):
    xx = np.meshgrid(*[np.arange(-n, n+1) for n in n_cell])
    X = np.vstack([v.reshape(-1) for v in xx]).T
    return X

# def _get_cell_index():

def _make_neighboring_cells(neighboring_cells, n_cell, convert):
    n_dim = n_cell.shape[0]
    grids = np.meshgrid(*([[-1, 0, 1]] * n_dim))
    grids = np.vstack([v.reshape(-1) for v in grids]).T
    cells = np.meshgrid(*[np.arange(n) for n in n_cell])
    cells = np.vstack([v.reshape(-1) for v in cells]).T
    for cell in cells:
        index = int((cell * convert).sum())
        i = 0
        for grid in grids:
            neighboring_cells[index, i] = int(((cell + grid) % n_cell * convert).sum())
            i += 1    

@njit
def _update_neighbor_list(new_raw_index, convert, neighbor_list, neighbor_list_head, in_which_cell):
    for i in range(len(new_raw_index)):
        new_index = (new_raw_index[i] * convert).sum()
        old_index = in_which_cell[i]
        if(new_index != old_index):
            in_which_cell[i] = new_index
            # remove particle i from old list
            next_particle = neighbor_list[i, 1]
            if(neighbor_list[i, 0] == _HEAD):
                neighbor_list_head[old_index] = next_particle
            else:
                neighbor_list[neighbor_list[i, 0], 1] = next_particle
            neighbor_list[next_particle, 0] = neighbor_list[i, 0]
            # insert particle i to new list
            neighbor_list[i, 0] = _HEAD
            neighbor_list[neighbor_list_head[new_index], 0] = i
            neighbor_list[i, 1] = neighbor_list_head[new_index]
            neighbor_list_head[new_index] = i

@njit
def _get_particles_in_neighboring_cells(temp, neighbor_list,
                     neighbor_list_head, neighboring_cells):
    i = 0
    for cell in neighboring_cells:
        curr_particle = neighbor_list_head[cell]
        while(curr_particle != _None):
            temp[i] = curr_particle
            i += 1
            curr_particle = neighbor_list[curr_particle, 1]
    return i

# pots and forces need to be defined separatedly, otherwise not supported by
# Numba nonpython jit mode.
@njit
def _pots_neighbor(n_cells, n_particles, neighbor_list_head, neighboring_cells, neighbor_list,
                    positions, l_box, pot_func, particle_info):
    neighboring_particles = np.empty(n_particles, np.int32)
    pots_sum = np.zeros(n_particles)
    half_l_box = np.divide(l_box, 2.)
    for current_cell in range(n_cells):
        particle_i = neighbor_list_head[current_cell]
        if(particle_i != _None):
            count = _get_particles_in_neighboring_cells(neighboring_particles, neighbor_list,
                                                        neighbor_list_head, neighboring_cells[current_cell])
        while(particle_i != _None):
            for particle_j in neighboring_particles[:count]:
                if(particle_j != particle_i):
                    #yield(((particle_i, particle_j), _calc_dist_neighbor(positions,
                    #                     particle_i, particle_j, l_box, half_l_box)))
                    type_i, type_j = particle_info[particle_i], particle_info[particle_j]
                    pots_sum[particle_i] += pot_func((type_i, type_j), _calc_dist_neighbor(positions,
                                         particle_i, particle_j, l_box, half_l_box))
            particle_i = neighbor_list[particle_i, 1]
    return pots_sum

@njit
def _forces_neighbor(n_cells, n_particles, neighbor_list_head, neighboring_cells, neighbor_list,
                    positions, l_box, force_func, particle_info):
    neighboring_particles = np.empty(n_particles, np.int32)
    n_dim = len(l_box)
    forces_sum = np.zeros((n_particles, n_dim))
    half_l_box = np.divide(l_box, 2.)
    for current_cell in range(n_cells):
        particle_i = neighbor_list_head[current_cell]
        if(particle_i != _None):
            count = _get_particles_in_neighboring_cells(neighboring_particles, neighbor_list,
                                                        neighbor_list_head, neighboring_cells[current_cell])
        while(particle_i != _None):
            for particle_j in neighboring_particles[:count]:
                if(particle_j != particle_i):
                    #yield(((particle_i, particle_j), _calc_dist_neighbor(positions,
                    #                     particle_i, particle_j, l_box, half_l_box)))
                    type_i, type_j = particle_info[particle_i], particle_info[particle_j]
                    forces_sum[particle_i] += force_func((type_i, type_j), _calc_dist_neighbor(positions,
                                         particle_i, particle_j, l_box, half_l_box))
            particle_i = neighbor_list[particle_i, 1]
    return forces_sum

@njit
def _calc_dist_neighbor(positions, particle_i, particle_j, l_box, half_l_box):
    dv = positions[particle_i] - positions[particle_j]
    dv += (dv < 0.) * l_box
    dv -= (dv > half_l_box) * l_box
    dist = np.sqrt(np.square(dv).sum())
    return dv, dist

@njit
def _pots_multi(n_particles, positions, l_box, n_image, image_grid, pot_func, particle_info):
    pots_sum = np.zeros(n_particles)
    for image in range(n_image):
        for i in range(n_particles):
            for j in range(n_particles):
                dv = positions[j] - positions[i] + image_grid[image]
                dist = np.sqrt(np.square(dv).sum())
                if(dist != 0.):
                    type_i, type_j = particle_info[i], particle_info[j]
                    pots_sum[i] += pot_func((type_i, type_j), (dv, dist))
    return pots_sum

@njit
def _forces_multi(n_particles, positions, l_box, n_image, image_grid, force_func, particle_info):
    n_dim = len(l_box)
    forces_sum = np.zeros((n_particles, n_dim))
    for image in range(n_image):
        for i in range(n_particles):
            for j in range(n_particles):
                dv = positions[j] - positions[i] + image_grid[image]
                dist = np.sqrt(np.square(dv).sum())
                if(dist != 0.):
                    type_i, type_j = particle_info[i], particle_info[j]
                    forces_sum[i] += force_func((type_i, type_j), (dv, dist))
    return forces_sum
