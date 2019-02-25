import numpy as np
from numba import njit

_HEAD = -1
_None = -1

class DVNumba:
    """Class for a transparent particle pair-list with Numba njit with auto switch
     between multiple images and nearest image convention (neighbor list).

    Usage:
    1. init with configuration and cutoff
    2. every step call .set_positions(q)
    3. loop over the generator .pairs, which will provide entries like:
        ((particle_i, particle_j), (distVec, dist))
        (particle_i/j is particle index for property retrieving)
    4. if you want to loop over particle pairs again, rewind the generator by calling
       .rewind() method
    """

    def __init__(self, config, cutoff):
        assert config.PBC, 'Only implemented for periodic case.'
        self.l_box = config.l_box
        self.n_particles = config.n_particles
        self.n_dim = config.n_dim
        self.positions = np.zeros((self.n_particles, self.n_dim))
        self.CUTOFF = cutoff
        # determine if multiple images are needed (cutoff ?> l_box / 2)
        # otherwise nearest image convention is used
        self.image_radii = (np.ceil(2 * self.CUTOFF / self.l_box) - 1).astype(int)
        if(np.all(self.image_radii == 0)):
            # nearest image convention + neighbor list
            self.MULTI = False
            # gridding the box
            self.n_cell = self.l_box // cutoff
            self.l_cell = self.l_box / self.n_cell
            self.n_cells = int(np.cumprod(self.n_cell)[-1])
            self.convert = np.ones(self.n_dim, np.int32)
            for i in range(self.n_dim - 1, 0, -1):
                self.convert[i - 1] = self.convert[i] * self.n_cell[i]
            self.new_raw_index = -np.ones((self.n_particles, self.n_dim), np.int32)
            # particle_info, every line (curr_box_index)
            self.particle_info = -np.ones(self.n_particles, np.int32)
            # neighbor list, a double linked list, every line (pred_, succ)
            self.neighbor_list = -np.ones((self.n_particles, 2), np.int32)
            self.neighbor_list_head = -np.ones(self.n_cells, np.int32)
            # cell info, every line (neighboring cells)
            self.cell_info = np.empty((self.n_cells, 3 ** self.n_dim), np.int32)
            _make_cell_info(self.cell_info, self.n_cell, self.convert)
        else:
            # multiple images within cutoff
            self.MULTI = True
            self.image_grid = self.l_box * _grid(self.image_radii)
            self.n_image = len(self.image_grid)
            # raise NotImplementedError
        # core generator for functionality
        self.pairs = None

    def set_positions(self, new_positions):
        self.positions[:, :] = new_positions % self.l_box
        if not self.MULTI:
            # update neighbor list
            self.new_raw_index[:, :] = self.positions // self.l_cell
            _update_neighbor_list(self.new_raw_index, self.convert, self.neighbor_list,
                                self.neighbor_list_head, self.particle_info)
        self.rewind()
    
    def rewind(self):
        if(self.MULTI):
            self.pairs = _pairs_multi(self.n_particles, self.positions, self.l_box,
                                      self.n_image, self.image_grid)
        else:
            self.pairs = _pairs_neighbor(self.n_cells, self.n_particles, self.neighbor_list_head,
                     self.cell_info, self.neighbor_list, self.positions, self.l_box)

    def get_particles_in_cutoff(self, particle_index):
        """Currently only for test use.
        """
        my_cell_index = self.particle_info[particle_index]
        temp = np.empty(self.n_particles, np.int32)
        count =  _get_particles_in_neighboring_cells(temp, self.neighbor_list,
                     self.neighbor_list_head, self.cell_info[my_cell_index])
        return temp[:count]

def _grid(n_cell):
    xx = np.meshgrid(*[np.arange(-n, n+1) for n in n_cell])
    X = np.vstack([v.reshape(-1) for v in xx]).T
    return X

# def _get_cell_index():

def _make_cell_info(cell_info, n_cell, convert):
    n_dim = n_cell.shape[0]
    grids = np.meshgrid(*([[-1, 0, 1]] * n_dim))
    grids = np.vstack([v.reshape(-1) for v in grids]).T
    cells = np.meshgrid(*[np.arange(n) for n in n_cell])
    cells = np.vstack([v.reshape(-1) for v in cells]).T
    for cell in cells:
        index = int((cell * convert).sum())
        i = 0
        for grid in grids:
            cell_info[index, i] = int(((cell + grid) % n_cell * convert).sum())
            i += 1    

@njit
def _update_neighbor_list(new_raw_index, convert, neighbor_list, neighbor_list_head, particle_info):
    for i in range(len(new_raw_index)):
        new_index = (new_raw_index[i] * convert).sum()
        old_index = particle_info[i]
        if(new_index != old_index):
            particle_info[i] = new_index
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

@njit
def _pairs_neighbor(n_cells, n_particles, neighbor_list_head, cell_info, neighbor_list,
                    positions, l_box):
    neighboring_particles = np.empty(n_particles, np.int32)
    half_l_box = np.divide(l_box, 2.)
    for current_cell in range(n_cells):
        particle_i = neighbor_list_head[current_cell]
        if(particle_i != _None):
            count = _get_particles_in_neighboring_cells(neighboring_particles, neighbor_list,
                                                        neighbor_list_head, cell_info[current_cell])
        while(particle_i != _None):
            for particle_j in neighboring_particles[:count]:
                if(particle_i != particle_j):
                    yield(((particle_i, particle_j), _calc_dist_neighbor(positions,
                                         particle_i, particle_j, l_box, half_l_box)))
            particle_i = neighbor_list[particle_i, 1]

@njit
def _calc_dist_neighbor(positions, particle_i, particle_j, l_box, half_l_box):
    dv = positions[particle_j] - positions[particle_i]
    dv += (dv < 0.) * l_box
    dv -= (dv > half_l_box) * l_box
    dist = np.sqrt(np.square(dv).sum())
    return dv, dist

@njit
def _pairs_multi(n_particles, positions, l_box, n_image, image_grid):
    for image in range(n_image):
        for i in range(n_particles):
            for j in range(n_particles):
                dv = positions[j] - positions[i] + image_grid[image]
                dist = np.sqrt(np.square(dv).sum())
                if(dist != 0.):
                    yield(((i, j), (dv, dist)))
