import numpy as np

def globalx(global_pot):
    """Decorator to wrap a global potential/force function to a 
    well defined class.
    
    Usage:
    @global
    def Blahblah(config, <parameters>...):
        # define global potential function and global 
        # force function under config and parameters.
        # Both functions should accept all particle
        # positions q as input.
        # q: position vector [q_i]
        # q_i will be provided as the image within the box.
        return pairwise_pot, pairwise_force
    
    The resulting class will have:
    1. set_positions(q): method
    2. pots, pot: property
    3. forces: property
    """
    def mk_cls(config, *args, **kwargs):
        return GlobalTemplate(config, global_pot(config, *args, **kwargs))
    return mk_cls 

class GlobalTemplate:
    def __init__(self, config, func):
        self.PBC = config.PBC
        self.l_box = config.l_box
        self.n_particles = config.n_particles
        self.n_dim = config.n_dim
        self.pot_func = func[0]
        self.force_func = func[1]
        self.positions = np.zeros((self.n_particles, self.n_dim))
        self._pot = None
        self._forces = None
        
    def set_positions(self, new_positions):
        if(self.PBC):
            self.positions[:, :] = new_positions % self.l_box
        else:
            self.positions[:, :] = new_positions
        self._pot_recalc = True
        self._force_recalc = True

    @property
    def pot(self):
        if(self._pot_recalc):
            self._pot = self.pot_func(self.positions)
            self._pot_recalc = False
        return self._pot

    @property
    def forces(self):
        if(self._force_recalc):
            self._forces = self.force_func(self.positions)
            self._force_recalc = False
        return self._forces
