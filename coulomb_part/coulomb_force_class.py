class CoulombForce:
    def __init__(self, config):
        self.n_dim = config.n_dim
        self.epsilon_zero = np.sqrt(np.array(config.epsilon_lj)[:, None] * np.array(config.epsilon_lj))
        self.sigma_arr = (0.5 * (np.array(config.sigma_lj)[:, None] + np.array(config.sigma_lj)))
        self.cutoff = config.cutoff_lj
self.switch_start = config.switch_start_lj        
        
        
    