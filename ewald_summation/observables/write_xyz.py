class WriteXyz:
    def __init__(self, config, qs, component_a, component_b):
        self.component_a = component_a
        self.component_b = component_b
        self.qs = qs
        self.n_dim = config.n_dim
        self.l_box = config.l_box
        self.n_particles = config.n_particles
        self.mole_fraction = config.mole_fraction
    def write_xyz(self):
        """
        writes a .xyz format trajectory
        qs = coordinates 
        component_a = string for example "Na"
        component_a = string for example "Cl"
        """
        file = open('vmd_file.xyz', 'w')
        for frame in range(self.qs.shape[0]):
            file.write(str(self.n_particles))
            file.write('\n')
            file.write(' ')
            file.write('\n')
            q = self.qs[frame] % self.l_box
            for i in range(self.n_particles):
                x = q[i, 0]
                if self.n_dim >= 2:
                    y = q[i, 1]
                else:
                    y,z = 0, 0
                if self.n_dim >= 3:
                    z = q[i,2]
                else:
                    z = 0
                if i <= int(self.n_particles * self.mole_fraction) - 1:
                    file.write(self.component_a)
                    file.write(' ')
                else:
                    file.write(self.component_b)
                    file.write(' ')
                file.write(str(round(x,3)))
                file.write(' ')
                file.write(str((round(y,3))))
                file.write(' ')
                file.write(str((round(z,3))))
                file.write('\n')
        
        