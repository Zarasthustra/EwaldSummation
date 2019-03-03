from pathlib import Path
import numpy as np

class PdbWriter:
    ATOM = 'ATOM  {:5d} {:>4s} {:>3s}  {:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n'
    def __init__(self, config, filepath='output.pdb'):
        self.l_box = config.l_box
        self.PBC = config.PBC
        self.n_particles = config.n_particles
        self.particle_info = config.particle_info
        self.particle_types = config.particle_types
        self.molecule_types = config.molecule_types
        self.mol_indices = -np.ones((self.n_particles, 2), dtype=np.int)
        for mol in config.mol_list:
            mol_index = mol[1][0] + 1
            mol_type = mol[0]
            for i in mol[1]:
                self.mol_indices[i, 0] = mol_index
                self.mol_indices[i, 1] = mol_type
        #print(self.mol_indices)
        if(config.n_dim != 3):
            raise NotImplementedError
        my_file = Path(filepath)
        if my_file.is_file():
            print('File %s existing, overwriting.' % filepath)
        self.file = open(filepath, 'w')
        self.file.write('CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f}\n'.format(self.l_box[0], self.l_box[1], self.l_box[2], 90., 90., 90.))
        self.is_finished = False
        self.current_model = 1

    def write_frame(self, positions):
        if(not self.is_finished):
            self.file.write('MODEL     {:4d}\n'.format(self.current_model))
            self.positions = positions.copy()
            #if(self.PBC):
                #self.positions %= self.l_box
            for i in range(self.n_particles):
                atom_name = self.particle_types[self.particle_info[i]][0].upper()
                mol_index = self.mol_indices[i, 0]
                if(mol_index == -1):
                    mol_index = i + 1
                    mol_name = 'UNK'
                else:
                    mol_name = self.molecule_types[self.mol_indices[i, 1]][0].upper()
                self.file.write(self.ATOM.format(i + 1, atom_name, mol_name, mol_index, self.positions[i, 0], self.positions[i, 1], self.positions[i, 2], 1., 0.))
            self.file.write('ENDMDL\n')
            self.current_model += 1

    def finalize(self):
        if(not self.is_finished):
            self.file.write('END\n')
            self.file.close()
            self.is_finished = True

    def __del__(self):
        self.finalize()
