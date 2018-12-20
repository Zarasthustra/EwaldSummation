import numpy as np
from .traj import Traj

class MD:
    def __init__(self, physical_world, simu_config, particle_initializer, step_runner):
        # TODO: sanity checks
        self.phy_world = physical_world
        self.config = simu_config
        # step runner, e.g. Langevin integrator or MCMC
        self.step_runner = step_runner
                
        # for easy fetching
        self.box_size = simu_config.box_size
        # move these initializations to initializer
        # self.masses = simu_config.masses
        # self.charges = simu_config.charges
        self.temp = simu_config.temp
        self.timestep = simu_config.timestep
        self.n_steps, self.n_particles, self.n_dim = simu_config.n_steps, simu_config.n_particles, simu_config.n_dim
        
        # traj obj
        self.traj = Traj(self.config)
        initial_frame = self.traj.get_current_frame()
        self.config.masses, self.config.charges, initial_frame.q, initial_frame.p = particle_initializer(self.box_size, self.n_particles)
        self.masses, self.charges = self.config.masses, self.config.charges

        # step runner initiation
        self.step_runner.init(self.phy_world, self.config)

        # observables now as an empty dict
        self.observables = {}

        # for managing all existing potentials in the system
        # Potential objects should offer method that calc potentials and forces
        #   for each particle, on given *q* and *config*
        self.global_potentials = []
        self.pairwise_potentials = []
        self.coulumb_potentials = []

    def add_global_potential(self, new_global_potential):
        # check
        self.global_potentials.append(new_global_potential)

    def add_pairwise_potential(self, new_pairwise_potential):
        # check
        self.pairwise_potentials.append(new_pairwise_potential)

    def add_coulumb_potential(self, new_coulumb_potential):
        # check
        self.coulumb_potentials.append(new_coulumb_potential)

    def sum_force(self, q):
        forces = [pot.calc_force(q, self.config) for pot in self.global_potentials]
        # TODO: include other potentials
        return np.sum(forces, axis = 0)

    def run_step(self):
        next_frame = self.step_runner.run(self.sum_force, self.traj.get_current_frame(), self.traj.make_new_frame())
        self.traj.set_new_frame(next_frame)
    
    def run_all(self):
        for _ in range(self.traj.current_frame_num, self.n_steps):
            self.run_step()


