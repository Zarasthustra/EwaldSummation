import numpy as np
import Traj

class MD:
    def __init__(self, physical_world, simu_config, init_values):
        # TODO: sanity checks
        self.phy_world = physical_world
        self.config = simu_config
        
        # for easy fetching
        self.box_size = simu_config.box_size
        self.masses = simu_config.masses
        self.charges = simu_config.charges
        self.temp = simu_config.temp
        self.timestep = simu_config.timestep
        self.n_steps, self.n_particles, self.n_dim = simu_config.n_steps, simu_config.n_particles, simu_config.n_dim
        
        # traj obj
        self.traj = Traj(self.box_size, self.n_particles, self.n_steps, self.timestep)
        self.traj.qs[0] = init_values.q0
        self.traj.ps[0] = init_values.p0

        # observables now as an empty dict
        self.observables = {}

        # for managing all existing potentials in the system
        # Potential objects should offer method that calc potentials and forces
        #   for each particle, on given *q* and *config*
        self.global_potentials = []
        self.pairwise_potentials = []
        self.coulumb_potentials = []

    def add_global_potential(new_global_potential):
        # check
        self.global_potentials.append(new_global_potential)

    def add_pairwise_potential(new_pairwise_potential):
        # check
        self.pairwise_potentials.append(new_pairwise_potential)

    def add_coulumb_potential(new_coulumb_potential):
        # check
        self.coulumb_potentials.append(new_coulumb_potential)

    def run_step(self):
        # TODO: pseudo-code
        # q_current = self.traj.fetch_q(current_step)
        # forces = [pot.calc_force(q_current, self.config) for pot in potentials]
        # sum_force = np.sum(forces, axis = 0)
        # self.traj[current_step + 1] = integrator(sum_force, self.traj[current_step])


