import numpy as np
from .traj import Traj
#import ewald_summation as es

class MD:
    #def __init__(self, physical_world, simu_config, particle_initializer, step_runner):
    def __init__(self, simu_config, particle_initializer, step_runner):
        self.config = simu_config
        self.phy_world = simu_config.phys_world
        # step runner, e.g. Langevin integrator or MCMC
        self.step_runner = step_runner

        # for easy fetching
        self.l_box = simu_config.l_box
        # move these initializations to initializer
        # self.masses = simu_config.masses
        # self.charges = simu_config.charges
        self.temp = simu_config.temp
        self.timestep = simu_config.timestep
        self.n_steps, self.n_particles, self.n_dim = simu_config.n_steps, simu_config.n_particles, simu_config.n_dim
        # distance_vectors obj
        #self.distance_vectors = es.distances.DistanceVectors(self.config)

        # traj obj
        self.traj = Traj(self.config)
        initial_frame = self.traj.get_current_frame()
        #self.config.masses, self.config.charges, initial_frame.q, initial_frame.p = particle_initializer(self.l_box, self.n_particles)
        initial_frame.q, initial_frame.p = particle_initializer(self.l_box, self.n_particles)
        initial_frame.p *= self.config.masses[:, None]
        #self.masses, self.charges = self.config.masses, self.config.charges
        # start sanity checks
        #check = es.util.SanityChecks(self.config, particle_initializer, self.masses, self.charges)
        #check.sanity_checks()
        # step runner initiation
        self.step_runner.init(self.config)

        # observables now as an empty dict
        self.observables = {}

        # for managing all existing potentials in the system
        # Potential objects should offer method that calc potentials and forces
        #   for each particle, on given *q* and *config*
        #self.global_potentials = []
        #self.pairwise_potentials = []
        #self.coulumb_potentials = []
        #self.lennard_jones_potentials = []
        self.potentials = []
        self.current_step = -1

    #def add_global_potential(self, new_global_potential):
        #self.global_potentials.append(new_global_potential)
    def add_potential(self, new_potentials):
        # to support single and multiple potentials
        if(np.shape(new_potentials) == ()):
            new_potentials = [new_potentials]
        for new_potential in new_potentials:
            self.potentials.append(new_potential)

    def check_recalc(self, q, step):
        if(step != self.current_step or step == None):
            for pot in self.potentials:
                pot.set_positions(q)
            self.current_step = step

    def sum_force(self, q, step):
        #forces = [pot.calc_force(q, self.config) for pot in self.global_potentials]
        #forces.extend([pot.calc_force(self.distance_vectors(q, step))
                                      #for pot in self.lennard_jones_potentials])
        #forces.extend([pot.calc_force(self.distance_vectors(q, step))
                                      #for pot in self.coulumb_potentials])
        self.check_recalc(q, step)
        forces = []
        for pot in self.potentials:
            forces.append(pot.forces)
        return np.sum(forces, axis=0)
    
    def sum_potential(self, q, step=None):
        self.check_recalc(q, step)
        pots = []
        for pot in self.potentials:
            pots.append(pot.pot)
        return np.sum(pots, axis=0)

    def run_step(self, step):
        next_frame = self.step_runner.run(self.sum_force,
                                          self.sum_potential,
                                          self.traj.get_current_frame(),
                                          self.traj.make_new_frame(),
                                          step,
                                          )
        self.traj.set_new_frame(next_frame)

    def run_all(self):
        # calc_potential initiation
        #self.calc_potential = es.potentials.CalcPotential(self.config, self.global_potentials)
        # calc_force initiation
        #self.calc_force = es.potentials.CalcForce(self.config, self.global_potentials)
        # run sim for n_steps
        for step in range(self.traj.current_frame_num, self.n_steps):
            print(step)
            self.run_step(step)
