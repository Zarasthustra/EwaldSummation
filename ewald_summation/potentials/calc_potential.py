# import numpy as np
# from numba import jit, njit, cuda, prange
# import math
# from .lj_potential import calc_potential_lj, calc_potential_lj_parallel
# from .coulomb_potential import calc_potential_coulomb_real, calc_potential_coulomb_rec
# from .coulomb_new import coulomb_potential_pairwise
#
#
# class CalcPotential:
#     def __init__(self, config, global_potentials):
#         self.n_dim = config.n_dim
#         self.n_particles = config.n_particles
#         self.l_box = np.array(config.l_box)
#         self.l_box_half = np.array(self.l_box) / 2
#         self.PBC = config.PBC
#         self.lj_flag = config.lj_flag
#         self.coulomb_flag = config.coulomb_flag
#         self.sigma_lj = config.sigma_lj
#         self.epsilon_lj = config.epsilon_lj
#         self.neighbour = config.neighbour
#         self.cutoff = config.cutoff
#         self.switch_start = config.switch_start
#         self.switch_width = self.cutoff - self.switch_start
#         self.parallel_flag = config.parallel_flag
#         self.global_potentials = global_potentials
#         if self.coulomb_flag:
#             self.charges = np.array(config.charges)
#             self.alpha = config.alpha
#             self.rec_reso = config.rec_reso
#             self.epsilon = 1. / (4. * np.pi)
#             self.prefactor_coulomb = 1. / (4. * np.pi * self.epsilon)
#             self.precalc = self.Coulomb_PreCalc(self.l_box, self.charges, self.rec_reso, self.alpha)
#
#     class Coulomb_PreCalc:
#         def __init__(self, l_box, charges, rec_resolution, alpha):
#             self.m = (1/l_box) * _grid_points_without_center(rec_resolution, rec_resolution, rec_resolution)
#             m_modul_sq = np.linalg.norm(self.m, axis = 1) ** 2
#             self.coeff_S = np.exp(-(np.pi / alpha) ** 2 * m_modul_sq) / m_modul_sq
#             self.v_rec_prefactor = 0.5 / np.pi / (l_box[0] * l_box[1] * l_box[2])
#             self.f_rec_prefactor = -charges / (l_box[0] * l_box[1] * l_box[2]) # j and 2pi parts come here
#             self.v_self = -alpha / np.sqrt(np.pi) * np.sum(charges**2)
#
#     def __call__(self, x):
#         potential = np.sum([pot.calc_potential(x) for pot in self.global_potentials])
#         if self.lj_flag and not self.coulomb_flag:
#             if not self.parallel_flag:
#                 return potential + calc_potential_lj(x, self.n_dim, self.n_particles, self.PBC, self.l_box, self.l_box_half,
#                                    self.switch_start, self.cutoff, self.switch_width,
#                                    self.sigma_lj, self.epsilon_lj)
#             else:
#                 return potential + calc_potential_lj_parallel(x, self.n_dim, self.n_particles, self.PBC, self.l_box, self.l_box_half,
#                                    self.switch_start, self.cutoff, self.switch_width,
#                                    self.sigma_lj, self.epsilon_lj)
#         if self.coulomb_flag and not self.lj_flag:
#             return potential + (calc_potential_coulomb_real(x, self.n_dim, self.n_particles, self.charges, self.alpha,
#                                                        self.l_box, self.l_box_half, self.cutoff)
#                                 + calc_potential_coulomb_rec(x, self.precalc.m, self.charges, self.precalc.v_rec_prefactor, self.precalc.coeff_S)
#                                 + self.precalc.v_self) * self.prefactor_coulomb
#
#
# def _grid_points_without_center(nx, ny, nz):
#     a, b, c = np.arange(-nx, nx+1), np.arange(-ny, ny+1), np.arange(-nz, nz+1)
#     xx, yy, zz = np.meshgrid(a, b, c)
#     X = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T
#     return np.delete(X, X.shape[0] // 2, axis=0)
