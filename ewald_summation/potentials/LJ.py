# implementation of a Lennard-Jones potential & gradient calculation with cutoff and switch function
# switch function: smoothstep S_1
import numpy as np

# constants
epsilon_lj = 1.
sigma_lj = 2
cutoff_lj = 3.5 * sigma_lj
switch_width_lj = 2.
ndim = 2

def LJ_potential_pairwise(distance):
    if(distance <= 0 or distance > cutoff_lj):
        return 0.
    else:
        inv_dist = sigma_lj / distance
        inv_dist2 = inv_dist * inv_dist
        inv_dist4 = inv_dist2 * inv_dist2
        inv_dist6 = inv_dist2 * inv_dist4
        phi_LJ = 4. * epsilon_lj * inv_dist6 * (inv_dist6 - 1.)
        if(distance <= cutoff_lj - switch_width_lj):
            return phi_LJ
        else:
            t = (distance - cutoff_lj) / switch_width_lj
            switch = t * t * (3. + 2. * t)
            return phi_LJ * switch

def LJ_force_pairwise(qij):
    """qij = qi - qj, vector
    """
    distance = np.linalg.norm(qij)
    if(distance <= 0 or distance > cutoff_lj):
        return np.zeros(ndim)
    else:
        inv_dist_pure = 1 / distance
        inv_dist = sigma_lj / distance
        inv_dist2 = inv_dist * inv_dist
        inv_dist4 = inv_dist2 * inv_dist2
        inv_dist6 = inv_dist2 * inv_dist4
        inv_dist8 = inv_dist4 * inv_dist4
        if(distance <= cutoff_lj - switch_width_lj):
            return 24. * epsilon_lj * inv_dist_pure * inv_dist_pure * inv_dist6 * (2 * inv_dist6 - 1.) * qij
        else:
            t = (distance - cutoff_lj) / switch_width_lj
            # d(SV) = dS.V + S.dV
            dsv = -24. * t * (1. + t) * inv_dist * epsilon_lj * inv_dist6 * (inv_dist6 - 1.)
            sdv = (t * t * (3. + 2. * t)) * 24. * epsilon_lj * inv_dist_pure * inv_dist_pure * inv_dist6 * (2 * inv_dist6 - 1.) * qij
            return dsv + sdv

