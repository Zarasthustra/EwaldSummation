import numpy as np
from numba import cuda, njit, float64
import math


def lj_ref(qij):
    """qij = qi - qj, vector
    """
    distance = np.linalg.norm(qij)
    if(distance <= 0 or distance > cutoff):
        return np.zeros(n_dim)
    else:
        inv_dist_pure = 1 / distance
        inv_dist = sigma / distance
        inv_dist2 = inv_dist * inv_dist
        inv_dist4 = inv_dist2 * inv_dist2
        inv_dist6 = inv_dist2 * inv_dist4
        inv_dist8 = inv_dist4 * inv_dist4
        if(distance <= cutoff - switch_width):
            return 24. * epsilon * inv_dist_pure * inv_dist_pure * inv_dist6 * (2 * inv_dist6 - 1.) * qij
        else:
            t = (distance - cutoff) / switch_width
            # -d(SV) = -dS.V + S.(-dV)
            dsv = -24. * t * (1. + t) / switch_width * inv_dist_pure * epsilon * inv_dist6 * (inv_dist6 - 1.) * qij
            sdv = (t * t * (3. + 2. * t)) * 24. * epsilon * inv_dist_pure * inv_dist_pure * inv_dist6 * (2 * inv_dist6 - 1.) * qij
            return dsv + sdv

# # # CUDA kernel
# @cuda.jit
# def kernel_force_pairwise(q_device, force_device, l_box_device):
#     i = cuda.grid(1)
#     if i < n_particles:
#         force_temp = cuda.local.array(n_dim, dtype=float64)
#         for j in range(n_particles):
#             if i != j:
#                 distance_squared = 0.
#                 dv = cuda.local.array(n_dim, dtype=float64)
#                 for k in range(n_dim):
#                     dv[k] = (q_device[i, k] - q_device[j, k]) % l_box_device[k]
#                     if dv[k] > l_box_device[k] / 2:
#                         dv[k] -= l_box_device[k]
#                     distance_squared += dv[k]**2
#                 distance = math.sqrt(distance_squared)
#                 # if distance < cutoff and distance > 0:
#                 type_i = particle_types[i]
#                 type_j = particle_types[j]
#                 # force_device[i, 0] += dv[k] * lj_force_pairwise(distance, distance_squared, type_i, type_j)
#                 for m in range(3):
#                     force_temp[m] += dv[m] * lj_force_pairwise(distance, distance_squared, type_i, type_j)
#         for l in range(3):
#             force_device[i, l] = force_temp[l]


# # CUDA kernel
@cuda.jit
def kernel_force_pairwise(q_device, force_device, l_box_device):
    i = cuda.grid(1)
    # pos_i = cuda.local.array(n_dim, dtype=float64)
    # for k in range(n_dim):
    #     pos_i[k] = q_device[i, k]
    dv = cuda.local.array(n_dim, dtype=float64)
    for j in range(n_particles):
        for l in range(n_dim):
            force_device[i, l] = (q_device[i, l] - q_device[j, l]) #% l_box_device[l]
            #     if dv[k] > l_box_device[k] / 2:
            #         dv[k] -= l_box_device[k]
            #     distance_squared += dv[k]**2
            # distance = math.sqrt(distance_squared)
            # if distance < cutoff and distance > 0:
            # type_i = particle_types[i]
            # type_j = particle_types[j]
            # for m in range(n_dim):
            #     force_device[i, m] += dv[m] #* lj_force_pairwise(distance, distance_squared, type_i, type_j)
        # for m in range(3):
        #     force_device[i, m] += dv[m] * lj_force_pairwise(distance, distance_squared, type_i, type_j)
# for l in range(3):
#     force_device[i, l] = force_temp[l]

@cuda.jit(device=True)
def lj_force_pairwise(distance, distance_squared, type_i, type_j):
    # calculate force below switch region, assumes all distances passed > 0
    sigma = lj_mixing_conditions[type_i * n_particles + type_j][0]
    sigma6 = lj_mixing_conditions[type_i * n_particles + type_j][1]
    epsilon = lj_mixing_conditions[type_i * n_particles + type_j][2]
    if(distance <= switch_start) and (distance > 0):
        output = (24 * epsilon * sigma6 / distance_squared**4
                  * (2 * sigma6 / distance_squared**3 - 1))
        return output

    # calculate potential in switch region, (gradient * switch -potential * dswitch)
    elif (distance > switch_start) and (distance <= cutoff):
        t = (distance - cutoff) / switch_width
        gradient = 24 * epsilon * sigma6 / distance**8 * (2 * sigma6 / distance**6 - 1)
        potential = 4. * epsilon * sigma6 / distance_squared**3 * (sigma6 / distance_squared**3 - 1)
        switch = 2 * t**3 + 3 * t**2
        dswitch = 6 / (cutoff - switch_start) / distance * (t**2 + t)
        output = gradient * switch - potential * dswitch
        return output

    # set rest to 0
    else:
        return 0.


# Host code

# Initialize the data arrays

n = 3
n_particles = n
n_dim = 3
switch_start = 2.5
switch_width = 1
cutoff = 3.5
l_box = np.array([11, 11, 11])
switch_start = 2.5
cutoff = 3.5
sigma = 1
epsilon = 1
sigma_lj = tuple([1.]*n)
epsilon_lj = tuple([1.]*n)
particle_types = (0, 1)
lj_mixing_conditions = tuple([(1, 1, 1)] * n**2)
# q = np.random.random((3, 3))
q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
PBC = True
force = np.zeros((n_particles, n_dim))

# Copy the arrays to the device
q_device = cuda.to_device(q)
l_box_device = cuda.to_device(l_box)
sigma_lj_device = cuda.to_device(sigma_lj)
epsilon_lj_device = cuda.to_device(epsilon_lj)
force_device = cuda.to_device(force)

# Allocate memory on the device for the result

# Configure the blocks
threadsperblock = 3
blockspergrid_x = int(math.ceil(q.shape[0] / threadsperblock))
blockspergrid = 1

# Start the kernel
kernel_force_pairwise[blockspergrid, threadsperblock](q_device, force_device, l_box_device)

# Copy the result back to the host
# print(x)
C = force_device.copy_to_host()
print(C)
# ref = np.sum(distance_vectors_periodic(x, l_box).sum(axis=-1), axis=-1)
# np.testing.assert_almost_equal(C.sum() / 2, ref)

ref = np.zeros((n, 3))
for i in range(n):
    for j in range(n):
        ref[i, :] += q[i, :] - q[j, :]
print('ref')
print(ref)
