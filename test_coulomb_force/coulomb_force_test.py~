import numpy as np
from scipy.special import erfc
from itertools import product




n_particles = 2
n_dim = 3
vac_per = 1 / 4 / np.pi
#l_box =  l_box[0] #assume that box is quadradic
#PBC = PBC
charge_a = 1
charge_b = -1 
mole_fraction = 0.5
resolution = 15

charge_vector_a = np.asarray([charge_a] * round(n_particles *mole_fraction))
charge_vector_b = np.asarray([charge_b] * (n_particles- round(n_particles * mole_fraction)))
charge_vector = np.append(charge_vector_a,charge_vector_b)



q=np.random.rand(n_particles,3)+0.5
#q=np.asarray([[0,0,0],[0,0,1]])
q=q/np.linalg.norm(q[0]-q[1])
print(q)
test_size = 100

def distance_vectors_periodic(q,l_box):
    # new implementation
    distance_vectors = q[:, None, :] - q[None, :, :]
    np.mod(distance_vectors, l_box, out=distance_vectors)
    mask = distance_vectors > np.divide(l_box, 2.)
    distance_vectors += mask * - l_box
    return distance_vectors


def test():
    l_box = np.arange(test_size)+2
    forces = np.zeros(test_size*3*n_particles).reshape(test_size,n_particles,3)
    for _ in range(test_size):
        real_cutoff = l_box[_] / 2
        gauss_scaling = 6 / l_box[_]
        forces[_] = real_space_coulomb_force(distance_vectors_periodic(q,np.asarray([l_box[_]]*3)), l_box[_],real_cutoff,gauss_scaling,q) + reciprocal_space_coulomb_force(q,l_box[_],gauss_scaling)
               
    return forces, l_box


def real_space_coulomb_force(distance_vectors,l_box, real_cutoff,gauss_scaling,q):
    prefactor = 2 * gauss_scaling / (np.sqrt(np.pi))
    force = np.zeros([n_particles,n_dim])
    surrounding_of_boxes =  np.sum(charge_vector[:,None] * q, axis=0)  * 4 * np.pi / 3 * float(l_box) ** (-float(n_dim))
    # sourrounding of the sphere of periodical boxes see 
    # Simulation of electrostatic systems in periodic boundary conditions 
    # I. Lattice sums and dielectric constants, de Leeuw et al., 1980.
    # print(distance_vectors)
    distances = np.linalg.norm(distance_vectors, axis=2)
    for i in range(n_particles):
        indices_to_delete = np.append(np.argwhere(distances[i]>real_cutoff),i)
        r_ij = np.delete(distance_vectors[i], indices_to_delete, axis=0)
        dist = np.delete(distances[i], indices_to_delete)
        #print(dist)
        charge = charge_vector[i]
        charges = np.delete(charge_vector, indices_to_delete)
        force_ = dist**(-3.) * charges * ( prefactor * dist * np.exp(-gauss_scaling*dist**2) + erfc(gauss_scaling * dist))
        force_ = r_ij * force_[:, np.newaxis]
        force[i] = charge  * np.sum(force_, axis=0) - charge * surrounding_of_boxes
    return vac_per*force

def reciprocal_space_coulomb_force(q,l_box,gauss_scaling):
    # input are the particle positions np array with shape n_particles, n_dim
    k = 2 * np.pi / l_box * np.array( list (product(range(-resolution,resolution+1), repeat=n_dim)))
    k = np.delete(k, k.shape[0] // 2,0)
    # removes the zerovector from k
    inv_V = float(l_box)**(-float(n_dim))
    k_dot_x = np.matmul(q,np.transpose(k))
    reciprocal_density_distribution = np.sum(charge_vector[:, np.newaxis] * np.exp(1j*k_dot_x), axis=0)
    k_sq = np.linalg.norm(k, axis = 1) ** 2
    reciprokal_gaus = 4 * np.pi * np.exp(-k_sq / (4 * gauss_scaling**(2)))/ k_sq
    reciprocal_force = np.zeros([n_particles,n_dim])
    for i in range(n_particles):
        reciprocal_force[i] = - inv_V * charge_vector[i] * np.sum(k * reciprokal_gaus[:, np.newaxis] * np.imag(np.exp(1j * q[i] * k) * reciprocal_density_distribution[:, np.newaxis]),axis=0)
    return vac_per * reciprocal_force


a,b=test()
#print(a)
distvec =  q[1]-q[0]
analytic_coulomb = distvec * 1 / 4 / np.pi  * np.linalg.norm(distvec)**(-3) * np.linalg.norm(distvec)
#print(analytic_coulomb)
import matplotlib.pyplot as plt
from matplotlib import rc






plt.subplot(3, 1, 1)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

plt.plot(b,a[:,0,0])
plt.plot(np.arange(test_size),[analytic_coulomb[0]] * test_size)

plt.title('components of Coulomb force')
plt.ylabel('x')


plt.subplot(3, 1, 2)
plt.plot(b,a[:,0,1])
plt.plot(np.arange(test_size),[analytic_coulomb[1]] * test_size)

plt.ylabel('y')



plt.subplot(3, 1, 3)
plt.plot(b,a[:,0,2])
plt.plot(np.arange(test_size),[analytic_coulomb[2]] * test_size)

plt.ylabel('z')
plt.xlabel(r'$l_{box} \sigma $')

plt.show()







