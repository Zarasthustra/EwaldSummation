import numpy as np

def get_charges(particle_kind_vector, charge_a, charge_b):
    """ gives back a vector of the particle charges
    """
    charge_vector_ = [charge_a  if i==0 else charge_b for i in particle_kind_vector]
    charge_vector = np.asarray(charge_vector_)
    return charge_vector
    