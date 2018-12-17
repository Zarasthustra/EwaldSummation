def get_charges(particle_kind_vector, charge_a, charge_b):
    """ gives back a vector of the particle charges
    """
    particle_kind_vector = particle_kind()
    charge_vector = [charge_a  if i==0 else charge_b for i in particle_kind_vector]
    return charge_vector
    