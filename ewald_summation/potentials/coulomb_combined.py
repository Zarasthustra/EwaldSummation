import numpy as np
from .coulomb_real import CoulombReal
from .coulomb_reciprocal import CoulombReciprocal
# to correct intramolecular calculation
from .coulomb_correction import CoulombCorrection

def Coulomb(config, accuracy=1e-8):
    """Class factory to generate coulomb potential/force for the MD system, will automa-
    tically select parameters (alpha, real and reciprocal cutoff) for ewald summation.
    
    Inputs:
    config: MD system configuration object.
    accuracy(float): the accuracy threshold for ewald summation.
    """
    
    # optimized for n^3 NaCl grid, 8<n<20
    ratio_real_rec = 5.5 # super parameter for balancing the calculation time of real
    # and reciprocal part. Ideally make them the same and total time achieving O(n^1.5)
    # In reality the ratio between real and reciprocal times may vary, but should be
    # within the 0.1 to 10.
    
    # optimal alpha and cutoff selections
    # ref: http://protomol.sourceforge.net/ewald.pdf
    V = config.l_box[0] * config.l_box[1] * config.l_box[2]
    alpha = ratio_real_rec * np.sqrt(np.pi) * (config.n_particles / V / V) ** (1/6)
    REAL_CUTOFF = np.sqrt(-np.log(accuracy)) / alpha
    REC_RESO = int(np.ceil(np.sqrt(-np.log(accuracy)) * 2 * alpha))
    
    return CoulombReal(config, alpha, REAL_CUTOFF), CoulombReciprocal(config, alpha, REC_RESO), CoulombCorrection(config, alpha)
