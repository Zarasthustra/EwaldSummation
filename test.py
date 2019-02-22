import numpy as np
from numba import njit

@njit
def test_func():
    a=np.sort(np.random.rand(5))
    return a
test_func()
