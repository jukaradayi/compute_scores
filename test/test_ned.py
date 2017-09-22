'''Testing the functions that compute the NED'''

import random
import sys
sys.path.insert(0, '../') 

import numpy as np

from compute_ned import func_ned

def test_func_ned():
    '''check the NED function returns valid values, 
       NED is computed from the Levenshtein distance, and it 
       is 0 when two words are the and 1 when all their characters
       are different
    '''

    # strings ...
    assert np.allclose(func_ned("cat", "cat"), 0.0, rtol=1e-02) 
    assert np.allclose(func_ned("rat", "cat"), 1.0/3.0, rtol=1e-02)
    assert np.allclose(func_ned("tac", "cat"), 2.0/3.0, rtol=1e-02)
    assert np.allclose(func_ned("toc", "cat"), 3.0/3.0, rtol=1e-02)

    # arrays with integers
    n, n_ = 100, 0     
    initial_array = range(n)
    modif_array = initial_array[:]
    assert np.allclose(func_ned(initial_array, modif_array), 0.0, rtol=1e-02)
    for idx in initial_array: # incrementaly change the arrays to get neds from 0 to 1
        n_+=1.0
        modif_array[idx] = random.randint(10000, 100000)
        assert np.allclose(func_ned(initial_array, modif_array),
                n_/n, rtol=1e-02)

