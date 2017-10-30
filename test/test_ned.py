'''Testing the functions that compute the NED'''

import random
import sys
import os
sys.path.insert(0, '../') 

import numpy as np

from compute_ned import func_ned
from compute_ned import ned_from_class

def test_func_ned():
    '''check the NED function returns valid values, 
       NED is computed from the Levenshtein distance, and it 
       is 0 when two words are the and 1 when all their characters
       are different
    '''
    print('test the levensthein distance')
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

def test_complete_ned():
    '''check the NED function returns correct NED for a 
       dummy alignment and a small number of classes.
       These classes are made to test several cases:
       class 0 : within, left edge contains less than 0.03 ms of phoneme so don't 
                take it into account
       class 1 : within, left edge contains more than 0.03 ms of phoneme so take
                it into account
       class 2 : within, left edge contains less than 50% of phone, don't take
       class 3 : within, left edge contains more than 50% of phone, take it
       class 4 : within, right and left edge are exactle on phone boundaries 
       class 5 : across: same
       class 6 : across
       class 7 : across, left edge is before every phone
       class 8 : across, right edge is after every phone
    '''
    print('testing complete ned')
    disc_class =  os.path.join(os.path.realpath(__file__), 'class_mock_test')
    overall, across, within = ned_from_class(disc_class)
    
    assert across == 0.25
    assert within == 0.5
    assert round(overall,2) == 0.39

if __name__=="__main__":
    test_func_ned()
    PHON_GOLD= os.path.join(os.path.realpath(__file__), 'mock.phn')
    test_complete_ned()
