'''Testing the function that read the gold phoneme transcription 
NED/coverage and rolling statistics'''

import sys
sys.path.insert(0, '../')
import numpy as np
from compute_ned import read_gold_phn

def test_verify_read_gold(gold_file):
    '''Check if it decode the file and returns the dictionary with "start", "end" and "phon"'''
    phons = read_gold_phn(gold_file)

    # check if the returning values are well labeled 
    assert len(phons.keys()) > 0
    assert "start" in phons.values()[0]
    assert "end" in phons.values()[0]
    assert "phon" in phons.values()[0]

 
def test_verify_num_rows(gold_file, num_rows):
    '''Check if the all rows in the mock file are read'''
    phons = read_gold_phn(gold_file)
    size_starts = []
    size_ends = []
    for f in phons.keys():
        size_starts += phons[f]['start'] 
        size_ends += phons[f]['end']
    
    # it will check if all data was read  
    assert len(size_starts) == num_rows
    assert len(size_ends) == num_rows 
