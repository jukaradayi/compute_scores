import string

import pytest
import numpy as np


def create_data():
    '''create the mock data, Phoneme gold is a utf-8 file that contains 4 columns
       separated by spaces as:
  
       file_name start_phon end_phon phon

       where:

            file_name: string
            start_phon, end_phon: are float
            phon: string 
       
    '''
    phones = [x for x in string.ascii_lowercase]
    intervals = [[x, x+1, phones[x]] for x in range(0, len(phones))]
    phons_data = np.array([(f, i[0], i[1], i[2]) 
        for f in ['A', 'B']  for i in intervals],
        dtype=[('file_name', 'S10'), ('start', 'f4'), ('end', 'f4'), ('phon', 'S10')])
    return phons_data
 
@pytest.fixture(scope='session')
def num_rows():
    return len(create_data())

@pytest.fixture(scope='session')
def gold_file(tmpdir_factory):
    '''Create the mock phoneme file'''

    phons_data = create_data() 
    fn = tmpdir_factory.mktemp('data').join('mock.gold')
    np.savetxt(str(fn), phons_data, fmt='%s %.4f %.4f %s')
    return fn

