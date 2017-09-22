'''Testing the function that compute the on-line statistics gives valid results'''

import sys
sys.path.insert(0, '../') 
import numpy as np
from compute_ned import Stream_stats #stream_stats

def test_stream_stats():
    '''Check if the streaming statistics gives the right answer'''

    # Sampling a gaussian distribution
    mu, sigma = 5.0, 0.1 # mean and standard deviation
    gaussian_samples = np.random.normal(mu, sigma, 1000) 

    ## initializing vars for the streaming stats
    online_stats = Stream_stats()
    for sample in gaussian_samples:
        online_stats.add(sample)

    computed_mean = gaussian_samples.mean() 
    assert np.allclose(computed_mean, online_stats.mean())

    computed_variance = gaussian_samples.var()
    assert np.allclose(computed_variance, online_stats.var(), rtol=1e-02)

    computed_std = gaussian_samples.std()
    assert np.allclose(computed_std, online_stats.std(), rtol=1e-02) 
