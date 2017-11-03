import sys
import math
import os
import logging
import codecs
from bisect import bisect_left, bisect_right, bisect
from itertools import combinations, count
import argparse
import pdb

import numpy as np 
import pandas as pd
import editdistance # see (1)

def read_gold_phn(phn_gold):
    ''' read the gold phoneme file with fields : speaker/file start end phon,
    returns a dict with the file/speaker as a key and the following structure
    
    gold['speaker'] = [{'start': list(...)}, {'end': list(...), 'phon': list(...)}]
    '''
    df = pd.read_table(phn_gold, sep='\s+', header=None, encoding='utf8',
            names=['file', 'start', 'end', 'phon'])
    df = df.sort_values(by=['file', 'start']) # sorting the data
    df['start'] = df['start'].round(decimals=4)
    df['end'] = df['end'].round(decimals=4)
    number_read_phons = len(df['phon'])

    # get the lexicon and translate to as integers
    symbols = list(set(df['phon']))
    symbol2ix = {v: k for k, v in enumerate(symbols)}
    ix2symbols = dict((v,k) for k,v in symbol2ix.iteritems())
    df['phon'] = df['phon'].map(symbol2ix)

    # timestamps in gold (start, end) must be in acending order for fast search
    gold = {}
    verification_num_phones = 0
    for k in df['file'].unique():
        start = df[df['file'] == k]['start'].values
        end = df[df['file'] == k]['end'].values
        phon = df[df['file'] == k]['phon'].values
        assert not any(np.greater_equal.outer(start[:-1] - start[1:], 0)), 'start in phon file is not odered!!!'
        assert not any(np.greater_equal.outer(end[:-1] - end[1:], 0)), 'end in phon file is not odered!!!'
        gold[k] = {'start': list(start), 'end': list(end), 'phon': list(phon)} 
        verification_num_phones += len(gold[k]['phon'])

    logging.debug("%d phonemes read from %s (%d returned)", number_read_phons,
            phn_gold, verification_num_phones) 
   
    return gold, ix2symbols

def check_phn_boundaries(gold_bg, gold_ed, gold, classes, elem):
    ''' check boundaries of discovered phone.
        If discovered "word" contains 50% of a phone, or more than
        30ms of a phone, we consider that phone discovered.
    '''
    # get discovered phones timestamps
    spkr, disc_bg, disc_ed = classes[elem]
    # get first phone timestamps
    first_ph_bg = gold[spkr]['start'][max(gold_bg-1,0)] # avoid taking last element if gold_bg = 0
    first_ph_ed = gold[spkr]['end'][max(gold_bg-1,0)] # avoid taking last element if gold_bg = 0
    first_ph_len = first_ph_ed - first_ph_bg
    first_ph_ov = float(first_ph_ed - disc_bg)/first_ph_len

    # get last phone timestamps
    last_ph_bg = gold[spkr]['start'][min(gold_ed,len(gold[spkr]['start'])-1)]
    last_ph_ed = gold[spkr]['end'][min(gold_ed,len(gold[spkr]['start'])-1)]
    last_ph_len = last_ph_ed - last_ph_bg
    last_ph_ov = float(disc_ed - last_ph_bg)/last_ph_len

    #pdb.set_trace()
    # check overlap between first phone in transcription and discovered word
    # Bugfix : when reading alignments, pandas approximates float values
    # and it can lead to problems when th difference between the two compared 
    # values is EXACTLY 0.03, so we have to round the values to 0.0001 precision ! 
    if (round(first_ph_len,4) >= 0.060 and round((first_ph_ed - disc_bg),4) >= 0.030) or \
       (round(first_ph_len,4) < 0.060 and first_ph_ov >= 0.5) and \
       (gold_bg !=0 or disc_bg >first_ph_bg):
        # avoid substracting - 1 when already first phone in Gold
        first_ph_pos = gold_bg - 1 if gold_bg > 0 else 0 
        
    elif (gold_bg == 0 and disc_bg <= round(first_ph_bg,4)):
        first_ph_pos = gold_bg
    else:
        first_ph_pos = gold_bg
    
    # check overlap between last phone in transcription and discovered word
    # Bugfix : when reading alignments, pandas approximates float values
    # and it can lead to problems when th difference between the two compared 
    # values is EXACTLY 0.03, so we have to round the values to 0.0001 precision ! 
    if (round(last_ph_len,4) >= 0.060 and round((disc_ed - last_ph_bg),4) >= 0.030) or \
       (round(last_ph_len,4) < 0.060 and last_ph_ov >= 0.5):
        # avoid adding + 1 if already last phone in Gold
        last_ph_pos = gold_ed + 1 if gold_ed < len(gold[spkr]['end']) - 1  else gold_ed
    else:
        last_ph_pos = gold_ed
    return first_ph_pos, last_ph_pos

class Stream_stats(object):
    '''Implements a on-line mean and variance, 
    
       it computes terms on-line to compute the median an variance from a stream
       this implementation of on-line statistics is based on:
       
       http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf

       see also https://arxiv.org/pdf/1510.04923.pdf

    '''
    
    def __init__(self, sample=0):

        # initilize mean and high order stats to 0
        self.n_ = 0.0
        self.mean_ = 0.0
        self.m2_ = 0.0 
        self.var_ = 0.0
        self.std_ = 0.0
        
        if sample:
            self._actualize(sample)

    def _actualize(self, sample):
        '''actualize deltas'''

        if sample < 0.0:
            logging.info("computed invalid NED in %s", classes_file)
            raise

        # compute the pterm used in the on-line stats
        self.n_ += 1
        delta = sample - self.mean_
        delta_n = delta / self.n_
        self.mean_ += delta_n
        self.m2_ += delta * (sample - self.mean_)

    def add(self, sample):
        '''add a sample'''
        self._actualize(sample)

    def mean(self):
        '''returns the on-line mean'''
        return self.mean_

    def var(self):
        '''return the on-line variance'''
        n_ = 2.0 if (self.n_ - 1.0) == 0 else self.n_
        self.var_ = self.m2_ / (n_ - 1.0)
        return self.var_

    def std(self):
        '''return the on-line standard error'''
        return np.sqrt(self.var())

    def n(self):
        '''return the number values used on in the computations standard error'''
        return self.n_

def nCr(n,r):
    '''Compute the number of combinations nCr(n,r)
    
    Parameters:
    -----------
    n : number of elements, integer
    r : size of the group, integer
    Returns:
    val : number of combinations 
    >> nCr(4,2)
    6
    
    >> nCr(50,2)
    1225L
    
    '''
    f = math.factorial
    
    # no negative values allow
    try:
        r_ = f(n) / f(r) / f(n-r)
    
    except:
        r_ = 0

    return r_

