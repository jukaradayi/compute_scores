#!/usr/bin/env python

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

import difflib
#from joblib import Parallel, delayed

# (1) I checked various edit-distance implementations (see https://github.com/aflc/editdistance)
# and I found that 'editdistance' gives the same result than our implementation of the
# distance (https://github.com/bootphon/tde, module tde.substrings.levenshtein) and it's a bit faster 

# load environmental varibles
try:
    PHON_GOLD=os.environ['PHON_GOLD']
except:
    print("PHON_GOLD not set")
    sys.exit

# if LOG environment doesnt exist then use the stderr  
try:
    LOG = os.environ['LOG_NED']
except:
    LOG = 'test.log' 

#LOG_LEV = logging.ERROR
LOG_LEV = logging.DEBUG
#LOG_LEV = logging.INFO

# configuration of logging
def get_logger(level=logging.WARNING):
    FORMAT = '%(asctime)s - {} - %(levelname)s - %(message)s'.format(disc_class)
    #logging.basicConfig(filename=LOG, format=FORMAT, level=LOG_LEV)
    logging.basicConfig(stream=sys.stdout, format=FORMAT, level=LOG_LEV)


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


def func_ned(s1, s2):
    return float(editdistance.eval(s1, s2)) / max(len(s1), len(s2))

def check_phn_boundaries(gold_bg, gold_ed, gold, classes, elem):
    ''' check boundaries of discovered phone.
        If discovered "word" contains 50% of a phone, or more than
        30ms of a phone, we consider that phone discovered.
    '''
    # get discovered phones timestamps
    spkr, disc_bg, disc_ed = classes[elem]
    if disc_bg == 0.6725:
        pdb.set_trace()
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


def ned_from_class(classes_file, transcription):
    '''compute the ned from the tde class file.'''
  
    ## reading the phoneme gold
    phn_gold = PHON_GOLD 
    gold, ix2symbols = read_gold_phn(phn_gold) 

    
    # parsing the class file.
    # class file begins with the Class header,
    # following by a list of intervals and ending
    # by an space ... once the space is reached it
    # is possible to compute the ned within the class

    # TODO : this code assume that the class file is build correctly but if not???
    logging.info("Parsing class file %s", classes_file)
    
    # initializing variables used on the streaming computations 
    classes = list()
    n_pairs = count() # used to debug
    total_expected_pairs = 0

    # objects with the streaming statistics
    cross = Stream_stats()
    within = Stream_stats()
    overall = Stream_stats()

    # to compute NED you'll need the following steps:
    # 1. search for the pair of words the correponding 
    #    phoneme anotations.
    # 2. compute the Levenshtein distance between the two string.
    # 
    # see bellow 

    # file is decoded line by line and ned statistics are computed in 
    # a streaming to avoid using a high amount of memory
    with codecs.open(classes_file, encoding='utf8') as cfile:
        for lines in cfile:
            line = lines.strip()
            if len(line) == 0: 
                # empty line means that the class has ended and it is possilbe to compute ned
          
                # compute the theoretical number of pairs in each class
                total_expected_pairs += nCr(len(classes), 2) 

                
                # compute the ned for all combination of intervals without replacement 
                # in group of two
                for elem1, elem2 in combinations(range(len(classes)), 2):

                    # 1. search for the intevals in the phoneme file
                    
                    # first file 
                    try:
                        b1_ = bisect_left(gold[classes[elem1][0]]['start'], classes[elem1][1])
                        e1_ = bisect_right(gold[classes[elem1][0]]['end'], classes[elem1][2])
                        b1_, e1_ = check_phn_boundaries(b1_, e1_, gold, classes, elem1)
                        #b1_ = b1_bis
                        #e1_ = e1_bis
                    except KeyError:
                        logging.error("%s not in gold", classes[elem1][0])
                        continue
                    
                    # second file
                    try: 
                        b2_ = bisect_left(gold[classes[elem2][0]]['start'], classes[elem2][1])
                        e2_ = bisect_right(gold[classes[elem2][0]]['end'], classes[elem2][2])
                        #pdb.set_trace()
                        b2_, e2_ = check_phn_boundaries(b2_, e2_, gold, classes, elem2)
                        #pdb.set_trace()
                    except KeyError:
                        logging.error("%s not in gold", classes[elem2][0])
                        continue

                    # get the phonemes (bugfix, don't take empty list if only 1 phone discovered)
                    try:
                        s1 = gold[classes[elem1][0]]['phon'][b1_:e1_] if e1_>b1_ else np.array([gold[classes[elem1][0]]['phon'][b1_]])
                    except:
                        s1 = []
                    try:
                        s2 = gold[classes[elem2][0]]['phon'][b2_:e2_] if e2_>b2_ else np.array([gold[classes[elem2][0]]['phon'][b2_]])
                    except IndexError:
                        # if detected phone is completely out of alignment
                        s2 = []

                    # get transcription 
                    t1 = [ix2symbols[sym] for sym in s1]
                    t2 = [ix2symbols[sym] for sym in s2]

                    # short time window then it not found the phonems  
                    if len(s1) == 0 and len(s2) == 0:
                        logging.debug("%s interv(%f, %f) and %s interv(%f, %f) not in gold", 
                                classes[elem1][0], b1_, e1_,
                                classes[elem2][0], b2_, e2_)
                        #neds_ = 1.0
                        continue
                  
                    # ned for an empty string and a string is 1
                    if len(s1) == 0 or len(s2) == 0:
                        #neds_ = 1.0
                        if s1 == 0:
                            logging.debug("%s interv(%f, %f) not in gold",
                                    classes[elem1][0], b1_, e1_)
                        else:
                            logging.debug("%s interv(%f, %f) not in gold",
                                    classes[elem2][0], b2_, e2_)
                        continue
                    else:
                        # 2. compute the Levenshtein distance and NED
                        ned = func_ned(s1, s2)
                    if transcription: 
                       logging.info(u'{} {} {} {}\t{} {} {} {} {}'.format(classes[elem1][0], classes[elem1][1], classes[elem1][2], ','.join(t1),
                                                                  classes[elem2][0], classes[elem2][1], classes[elem2][2], ','.join(t2), ned))
                    #python standard library difflib that is not the same that levenshtein
                    #it does not yield minimal edit sequences, but does tend to 
                    #yield matches that look right to people
                    # neds_ = 1.0 - difflib.SequenceMatcher(None, s1, s2).real_quick_ratio()
                    
                    # streaming statisitcs  
                    if classes[elem1][0] == classes[elem2][0]: # within 
                        within.add(ned)
                        
                    else: # cross speaker 
                        cross.add(ned)

                    # overall speakers = all the information
                    overall.add(ned)
                    
                    # it will show some work is been done ...
                    n_total = n_pairs.next()
                    if (n_total%1e6) == 0.0 and n_total>0:
                        logging.debug("done %s pairs", n_total)

                # clean the varibles that contains the tokens
                classes = list()

            # if is found the label Class do nothing  
            elif line[:5] == 'Class': # the class + number + ngram if available
                pass
           
            # getting the information of the pairs
            else:
                fname, start, end = line.split(' ')
                classes.append([fname, float(start), float(end)])

    # logging the results
    logging.info('overall: NED=%.2f std=%.2f pairs=%d (%d total pairs)', overall.mean(),
                 overall.std(), overall.n(), total_expected_pairs) 
    logging.info('cross: NED=%.2f std=%.2f pairs=%d', cross.mean(), 
                 cross.std(), cross.n())
    logging.info('within: NED=%.2f std=%.2f pairs=%d', within.mean(), 
                 within.std(), within.n())
    return overall.mean(), cross.mean(), within.mean()


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


if __name__ == '__main__':

    command_example = '''example:
    
        compute_ned.py file.class

    '''
    parser = argparse.ArgumentParser(epilog=command_example)
    parser.add_argument('fclass', metavar='CLASS_FILE', nargs=1, \
            help='Class file in tde format')
    parser.add_argument('--transcription', action='store_true', \
            help='Enable to output complete transcription of pairs found')
    args = parser.parse_args()

    # TODO: check file
    disc_class = args.fclass[0]

    get_logger(level=LOG_LEV)
    logging.info("Begining computing NED for %s", disc_class)
    ned_from_class(disc_class, args.transcription)
    logging.info('Finished computing NED for %s', disc_class)

