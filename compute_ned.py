#!/usr/bin/env python

import sys
import os
import logging
import codecs
from bisect import bisect_left, bisect_right, bisect
from itertools import combinations, count
import argparse

import numpy as np 
import pandas as pd
import editdistance # see (1) 

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


def stream_stats(n, ned, mean_ned):
    ''' compute the straming statistics of order 2 for a ned stream 
    this implementation of stream statistics is based on 
    https://arxiv.org/pdf/1510.04923.pdf

    see also http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf

    one of the requiriments is that the variable is positive
    '''

    # check if it's a valid value
    if ned < 0.0:
        logging.info("computed invalid NED in %s", classes_file)
        raise

    # the rolling statistics ...
    n += 1
    delta = ned - mean_ned
    delta_n = delta / n
    mean_ned += delta_n
    m2 = delta * (delta - delta_n)
    return n, mean_ned, m2 


def ned_from_class(classes_file):
    '''compute the ned from the tde class file.'''
  
    ## reading the phoneme gold
    phn_gold = PHON_GOLD 
    gold = read_gold_phn(phn_gold) 

    
    # parsing the class file.
    # class file begins with the Class header,
    # following by a list of intervals and ending
    # by an space ... once the space is reached it
    # is possible to compute the ned within the class

    # TODO : this code assume that the class file is build correctly but if not???
    logging.info("Parsing class file %s", classes_file)
    
    # initializing things
    classes = list()
    neds = list()
    n_pairs = count()
    n_cross, n_within, n_overall = 0, 0, 0
    mean_cross, mean_within, mean_overall = 0, 0, 0 
    m2_cross, m2_within, m2_overall = 0, 0, 0

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
                
                # compute the ned for all combination of intervals without replacement 
                # in group of two
                for elem1, elem2 in combinations(range(len(classes)), 2):

                    # 1. search for the intevals in the phoneme file
                    
                    # first file 
                    try:
                        b1_ = bisect_left(gold[classes[elem1][0]]['start'], classes[elem1][1])
                        e1_ = bisect_right(gold[classes[elem1][0]]['end'], classes[elem1][2])
                    except KeyError:
                        logging.error("%s not in gold", classes[elem1][0])
                        continue
                    
                    # second file
                    try: 
                        b2_ = bisect_left(gold[classes[elem2][0]]['start'], classes[elem2][1])
                        e2_ = bisect_right(gold[classes[elem2][0]]['end'], classes[elem2][2])
                    except KeyError:
                        logging.error("%s not in gold", classes[elem2][0])
                        continue

                    # get the phonemes 
                    s1 = gold[classes[elem1][0]]['phon'][b1_:e1_] 
                    s2 = gold[classes[elem2][0]]['phon'][b2_:e2_]
           
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
                        neds_ = float(editdistance.eval(s1, s2)) / max(len(s1), len(s2))
                    
                    # streaming statisitcs  
                    if classes[elem1][0] == classes[elem2][0]: # within 
                        n_within, mean_within, m2_within = \
                                stream_stats(n_within, neds_, mean_within)
                        
                    else: # cross speaker 
                        n_cross, mean_cross, m2_cross = \
                                stream_stats(n_cross, neds_, mean_cross)

                    # overall speakers = all the information
                    n_overall, mean_overall, m2_overall = \
                            stream_stats(n_overall, neds_, mean_overall)
                    
                    # it will show some work is been done ...
                    #sys.stderr.write("{:5.2f}\n".format(neds_))
                    n_total = n_pairs.next()
                    if (n_total%1e6) == 0.0:
                        logging.debug("done %s pairs", n_total)

                # clean the varibles 
                classes = list()

            # if is found the label Class do nothing  
            elif line[:5] == 'Class': # the class + number + ngram if available
                pass
           
            # getting the information of the pairs
            else:
                fname, start, end = line.split(' ')
                classes.append([fname, float(start), float(end)])

    # avoid a division by 0 by setting the min value = 2.0
    n_within = 2.0 if (n_within-1.0)==0 else n_within 
    n_cross = 2.0 if (n_cross-1.0)==0 else n_cross 
    n_overall = 2.0 if (n_overall-1.0)==0 else n_overall 

    # computing the variance
    var_within = m2_within / (n_within - 1.0) 
    var_cross = m2_cross / (n_cross - 1.0)
    var_overall = m2_overall / (n_overall - 1.0)

    # logging the results
    logging.info('overall: NED=%.2f std=%.2f pairs=%d', mean_overall,
                 np.sqrt(var_overall), n_overall) 
    logging.info('cross: NED=%.2f std=%.2f pairs=%d', mean_cross, 
                 np.sqrt(var_cross), n_cross)
    logging.info('within: NED=%.2f std=%.2f pairs=%d', mean_within, 
                 np.sqrt(var_within), n_within)


def read_gold_phn(phn_gold):
    ''' read the gold phoneme file with fields : speaker/file start end phon,
    returns a dict with the file/speaker as a key and the following structure
    
    gold['speaker'] = [{'start': list(...)}, {'end': list(...), 'phon': list(...)}]
    '''
    df = pd.read_table(phn_gold, sep='\s+', header=None, encoding='utf8',
            names=['file', 'start', 'end', 'phon'])
    df = df.sort_values(by=['file', 'start']) # sorting the data

    # get the lexicon and translate to as integers
    symbols = list(set(df['phon']))
    symbol2ix = {v: k for k, v in enumerate(symbols)}
    #ix2symbols = dict((v,k) for k,v in symbol2ix.iteritems())
    df['phon'] = df['phon'].map(symbol2ix)

    # timestamps in gold (start, end) must be in acending order for fast search
    gold = {}
    for k in df['file'].unique():
        start = df[df['file'] == k]['start'].values
        end = df[df['file'] == k]['end'].values
        phon = df[df['file'] == k]['phon'].values
        assert not any(np.greater_equal.outer(start[:-1] - start[1:], 0)), 'start in phon file is not odered!!!'
        assert not any(np.greater_equal.outer(end[:-1] - end[1:], 0)), 'end in phon file is not odered!!!'
        gold[k] = {'start': list(start), 'end': list(end), 'phon': list(phon)} 
   
    return gold


if __name__ == '__main__':

    command_example = '''example:
    
        compute_ned.py file.class

    '''
    parser = argparse.ArgumentParser(epilog=command_example)
    parser.add_argument('fclass', metavar='CLASS_FILE', nargs=1, \
            help='Class file in tde format')
    args = parser.parse_args()

    # TODO: check file
    disc_class = args.fclass[0]

    get_logger(level=LOG_LEV)
    logging.info("Begining computing NED for %s", disc_class)
    ned_from_class(disc_class)
    logging.info('Finished computing NED for %s', disc_class)

