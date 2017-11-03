#!/usr/bin/env python

import sys
import math
import os
import logging
import codecs
from bisect import bisect_left, bisect_right, bisect
from itertools import combinations, count
import argparse
import ipdb


import numpy as np 
import pandas as pd
import editdistance # see (1)

from utils import read_gold_phn, check_phn_boundaries, Stream_stats, nCr

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


def func_ned(s1, s2):
    return float(editdistance.eval(s1, s2)) / max(len(s1), len(s2))


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
        # Sanity check : check last line of class file to check if it ends with a blank line as it should
        #assert cfile.readlines()[-1]=='\n', "Error : class file doesn't end with a blank line, \
        #                                     there was a problem during generation, correct it by adding blank line"
        for lines in cfile:
            line = lines.strip()
            if len(line) == 0: 
                # empty line means that the class has ended and it is possilbe to compute ned
          
                # compute the theoretical number of pairs in each class.
                # remove from that count the pairs that were not found in the gold set.
                total_expected_pairs += nCr(len(classes), 2) 
                throwaway_pairs = 0
                
                # compute the ned for all combination of intervals without replacement 
                # in group of two
                for elem1, elem2 in combinations(range(len(classes)), 2):

                    # 1. search for the intevals in the phoneme file
                    
                    # first file 
                    try:
                        b1_ = bisect_left(gold[classes[elem1][0]]['start'], classes[elem1][1])
                        e1_ = bisect_right(gold[classes[elem1][0]]['end'], classes[elem1][2])
                        b1_, e1_ = check_phn_boundaries(b1_, e1_, gold, classes, elem1)
                    except KeyError:
                        logging.error("%s not in gold", classes[elem1][0])
                        continue
                    
                    # second file
                    try: 
                        b2_ = bisect_left(gold[classes[elem2][0]]['start'], classes[elem2][1])
                        e2_ = bisect_right(gold[classes[elem2][0]]['end'], classes[elem2][2])
                        b2_, e2_ = check_phn_boundaries(b2_, e2_, gold, classes, elem2)
                    except KeyError:
                        logging.error("%s not in gold", classes[elem2][0])
                        continue

                    # get the phonemes (bugfix, don't take empty list if only 1 phone discovered)
                    try:
                        s1 = gold[classes[elem1][0]]['phon'][b1_:e1_] if e1_>b1_ \
                             else np.array([gold[classes[elem1][0]]['phon'][b1_]])
                    except:
                        # if detected phone is completely out of alignment

                        s1 = []
                    try:
                        s2 = gold[classes[elem2][0]]['phon'][b2_:e2_] if e2_>b2_ \
                             else np.array([gold[classes[elem2][0]]['phon'][b2_]])
                    except IndexError:
                        # if detected phone is completely out of alignment
                        s2 = []

                    # get transcription 
                    t1 = [ix2symbols[sym] for sym in s1]
                    t2 = [ix2symbols[sym] for sym in s2]

                    # if on or both of the word is not found in the gold, go to next pair  
                    if len(s1) == 0 and len(s2) == 0:
                        throwaway_pairs += 1
                        logging.debug("%s interv(%f, %f) and %s interv(%f, %f) not in gold", 
                                classes[elem1][0], b1_, e1_,
                                classes[elem2][0], b2_, e2_)
                        #neds_ = 1.0
                        continue
                  
                    if len(s1) == 0 or len(s2) == 0:
                        throwaway_pairs += 1
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

                    # if requested, output the transcription of current pair, along with its ned
                    if transcription: 
                       logging.info(u'{} {} {} {}\t{} {} {} {} {}'.format(classes[elem1][0], classes[elem1][1],
                                                                          classes[elem1][2], ','.join(t1),
                                                                          classes[elem2][0], classes[elem2][1],
                                                                          classes[elem2][2], ','.join(t2), ned))

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
                    
                    # it will show some work has been done ...
                    n_total = n_pairs.next()
                    if (n_total%1e6) == 0.0 and n_total>0:
                        logging.debug("done %s pairs", n_total)

                # clean the varibles that contains the tokens
                total_expected_pairs -= throwaway_pairs
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

