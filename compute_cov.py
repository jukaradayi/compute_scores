#!/usr/bin/env python

import sys
import os
import logging
import codecs
from bisect import bisect_left, bisect_right, bisect
from itertools import combinations, count
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd

from compute_ned import read_gold_phn


# load environmental varibles
try:
    PHON_GOLD=os.environ['PHON_GOLD']
except:
    print("PHON_GOLD not set")
    sys.exit()

try:
    CLASS_GOLD=os.environ['CLASS_GOLD']
except:
    print("CLASS_GOLD not set")
    sys.exit()

# if LOG environment doesnt exist then use the stderr
try:
    LOG = os.environ['LOG_COV']
except:
    LOG = 'test.log'

#LOG_LEV = logging.ERROR
LOG_LEV = logging.DEBUG
#LOG_LEV = logging.INFO

def get_logger(level=logging.WARNING):
    FORMAT = '%(asctime)s - {} - %(levelname)s - %(message)s'.format(disc_class)
    logging.basicConfig(stream=sys.stdout, format=FORMAT, level=LOG_LEV)


def cov_from_class(classes_file):
    '''compute the cov from the tde class file.'''

    ## reading the phoneme gold
    class_gold = CLASS_GOLD
    gold, count_gold = read_gold_class(class_gold)

    # TODO : this code assume that the class file is build correctly but if not???
    logging.info("Parsing class file %s", classes_file)

    # initializing things
    classes = list()
    n_pairs = count()
    n_overall = 0
    n_phones = sum([len(gold[k]['start']) for k in gold.keys()])
    count_overall = np.zeros(n_phones)

    # file is decoded line by line and ned statistics are computed in
    # a streaming to avoid using a high amount of memory
    with codecs.open(classes_file, encoding='utf8') as cfile:
        for lines in cfile:
            line = lines.strip()
            if len(line) == 0:
                # empty line means that the class has ended and it is possilbe to compute cov

                # compute the cov for the found intervals
                for elem1 in range(len(classes)):

                    # search for intevals in the phoneme file
                    b1_ = bisect_left(gold[classes[elem1][0]]['start'], classes[elem1][1])
                    e1_ = bisect_right(gold[classes[elem1][0]]['end'], classes[elem1][2])

                    # overall speakers = all the information
                    n_overall+=1
                    count_overall[b1_:e1_] = 1

                    # print len(count_overall[b1_:e1_]) 
                        
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

    # logging the results
    cov_overall = np.sum(count_overall.astype('int') & count_gold.astype('int')) / count_gold.sum() 
    #cov_overall = count_overall.sum() / n_phones 
    logging.info('overall: COV={:.2f} elements={}'.format(cov_overall, n_overall))


def read_gold_class(class_gold):
    '''read the class gold file, it contains the gold tokens'''
  
    ## reading the phoneme gold 
    phn_gold = PHON_GOLD 
    gold = read_gold_phn(phn_gold) 

    # create the counting vector. Gold tokens could covers less found phonemes   
    n_phones = sum([len(gold[k]['start']) for k in gold.keys()])
    count_gold = np.zeros(n_phones)

    # decode class gold file and store intervals by speaker
    tokens_by_spaker = defaultdict(list)
    with codecs.open(class_gold, encoding='utf8') as cfile:
        for lines in cfile:
            line = lines.strip()
            if line[:5] == 'Class' or len(line) == 0:
                pass

            else:
                fname, start, end = line.split(' ')
                tokens_by_spaker[fname].append([float(start), float(end)])
    
    # find all found fragments in gold and mark them in count_gold
    for speaker in tokens_by_spaker.keys():
        # search for intevals in the phoneme file                             
        for interval in tokens_by_spaker[speaker]:
            
            b1_ = bisect_left(gold[speaker]['start'], interval[0])
            e1_ = bisect_right(gold[speaker]['end'], interval[1]) 
            count_gold[b1_:e1_] = 1

    return gold, count_gold


if __name__ == '__main__':

    command_example = '''example:

        compute_cov.py file.class

    '''
    parser = argparse.ArgumentParser(epilog=command_example)
    parser.add_argument('fclass', metavar='CLASS_FILE', nargs=1, \
            help='Class file in tde format')
    args = parser.parse_args()

    # TODO: check file
    disc_class = args.fclass[0]

    get_logger(level=LOG_LEV)
    logging.info("Begining computing COV for %s", disc_class)
    cov_from_class(disc_class)
    logging.info('Finished computing COV for %s', disc_class)
