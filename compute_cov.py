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

from compute_ned import read_gold_phn


# load environmental varibles
try:
    PHON_GOLD=os.environ['PHON_GOLD']
except:
    print("PHON_GOLD not set")
    sys.exit

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
    phn_gold = PHON_GOLD 
    gold = read_gold_phn(phn_gold) 

    # TODO : this code assume that the class file is build correctly but if not???
    logging.info("Parsing class file %s", classes_file)
    
    # initializing things
    classes = list()
    n_pairs = count()
    n_cross, n_within, n_overall = 0, 0, 0
    n_phones = sum([len(gold[k]['start']) for k in gold.keys()])

    count_cross = np.zeros(n_phones)
    count_within = np.zeros(n_phones)
    count_overall = np.zeros(n_phones)

    # file is decoded line by line and ned statistics are computed in 
    # a streaming to avoid using a high amount of memory
    with codecs.open(classes_file, encoding='utf8') as cfile:
        for lines in cfile:
            line = lines.strip()
            if len(line) == 0: 
                # empty line means that the class has ended and it is possilbe to compute cov 
                
                # compute the cov for all combination of intervals without replacement 
                # in group of two
                for elem1, elem2 in combinations(range(len(classes)), 2):

                    # 1. search for the intevals in the phoneme file
                    # first file 
                    b1_ = bisect_left(gold[classes[elem1][0]]['start'], classes[elem1][1])
                    e1_ = bisect_right(gold[classes[elem1][0]]['end'], classes[elem1][2])
                    # second file
                    b2_ = bisect_left(gold[classes[elem2][0]]['start'], classes[elem2][1])
                    e2_ = bisect_right(gold[classes[elem2][0]]['end'], classes[elem2][2])

                    # short time window then it not found the phonems  
                    if (e1_-b1_)  == 0 or (e2_-b2_) == 0:
                        continue
                  
                    # counting seen  
                    if classes[elem1][0] == classes[elem2][0]: # within 
                        n_within+=1
                        count_within[b1_:e1_]=1
                        count_within[b2_:e2_]=1
                        
                    else: # cross speaker 
                        n_cross+=1
                        count_cross[b1_:e1_]=1
                        count_cross[b2_:e2_]=1


                    # overall speakers = all the information
                    n_overall+=1
                    count_overall[b1_:e1_]=1
                    count_overall[b2_:e2_]=1
                    
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
           
            # getting the informati of the pairs
            else:
                fname, start, end = line.split(' ')
                classes.append([fname, float(start), float(end)])

    # logging the results
    cov_overall = count_overall.sum() / len(count_overall) 
    cov_cross = count_cross.sum() / len(count_cross) 
    cov_within = count_within.sum() / len(count_within) 

    logging.info('overall: COV={:5.2f} pairs={}'.format(cov_overall, n_overall)) 
    logging.info('cross: COV={:5.2f} pairs={}'.format(cov_cross, n_cross))
    logging.info('within: COV={:5.2f} pairs={}'.format(cov_within, n_within))


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
