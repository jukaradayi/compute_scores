#!/usr/bin/env python

import sys
import os
import logging
import codecs
from bisect import bisect_left, bisect_right, bisect
from itertools import combinations, count
from collections import defaultdict
import argparse
import pdb


import numpy as np
import pandas as pd

from compute_ned import read_gold_phn


# load environmental varibles
try:
    PHON_GOLD=os.environ['PHON_GOLD']
except:
    print("PHON_GOLD not set")
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
    phn_gold = PHON_GOLD 
    gold = read_gold_phn(phn_gold)

    # get the vector ngram_mask filled with 1s at positions in the gold vector
    # with repeted ngrams more that once in the gold. 
    ngram_mask = find_mask_ngrams(gold, ngrams=3) 

    # TODO : this code assume that the class file is build correctly but if not???
    logging.info("Parsing class file %s", classes_file)

    # initializing things
    classes = list()
    n_pairs = count()
    n_overall = 0
    n_phones = sum([len(gold[k]['start']) for k in gold.keys()])
    count_phonemes = {k:np.zeros(len(gold[k]['start'])) for k in gold.keys()}

    # file is decoded line by line and ned statistics are computed in
    # a streaming to avoid using a high amount of memory
    with codecs.open(classes_file, encoding='utf8') as cfile:
        for lines in cfile:
            line = lines.strip()
            if len(line) == 0:
                # empty line means that the class has ended and it is possilbe to compute cov

                # compute the cov for the found intervals
                for elem1 in range(len(classes)):
                    file_name = classes[elem1][0]

                    # search for intevals in the phoneme file
                    b1_ = bisect_left(gold[file_name]['start'], classes[elem1][1])
                    e1_ = bisect_right(gold[file_name]['end'], classes[elem1][2])

                    # overall speakers = all the information
                    n_overall+=1
                    count_phonemes[file_name][b1_:e1_] = 1

                    # it will show some work is been done ...
                    n_total = n_pairs.next()
                    if (n_total%1e4) == 0.0:
                        logging.debug("done %s intervals", n_total)

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
    count_overall = np.array([])
    for file_name in count_phonemes.keys():
        count_overall = np.append(count_overall, count_phonemes[file_name])
    
    cov_overall = np.sum(count_overall.astype('int') & ngram_mask.astype('int')) / ngram_mask.sum() 
    #cov_overall = np.sum(count_overall.astype('int')) / ngram_mask.sum() 
    #cov_overall = count_overall.sum() / n_phones 
    logging.info('overall: COV={:.3f} intervals={}'.format(cov_overall, n_overall))


def find_ngrams(input_list, n=3):
    '''return a list with n-grams from the input list'''
    # http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    return zip(*[input_list[i:] for i in range(n)])


def find_mask_ngrams(gold, ngrams=3):
    ''' create a mask with the size of the gold for ngrams that are found more than once in the corpus'''

    n_phones = sum([len(gold[k]['start']) for k in gold.keys()])
    mask = np.zeros(n_phones) 
    
    all_grams = defaultdict(int)
    seen_once = defaultdict(list)
    high_index = 0
    for k in gold.keys():
        phns = gold[k]['phon']
       
        # make a list with all the phon-ngrams and their indexes respect to all corpus
        indexes = np.arange(high_index, high_index+len(phns))
        index_ngrams = find_ngrams(indexes, ngrams) # list of indexes of the n-grams
        phon_ngrams = find_ngrams(phns, ngrams) # list of n-grams
        high_index+=len(phns)
        
        for n_, n_gram in enumerate(phon_ngrams):
            # convert the n-grams to a single hash/key
            n_g = ' '.join([str(x) for x in n_gram])
            all_grams[n_g]+=1 # track the number of times the n-grams has been seen
            if all_grams[n_g] > 1: # if see more than once, then include in the mask
                mask[index_ngrams[n_][0]:index_ngrams[n_][-1]] = 1
                
                # also include the first n-grams
                if n_g in seen_once:
                    seen_ngram = seen_once.pop(n_g)
                    mask[seen_ngram[0][0]:seen_ngram[0][-1]] = 1
                
            else: # first time that the n-gram has been seen
                seen_once[n_g].append(index_ngrams[n_]) 
    return mask


def read_gold_class(class_gold, gold_phn):
    '''read the class gold file that contains the gold tokens, return a mask with the
    size of the gold phonemes with 1s in the places where phonemes are present in the class'''
  
    # create the counting vector. Gold tokens could covers less found phonemes   
    n_phones = sum([len(gold_phn[k]['start']) for k in gold_phn.keys()])
    mask = np.zeros(n_phones)

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
    
    # find all found fragments in gold and mark them in mask
    for speaker in tokens_by_spaker.keys():
        # search for intevals in the phoneme file                             
        for interval in tokens_by_spaker[speaker]:
            
            b1_ = bisect_left(gold_phn[speaker]['start'], interval[0])
            e1_ = bisect_right(gold_phn[speaker]['end'], interval[1]) 
            mask[b1_:e1_] = 1

    return mask


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
