#!/usr/bin/env python

import sys
import os
import logging
import codecs
from bisect import bisect_left, bisect_right, bisect
from itertools import izip, izip_longest, combinations, count
from collections import defaultdict
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

try:
    disc_class = sys.argv[1]
except:
    print("use compute_ned.py [FILE]")
    sys.exit


# if LOG not exist then use the stderr  
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


# from itertools examples  
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def ned_from_aren(classes_file):
    ''' get the intervals from Arens out.1 '''

    # decoding the output from Aren's ZRTools 
    logging.info("Parsing out.1 file %s", classes_file)
    dpairs = defaultdict(list)
    with codecs.open(classes_file, encoding='utf8') as fdisc:
        for line in fdisc.readlines():
            l = line.strip().split(' ')
            if len(l) == 2: # names of files
                pair_files = ' '.join(l)
            elif len(l) == 5 or len(l) == 7: # 5 from knn, 7 from Aren result
                dpairs[pair_files].append([float(x) for x in l])
            else:
                print(l)
                logging.error("Error in file %s", classes_file) 
                logging.error("%s", l)
                sys.exit()

    logging.info("Generating pairs")    
    b1, e1, b2, e2, n1, n2, se, cs = ([] for i in range(8)) 
    class_num = count() # the class number
    for file_pair in dpairs.keys():
        fileX, fileY = file_pair.split(' ')
        # same speaker = 1 if within- and 0 if cross-speaker
        se_ = 1 if fileX==fileY else 0 
        
        for res in dpairs[file_pair]:
            n1 += [fileX]
            b1 += [res[0]/100.0] # res[0] is in frames (1/100s)
            e1 += [res[1]/100.0]
            
            n2 += [fileY]
            b2 += [res[2]/100.0]
            e2 += [res[3]/100.0]
           
            se += [se_]
            cs += [class_num.next()]

    logging.info("Joining all intevals and names")    
    intervals = np.vstack((b1, e1, b2, e2, se, cs)).T
    names = np.vstack((n1, n2)).T
 
    return intervals, names


def stream_stats(n, ned, mean_ned):
    ''' '''
    n += 1                                      
    delta_0 = neds - mean_ned           
    mean_ned += delta_0 / n       
    delta_1 = neds - mean_ned           
    var_ned += delta_within_0 * delta_within_1  




def ned_from_class(classes_file):
    '''compute the ned from the tde class file'''
  
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
    classes = list()
    neds = list()
    n_pairs = count()
    n_cross, n_within, n_overall = 0, 0, 0
    mean_ned_cross, mean_ned_within, mean_ned_overall = 0, 0, 0 
    var_ned_cross, var_ned_within, var_ned_overall = 0, 0, 0

    # file is decoded line by line and ned is computed online
    # to avoid using a huge amount of memory 
    with codecs.open(classes_file, encoding='utf8') as cfile:
        for lines in cfile:
            line = lines.strip()
            if len(line) == 0: 
                # empty line means that the class has
                # ended and it is possilbe to compute the ne
                #logging.debug("Doing class %s", class_num)
                for elem1, elem2 in combinations(range(len(classes)), 2):

                    b1_ = bisect_left(gold[classes[elem1][0]]['start'], classes[elem1][1])
                    e1_ = bisect_right(gold[classes[elem1][0]]['end'], classes[elem1][2])
                    
                    b2_ = bisect_left(gold[classes[elem2][0]]['start'], classes[elem2][1])
                    e2_ = bisect_right(gold[classes[elem2][0]]['end'], classes[elem2][2])
                  
                    if b1_ == e1_:
                        e1_ = b1_+1
                    if b2_ == e2_:
                        e2_ == b2_+1
           
                    s1 = gold[classes[elem1][0]]['phon'][b1_:e1_] 
                    s2 = gold[classes[elem2][0]]['phon'][b2_:e2_]
           
                    # short time window then it not found the phonems  
                    if len(s1) == 0 and len(s2) == 0:
                        #neds_ = 1.0
                        continue
                  
                    # ned for an empty string and a string is 1
                    if len(s1) == 0 or len(s2) == 0:
                        #neds_ = 1.0
                        continue
                    else:
                        neds_ = float(editdistance.eval(s1, s2)) / max(len(s1), len(s2))
                    

                    # streaming statisitcs from from https://arxiv.org/pdf/1510.04923.pdf 
                    # this equations are valid for positive values (ned>0)
                    if classes[elem1][0] == classes[elem2][0]: # within 
                        n_within += 1
                        delta_within_0 = neds_ - mean_ned_within
                        mean_ned_within += delta_within_0 / n_within 
                        delta_within_1 = neds_ - mean_ned_within 
                        var_ned_within += delta_within_0 * delta_within_1

                    else: # cross speaker 
                        n_cross += 1
                        delta_cross_0 = neds_ - mean_ned_cross
                        mean_ned_cross += delta_cross_0 / n_cross 
                        delta_cross_1 = neds_ - mean_ned_cross 
                        var_ned_cross += delta_cross_0 * delta_cross_1

                    # overall speakers
                    # https://arxiv.org/abs/1510.04923 
                    n_overall += 1
                    delta_overall_0 = neds_ - mean_ned_overall
                    mean_ned_overall += delta_overall_0 / n_overall 
                    delta_overall_1 = neds_ - mean_ned_overall 
                    var_ned_overall += delta_overall_0 * delta_overall_1

                    #sys.stderr.write("{:5.2f}\n".format(neds_))
 
                    #neds.append([neds_, same_speaker, class_num])
                    n_total = n_pairs.next()
                    if (n_total%1e6) == 0.0:
                        logging.debug("done %s pairs", n_total)

                # clean the varibles 
                classes = list()

            elif line[:5] == 'Class': # the class + number + ngram if available
                class_num = int(line.split(' ')[1]) # Class classnb [anything]
            
            else:
                fname, start, end = line.split(' ')
                classes.append([fname, float(start), float(end)])

    tot_within = 1.0 or n_within-1.0
    tot_cross = 1.0 or n_cross-1.0
    tot_overall = 1.0 or n_overall-1.0
    var_ned_within = var_ned_within / tot_within
    var_ned_cross = var_ned_cross / tot_cross
    var_ned_overall = var_ned_overall / tot_overall

    logging.info('overall: NED={:5.2f} std={:5.2f} pairs={}'.format(mean_ned_overall,
                 np.sqrt(var_ned_overall), n_overall)) 
    
    logging.info('cross: NED={:5.2f} std={:5.2f} pairs={}'.format(mean_ned_cross, 
                 np.sqrt(var_ned_cross), n_cross))

    logging.info('within: NED={:5.2f} std={:5.2f} pairs={}'.format(mean_ned_within, 
                 np.sqrt(var_ned_within), n_within))


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


    get_logger(level=logging.DEBUG)
    logging.info("Begining computing NED for %s", disc_class)
    ned_from_class(disc_class)
    logging.info('Finished computing NED for %s', disc_class)

