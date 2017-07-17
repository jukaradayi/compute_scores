#!/usr/bin/env python

import sys
import argparse
from collections import defaultdict
from itertools import count


def dec_zrtools(f_disc_pairs):
    '''Decode Aren ZRTools file and output to the stdout the tde class file'''

    # decoding the file
    dpairs = defaultdict(list)
    with open(f_disc_pairs) as fdisc:
        for line in fdisc.readlines():
            l = line.strip().split(' ')
            if len(l) == 2: # names of files
                pair_files = ' '.join(l)
            elif len(l) == 7: # the resulting pairs
                dpairs[pair_files].append([float(x) for x in l])
            elif len(l)==6: # old version of ZRTool files 
                dpairs[pair_files].append([float(x) for x in l])
            else:
                print(l)
                sys.exit()

    # print the class file from the decoded ZRTools file 
    n = count()
    for file_pair in dpairs.keys():
        fileX, fileY = file_pair.split(' ')
        for res in dpairs[file_pair]:
            t_ = 'Class {}\n'.format(n.next())
            t_+= '{} {:5.4f} {:5.4f}\n'.format(fileX, res[0]/100.0, res[1]/100.0)
            t_+= '{} {:5.4f} {:5.4f}\n'.format(fileY, res[2]/100.0, res[3]/100.0)
            t_+= '\n'
            sys.stdout.write(t_)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('zrtools', metavar='ZRTOOLS_FILE', nargs=1, \
            help='Aren ZRTools file format')
    args = parser.parse_args()
    zfile = args.zrtools[0]
    dec_zrtools(zfile)
