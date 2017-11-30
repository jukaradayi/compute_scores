#!/usr/bin/env python
#
# author = julien karadayi
#
# Check the consistency of the class file you're about to evaluate.
# First, check if the format is good (in particular, check if the line
# of the file is a blank line).
# Then, check that none of the intervals found in the class file contains
# silence. To do that, we build an interval tree with the intervals of
# silences, using the vad, and check for each interval found by the
# algorithm that none overlap with the silence. Overlap of a very small
# number of frames (2 ?) is permissible.
# Some solutions are suggested to treat the non conform pairs.
#
# Input
# - Class File : file containing the found pairs, in the ZR challenge format:
#
#       Class1
#       file1 on1 off1
#       file2 on2 off2
#
#       Class2
#       [...]
#
# - VAD : VAD corresponding to the corpus on which you
#
# Output
# - List of all the intervals that overlap with Silence (outputs nothing if
#   all the intervals are okay)

import os
import sys
import intervaltree
import argparse
import ipdb
from collections import defaultdict


def create_silence_tree(vad_file):
    """Read the vad, and create an interval tree
    with the silence intervals.
    We are interested in the silence intervals, so for each
    file in the vad, sort all the intervals, get the min and max
    of the timestamps, and take the complementary of the set 
    of intervals delimitated by the vad.

    Input
        - vad_file: path to the vad file indicating segments of speech
    Output
        - silence_tree: dict {fname: IntervalTree} : for each file in the vad
          keep an interval tree of silences
    """

    assert os.path.isfile(vad_file), "ERROR: VAD file doesn't exist, check path"
    
    vad = defaultdict(list)
    # Read vad and create dict {file: [intervals]}
    with open(vad_file, 'r') as fin:
        speech = fin.readlines()

        for line in speech:
            fname, on, off = line.strip('\n').split()
            vad[fname].append((float(on), float(off)))

    # Sort the intervals by their onset for each file,
    # and create a "Silence activity detection" by 
    # taking the complementary of each set of intervals
    silence_tree = dict()
    for fname in vad:

        silences = []

        # sort intervals by their onset - there is no overlap
        # between intervals.
        vad[fname].sort()
        on_min, on_max = vad[fname][0][0], vad[fname][-1][1]
        if on_min > 0:
            silences.append((0, on_min))
        for it, times in enumerate(vad[fname]):
            on, off = times
            if it == 0:
                # skip first interval
                prev_off = off
                continue

            # add the gap between previous interval and current
            # interval as "silence"
            if prev_off > on:
                silences.append((prev_off, on))
            
            # keep previous offset
            prev_off = off

        # Create interval tree for each file in the vad.
        silence_tree[fname] = intervaltree.IntervalTree.from_tuples(silences)

    return silence_tree


def parse_class_file(class_file):
    """Create list of all the intervals found"""
    
    assert os.path.isfile(class_file), "ERROR: class file not found"
    
    intervals_list = defaultdict(list)
    # Read Class file. Keep lines if they're not blank and 
    # don't start by "Class", because those are the ones 
    # that contain the found intervals.
    with open(class_file, 'r') as fin:
        classes = fin.readlines()
        
        for line in classes:
            # skip blanks & "Class [...]"
            if line.startswith('Class') or len(line.strip('\n')) == 0:
                continue
            
            # retrieve file name and interval
            # ipdb.set_trace()
            fname, on, off = line.strip('\n').split()
            intervals_list[fname].append((float(on), float(off)))
            
    
    return intervals_list


def check_intervals_foud(intervals_list, silence_tree):
    """Check that None of the intervals found overlap
    with Silence.
    If an interval overlaps with silence, check three cases :
        1 - The silence is completely inside the interval
            found
        2 - The silence overlaps with one border of the
            found interval
        3 - Any combination (with repetition) of these two cases

    Suggested treatements for such pairs:
        - For case 1: cut interval in half (decide how to reconstruct
            the found pairs), remove pair.
        - For case 2: trim the interval by removing the part that
            overlaps with silence
        - For case 3: if only a combination of case 2, trim pair,
            if case 1 is also occuring, cut interval, rinse and repeat.

    Input: 
        - intervals_list: dictionnary {fname: [list of intervals]},
                          that returns a list of all found intervals
                          for a filename
        - silence_tree: dictionnary that returns an interval tree 
                        built with the intervals of silence for each
                        filename.
    Output:
        - bad_pairs: dict {fname: [(on_pair, off_pair, on_SIL, offSIL)]}
                     that returns for each filename a list of tuples 
                     that indicate the intervals that overlap with silence 
                     (and the silences they overlap with)
    """
    bad_pairs = defaultdict(list)
    for fname in intervals_list:
        for on, off in intervals_list[fname]:
            ov = silence_tree[fname].search(on, off)

            # check if current interval overlaps with silences,
            # if so, add to bad_pairs.
            if len(ov) > 0:
                bad_pairs[fname].append((on, off))


    return bad_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('vad_file', metavar='<VAD>',
                        help="""text file containing all the intervals of"""
                             """speech""")
    parser.add_argument('class_file', metavar='<CLASS>',
                        help="""Class file, in the ZR Challenge format:"""
                             """       """
                             """       Class1"""
                             """       file1 on1 off1"""
                             """       file2 on2 off2"""
                             """       """
                             """       Class2"""
                             """       [...]"""
                             """       """)
    args = parser.parse_args()
    sil_tree = create_silence_tree(args.vad_file)
    disc_int = parse_class_file(args.class_file)
    bad_pairs = check_intervals_foud(disc_int, sil_tree)
    print bad_pairs
