import argparse
import csv
import logging
import math
import shutil
import sys
import os
import automata as atma
import automata.FST.options as options
import automata.HDL.hdl_generator as hd_gen
import automata.FST.group_compiler as gc
import automata.FST.compilation_statistics as compilation_statistics

sys.setrecursionlimit(25000)

def process(file_groups, name, print_compression_stats=False, options=None):
    file_groups = [os.path.join(file_groups, f) for f in os.listdir(file_groups) if os.path.isfile(os.path.join(file_groups, f))]
    print "Extracting ", len(file_groups), " groups"
    # Output is: HDL file with the important automata
    # selected + table programming.

    automata_components = []
    index = 0
    for file in file_groups:
        if not file.endswith('.anml'):
            continue
        if options.print_file_info:
            print "Group Index: ", index, "comes from file", file
            index += 1

        automatas = atma.parse_anml_file(file)
        # Need to remove ors to avoid 'special elements' that
        # prevent us from getting the connected components.
        automatas.remove_ors()
        automata_components.append(automatas.get_connected_components_as_automatas())

    groups, assignments = gc.compile(automata_components, options)

    if print_compression_stats:
        self_compiles = 0
        other_compiles = 0
        for i in range(len(assignments)):
            for j in range(len(assignments[i])):
                compilation_index = assignments[i][j]
                if compilation_index.i == i and compilation_index.j == j:
                    self_compiles += 1
                else:
                    # Compiled to something else
                    other_compiles += 1

                    print "Achieved compilation from ", str(groups[i][j].algebra)
                    print " to ", str(groups[compilation_index.i][compilation_index.j].algebra)

        print "COMPILATION STATISTICS: self compiles = ", self_compiles
        print "COMPILATION STATISTICS: other compiles = ", other_compiles
        print "COMPILATION STATISTICS: unifications = ", compilation_statistics.single_state_unification_success
        print "COMPILATION STATISTICS: Of those,  ", compilation_statistics.exact_same_compilations, " were equal"
        print "Tried to cross compile ", compilation_statistics.executed_comparisons, "regexes to each other"
        print "Avoided ", compilation_statistics.cutoff_comparisons, "detailed comparison attempts with heuristics"
        print "Of the algebras ", compilation_statistics.failed_algebra_computations, " failed (likely) due to incompleteness of current implementation"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('anml_file_groups')
    parser.add_argument('--compression-stats', default=False, dest='compression_stats', action='store_true')

    options.add_to_parser(parser)

    args = parser.parse_args()

    process(args.anml_file_groups, args.name, args.compression_stats, options.create_from_args(args))
