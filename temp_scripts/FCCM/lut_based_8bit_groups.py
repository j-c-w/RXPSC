import argparse
import csv
import logging
import math
import shutil
import os
import automata as atma
import automata.FST.options as options
import automata.HDL.hdl_generator as hd_gen
import automata.FST.group_compiler as gc

def process(file_groups, name, print_compression_stats=False, options=None):
    # Output is: HDL file with the important automata
    # selected + table programming.

    automata_components = []
    for file in file_groups:
        if file.endswith('mnrl'):
            print("ERROR: Entered a MNRL file, likely going to fail")

        automatas = atma.parse_anml_file(file)
        # Need to remove ors to avoid 'special elements' that
        # prevent us from getting the connected components.
        automatas.remove_ors()
        automata_components.append(automatas.get_connected_components_as_automatas())

    groups, assignments = gc.compile(automata_components, options)

    if print_compression_stats:
        self_compiles = 0
        equality = 0
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

                    if compilation_index.conversion_machine.isempty():
                        equality += 1
        print "COMPILATION STATISTICS: self compiles = ", self_compiles
        print "COMPILATION STATISTICS: Of those, ", equality, " were equal"
        print "COMPILATION STATISTICS: other compiles = ", other_compiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('anml_file_groups', nargs='+')
    parser.add_argument('--compression-stats', default=False, dest='compression_stats', action='store_true')

    options.add_to_parser(parser)

    args = parser.parse_args()

    process(args.anml_file_groups, args.name, args.compression_stats, options.create_from_args(args))
