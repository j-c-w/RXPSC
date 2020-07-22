import argparse
import csv
import logging
import math
import shutil
import os
import automata as atma
import automata.HDL.hdl_generator as hd_gen
import automata.FST.group_compiler as gc

def process(file_groups, name):
    # Output is: HDL file with the important automata
    # selected + table programming.

    automata_components = []
    for file in file_groups:
        if file.endswith('mnrl'):
            print("ERROR: Entered a MNRL file, likely going to fail")

        automatas = atma.parse_anml_file(file)
        automata_components.append(automatas.get_connected_components_as_automatas())

    onboard_automata, table_configurations = gc.compile(automata_components)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('anml_file_groups', nargs='+')

    args = parser.parse_args()

    process(args.anml_file_groups, args.name)
