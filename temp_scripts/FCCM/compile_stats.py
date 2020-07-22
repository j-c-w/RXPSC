import argparse
import csv
import logging
import math
import shutil
import os
import automata as atma
import automata.HDL.hdl_generator as hd_gen
import automata.FST.group_compiler as gc

def process(inpfile, name):
    # Output is: HDL file with the important automata
    # selected + table programming.

    if inpfile.endswith('mnrl'):
        print("ERROR: Entered a MNRL file, likely going to fail")

    automatas = atma.parse_anml_file(inpfile)
    automata_components = automatas.get_connected_components_as_automatas()

    gc.compile_statistics(automata_components)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('anml_file')

    args = parser.parse_args()

    process(args.anml_file, args.name)
