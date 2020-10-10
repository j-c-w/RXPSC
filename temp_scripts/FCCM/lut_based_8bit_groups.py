import argparse
import csv
import logging
import math
import shutil
import sys
import os
import time
import automata as atma
import automata.FST.options as options
import automata.HDL.hdl_generator as hd_gen
import automata.FST.group_compiler as gc
import automata.FST.compilation_statistics as compilation_statistics

sys.setrecursionlimit(25000)

def process(file_groups, name, print_compression_stats=False, options=None):
    start_time = time.time()
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

    assignments = gc.compile(automata_components, options)

    if print_compression_stats:
        self_compiles = len(assignments)
        other_compiles = 0
        nodes_required = 0
        nodes_saved = 0
        for assignment in assignments:
            other_compiles += len(assignment.translators)
            nodes_required += len(assignment.physical_automata.nodes)
            for supported_automata in assignment.supported_automata:
                nodes_saved += len(supported_automata.nodes)
        # Don't print the results of the comparison if we are doing --compile-only (which skips the comparisons)
        if not options.compile_only:
            print "COMPILATION STATISTICS: self compiles = ", self_compiles
            print "COMPILATION STATISTICS: other compiles = ", other_compiles
            print "COMPILATION STATISTICS: reduction in regexes = ", other_compiles - self_compiles
            # The underlying hardware accelerator nodes
            # are double counted since it is also in the 'supported'
            # list.
            print "COMPILATION STATISTICS: total states required = ", nodes_required
            print "COMPLIATION STATISTICS: total states saved = ", nodes_saved - nodes_required

        print "COMPILATION STATISTICS: unifications = ", compilation_statistics.unification_successes
        print "COMPILATION STATISTICS: Of those,  ", compilation_statistics.exact_same_compilations, " were equal"
        print "Tried to cross compile ", compilation_statistics.executed_comparisons, "regexes to each other"
        print "Successfully  compiled", compilation_statistics.algebras_compiled, "algebras"
        print "Avoided ", compilation_statistics.cutoff_comparisons, "detailed comparison attempts with heuristics"
        print "Of the algebras ", compilation_statistics.failed_algebra_computations, " failed (likely) due to incompleteness of current implementation"
    if options.print_unification_statistics:
        print "Single state completeness fails", compilation_statistics.ssu_complete_mapping_failed
        print "Single state correctness fails", compilation_statistics.ssu_correct_mapping_failed
        print "Single state structural modifications unification failed", compilation_statistics.ssu_additions_failed
        print "Single state disable edges failed", compilation_statistics.ssu_disable_edges_failed
        print "Single state disable symbols failed", compilation_statistics.ssu_disable_symbols_failed
        print "Single state successes", compilation_statistics.ssu_success

    if options.print_leq_failure_reasons:
        prefix = "LEQ Failure Reason:"
        print prefix, "Recursion Depth Exceeded", compilation_statistics.recursion_depth_exceeded
        print prefix, "Single Arm to Branch not Found", compilation_statistics.single_arm_to_branch_not_found
        print prefix, "Branch to Single Arm not Possible", compilation_statistics.branch_to_single_arm_not_possible
        print prefix, "Const to Non-Const", compilation_statistics.const_to_non_const
        print prefix, "Product to Product Failed", compilation_statistics.product_to_product_failed
        print prefix, "Sum Failed", compilation_statistics.sum_failure
        print prefix, "Branch to Branch Failure", compilation_statistics.branch_to_branch_failure
        print prefix, "Types differ", compilation_statistics.types_differ
        print prefix, "Accept to non-accept", compilation_statistics.accept_to_non_accept
        print prefix, "Const to Sum", compilation_statistics.const_to_sum_failed
        print prefix, "Sum to Sum", compilation_statistics.sum_to_const_failed

    if options.time:
        print "Time taken was", time.time() - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('anml_file_groups')
    parser.add_argument('--compression-stats', default=False, dest='compression_stats', action='store_true')

    options.add_to_parser(parser)

    args = parser.parse_args()

    process(args.anml_file_groups, args.name, args.compression_stats, options.create_from_args(args))
