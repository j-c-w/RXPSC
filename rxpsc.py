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

def extract_file_groups(file_groups, file_input=False):
    assert file_groups is not None
    if file_input:
        file_groups = [file_groups]
    else:
        file_groups = [os.path.join(file_groups, f) for f in os.listdir(file_groups) if os.path.isfile(os.path.join(file_groups, f))]

    return file_groups

def extract_automata_components(file_groups, opts):
    automata_components = []

    index = 0
    for file in file_groups:
        if not file.endswith('.anml'):
            continue
        if opts.print_file_info:
            print "Group Index: ", index, "comes from file", file
            index += 1

        automatas = atma.parse_anml_file(file)
        # Need to remove ors to avoid 'special elements' that
        # prevent us from getting the connected components.
        automatas.remove_ors()
        automata_components.append(automatas.get_connected_components_as_automatas())
    return automata_components


def compute_initial_state_count(automata_components):
    initial_state_count = 0
    for comp in automata_components:
        for cc in comp:
            initial_state_count += len(cc.nodes)
    return initial_state_count


# This function takes every automata in the add_from files,
# and computes whether it /individually/ could be added to
# a set of accelerators specified in the add_to set of files.
def run_addition_experiment(add_from, add_to, options):
    from_file_groups = extract_file_groups(add_from)
    to_file_groups = extract_file_groups(add_to)

    automata_components_from = extract_automata_components(from_file_groups, options)

    # Run each individual experiment:
    for i in range(len(automata_components_from)):
        for j in range(len(automata_components_from[i])):
            # This needs to be reloaded every time, since the underlying
            # functions change it (doh)
            automata_components_to = extract_automata_components(to_file_groups, options)

            add_to_check([[automata_components_from[i][j]]], automata_components_to, options)
    print "Finished experiment run!"


# This function takes automata to add, and automata to add_to,
# where we assume that add_to is already on an accelerator.
# We treat the 'add_from' group as we would any normal group ---
# that is that elements do not have to be run at the same time
# unless they come from the same folder/file, in which case
# the do have to be run at the same time.
def add_to(add_from, add_to, options):
    from_file_groups = extract_file_groups(add_from)
    to_file_groups = extract_file_groups(add_to)

    automata_components_from = extract_automata_components(from_file_groups, options)
    automata_components_to = extract_automata_components(to_file_groups, options)

    add_to_check(automata_components_from, automata_components_to, options)

def add_to_check(automata_components_from, automata_components_to, options):
    conversions = gc.compile_to_existing(automata_components_from, automata_components_to, options)

    if options.compression_stats or options.print_regex_injection_stats:
        # Check that all the conversions are individually not none.
        failed = False
        for conv in conversions:
            if conv is None:
                failed = True

        if failed:
            print "COMPRESSION RESULT: Failed to convert regexes to the existing automata"
        else:
            print "COMPRESSION RESULT: Converted regexes to the existing automata!"


def process(file_groups, file_input=False, options=None):
    start_time = time.time()
    file_groups = extract_file_groups(file_groups, file_input)
    print "Extracting ", len(file_groups), " groups"
    # Output is: HDL file with the important automata
    # selected + table programming.

    automata_components = extract_automata_components(file_groups, options)

    # State count of unoptimized regexes.
    initial_state_count = compute_initial_state_count(automata_components)

    assignments = gc.compile(automata_components, options)

    if assignments is None:
        print "Compilation finished --- Assignments was None, probably due to a command line flag not asking for the entire compilation"
        return

    if options.compression_stats:
        self_compiles = len(assignments)
        other_compiles = 0
        nodes_required = 0
        nodes_saved_by_sst = 0
        for assignment in assignments:
            other_compiles += len(assignment.translators)
            nodes_required += len(assignment.physical_automata.component.nodes)
            for supported_automata in assignment.supported_automata:
                nodes_saved_by_sst += len(supported_automata.component.nodes)
        # Don't print the results of the comparison if we are doing --compile-only (which skips the comparisons)
        if not options.compile_only:
            print "COMPILATION STATISTICS: self compiles = ", self_compiles
            print "COMPILATION STATISTICS: other compiles = ", other_compiles
            print "COMPILATION STATISTICS: reduction in regexes (from translation) = ", other_compiles - self_compiles
            # The underlying hardware accelerator nodes
            # are double counted since it is also in the 'supported'
            # list.
            print "COMPILATION STATISTICS: total states required = ", nodes_required
            print "COMPLIATION STATISTICS: total states saved = ", initial_state_count - nodes_required
            print "COMPILATION STATISTICS: total states saved by translation = ", nodes_saved_by_sst - nodes_required

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
        print "Single state structural additon completeness fail", compilation_statistics.ssu_addition_completeness_fail
        print "Single state disable addition symbols failed", compilation_statistics.ssu_addition_correctness_fail
        print "Single state addition failures due to homogeneity preservation", compilation_statistics.ssu_structural_addition_homogeneity_fail
        print "Single state comparisons avoided with heuristics", compilation_statistics.ssu_heuristic_fail
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
    subparsers = parser.add_subparsers()

    compress_parser = subparsers.add_parser('compress', help="Given a set of groups of regexes, try to statically compress the whole group as much as possible")
    compress_parser.set_defaults(mode='compress')
    compress_parser.add_argument('anml_file_groups')
    compress_parser.add_argument('-f', default=False, action='store_true', dest='file_input', help='Use a file as input rather than a folder.')


    addition_parser = subparsers.add_parser('addition', help="Given a set of implemented regular expressions, and a regular expression to add to the implemented set, try to add the regex using a translator")
    addition_parser.set_defaults(mode='addition')
    addition_parser.add_argument('addition_file', help="File with the automata to be added")
    addition_parser.add_argument('accelerator_file', help='File with the automata currently accelerated (in different groups)')

    addition_experiment_parser = subparsers.add_parser('addition-experiment', help='Given a regex file, try converting every regex in that file to the other groups /individually/. ')
    addition_experiment_parser.set_defaults(mode='addition-experiment')
    addition_experiment_parser.add_argument('addition_file', help='File with the automatas to be test-added')
    addition_experiment_parser.add_argument('accelerator_file', help='File with the automata currently accelerated (in different groups)')

    options.add_to_parser(compress_parser)
    options.add_to_parser(addition_parser)
    options.add_to_parser(addition_experiment_parser)

    args = parser.parse_args()
    opts = options.create_from_args(args)

    if args.mode == 'compression':
        compress(args.anml_file_groups, args.file_input, opts)
    elif args.mode == 'addition':
        add_to(args.addition_file, args.accelerator_file, opts)
    else:
        run_addition_experiment(args.addition_file, args.accelerator_file, opts)
