import os
import algebra
import unifier
import group_compiler

class Options(object):
    def __init__(self):
        self.tail_approximation = False
        self.assume_perfect_unification = False
        self.disabled_edges_approximation = False
        self.print_algebras = False
        self.profile = False

        self.leq_iterations_file = None
        self.leq_calls_threshold = 100000
        self.use_leq_heuristics = True

        self.group_size_distribution = None
        self.print_file_info = False
        self.dump_nodes_and_edges = None

        self.graph_size_threshold = 2000
        self.cross_compilation_threading = 0

        self.size_difference_cutoff_factor = 2.0

def create_from_args(args):
    algebra.LEQ_DEBUG = args.debug_leq
    unifier.DEBUG_UNIFICATION = args.debug_unification
    group_compiler.DEBUG_COMPUTE_COMPAT_MATRIX = args.debug_compute_compat_matrix

    opts = Options()
    opts.tail_approximation = args.tail_approximation
    opts.assume_perfect_unification = args.assume_perfect_unification
    opts.disabled_edges_approximation = args.disabled_edges_approximation
    opts.print_algebras = args.print_derived
    opts.profile = args.profile
    opts.leq_iterations_file = args.leq_iterations_file
    opts.leq_calls_threshold = args.leq_calls_threshold

    opts.group_size_distribution = args.group_size_distribution
    opts.print_file_info = args.print_file_info
    opts.dump_nodes_and_edges = args.dump_nodes_and_edges

    opts.graph_size_threshold = args.graph_size_threshold
    opts.cross_compilation_threading = args.cross_compilation_threading
    opts.size_difference_cutoff_factor = args.size_difference_cutoff
    opts.use_leq_heuristics = args.use_leq_heuristics

    if opts.dump_nodes_and_edges:
        # Clear the file:
        if os.path.exists(opts.dump_nodes_and_edges):
            os.remove(opts.dump_nodes_and_edges)
    return opts

def add_to_parser(parser):
    parser.add_argument('--tail-approximation', default=False, dest='tail_approximation', action='store_true', help='Use the Tail Cutoff approximation in conversions.')
    parser.add_argument('--assume-perfect-unification', default=False, dest='assume_perfect_unification', action='store_true', help='Assume that the unifier always works')
    parser.add_argument('--disabled-edges-approximation', default=False, dest='disabled_edges_approximation', action='store_true', help='Do not require edges to be disabled.')
    parser.add_argument('--print-algebras', default=False, dest='print_derived',
            help='Print Derived Algebras', action='store_true')
    parser.add_argument('--profile', action='store_true', default=False, dest='profile', help='Run the profiler')
    parser.add_argument('--leq-iterations-file', default=None, dest='leq_iterations_file', help='Dump file for the number of iterations of the LEQ operation')
    parser.add_argument('--leq-calls-threshold', default=100000, dest='leq_calls_threshold', help='Number of recursive calls to make before giving up in the LEQ computation')
    parser.add_argument('--print-file-info', default=False, dest='print_file_info', help='Print the file information', action='store_true')

    parser.add_argument('--group-size-distribution', default=None, dest='group_size_distribution', help='Dump the group size distribution to this file')
    parser.add_argument('--dump-nodes-and-edges', default=None, dest='dump_nodes_and_edges', help='Dump nodes and edges for each CC into a file')
    parser.add_argument('--graph-size-threshold', default=2000, dest='graph_size_threshold', help="Exclude graphs larger than this value (deal with python recursion limit)")
    parser.add_argument('--cross-compilation-threading', default=0,dest='cross_compilation_threading', help='How many threads should be used for genreating comparisons. 0 disables the thread pool entirely.', type=int)
    parser.add_argument('--size-difference-cutoff-factor', default=2.0, dest='size_difference_cutoff', help='If algebra X is this many times larger than algebra Y, then assume that X </= Y, 0 disables', type=float)
    parser.add_argument('--use-leq-heuristics', default=True, dest='use_leq_heuristics', help='Use heuristics to skip some of the comparisons that seem likely to fail anyway')
    
    # Debug flags.
    parser.add_argument('--debug-leq', default=False, dest='debug_leq', action='store_true')
    parser.add_argument('--debug-unification', default=False, dest='debug_unification', action='store_true')
    parser.add_argument('--debug-compute-compat-matrix', default=False, dest='debug_compute_compat_matrix', action='store_true')

EmptyOptions = Options()

