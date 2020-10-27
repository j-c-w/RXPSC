import os
import algebra
import unifier
import terms
import group_compiler

class Options(object):
    def __init__(self):
        self.tail_approximation = False
        self.disabled_edges_approximation = False
        self.print_algebras = False
        self.profile = False
        self.print_compile_time = False
        self.correct_mapping = True
        self.verify = None
        self.no_groups = False

        self.leq_iterations_file = None
        self.leq_calls_threshold = 100000
        self.no_leq_heuristics = True
        self.use_unification_heuristics = True
        self.use_inline_unification_heuristics = True

        self.group_size_distribution = None
        self.print_file_info = False
        self.print_unification_statistics = False
        self.dump_nodes_and_edges = None
        self.print_successful_conversions = False

        self.use_size_limits = True
        self.graph_size_threshold = 2000
        self.cross_compilation_threading = 0
        self.algebra_size_threshold = 2000
        self.max_branching_factor = 150

        self.size_difference_cutoff_factor = 5.0
        self.memory_debug = False
        self.time = False
        self.size_limit = None
        self.target = 'single-state'
        self.skip_on_fail = True
        self.line_profile = False

        self.use_structural_change = True

        self.comparison_cache = None
        self.dump_comparison_cache = None
        self.compile_ony = False
        self.print_leq_failure_reasons = False
        self.print_unification_failure_reasons = False

def create_from_args(args):
    algebra.LEQ_DEBUG = args.debug_leq
    algebra.ALG_DEBUG = args.debug_alg
    algebra.CACHE_ENABLED = not args.no_cache
    terms.TERMS_DEBUG = args.debug_terms
    unifier.DEBUG_UNIFICATION = args.debug_unification
    unifier.MAX_UNIFIERS = args.max_unifiers
    unifier.PRINT_UNIFICATION_FAILURE_REASONS = args.print_unification_failure_reasons
    group_compiler.DEBUG_COMPUTE_COMPAT_MATRIX = args.debug_compute_compat_matrix
    group_compiler.DEBUG_GENERATE_BASE = args.debug_generate_base
    group_compiler.MODIFICATION_LIMIT = args.modification_limit

    opts = Options()
    opts.tail_approximation = args.tail_approximation
    opts.disabled_edges_approximation = args.disabled_edges_approximation
    opts.correct_mapping = not args.allow_overapproximation
    opts.verify = args.verify
    opts.no_groups = args.no_groups
    opts.print_algebras = args.print_derived
    opts.profile = args.profile
    opts.leq_iterations_file = args.leq_iterations_file
    opts.leq_calls_threshold = args.leq_calls_threshold
    opts.use_structural_change = not args.no_structural_change
    opts.use_unification_heuristics = not args.no_unification_heuristics
    opts.use_inline_unification_heuristics = not args.no_inline_unification_heuristics

    opts.group_size_distribution = args.group_size_distribution
    opts.print_file_info = args.print_file_info
    opts.print_unification_statistics = args.print_unification_statistics
    opts.dump_nodes_and_edges = args.dump_nodes_and_edges
    opts.print_successful_conversions = args.print_successful_conversions
    opts.use_size_limits = not args.no_size_limits

    opts.graph_size_threshold = args.graph_size_threshold
    opts.cross_compilation_threading = args.cross_compilation_threading
    opts.size_difference_cutoff_factor = args.size_difference_cutoff
    opts.no_leq_heuristics = args.no_leq_heuristics
    opts.memory_debug = args.memory_debug
    opts.time = args.time
    opts.algebra_size_threshold = args.algebra_size_threshold
    opts.target = args.target
    opts.print_compile_time = args.print_compile_time
    opts.skip_on_fail = not args.no_skip_on_fail

    opts.comparison_cache = args.comparison_cache
    opts.dump_comparison_cache = args. dump_comparison_cache
    opts.compile_only = args.compile_only
    opts.print_leq_failure_reasons = args.print_leq_failure_reasons
    opts.line_profile = args.line_profile

    if opts.dump_nodes_and_edges:
        # Clear the file:
        if os.path.exists(opts.dump_nodes_and_edges):
            os.remove(opts.dump_nodes_and_edges)
    return opts

def add_to_parser(parser):
    parser.add_argument('--tail-approximation', default=False, dest='tail_approximation', action='store_true', help='Use the Tail Cutoff approximation in conversions.')
    parser.add_argument('--allow-overapproximation', default=False, dest='allow_overapproximation', action='store_true', help='Allow overapproximation of automata when compiling (completeness but not correctness)')
    parser.add_argument('--no-groups', default=False, dest='no_groups', action='store_true', help='Don\'t use the input groups --- assume every regex can compile to every other regex.')
    parser.add_argument('--disabled-edges-approximation', default=False, dest='disabled_edges_approximation', action='store_true', help='Do not require edges to be disabled.')
    parser.add_argument('--print-algebras', default=False, dest='print_derived',
            help='Print Derived Algebras', action='store_true')
    parser.add_argument('--profile', action='store_true', default=False, dest='profile', help='Run the profiler')
    parser.add_argument('--leq-iterations-file', default=None, dest='leq_iterations_file', help='Dump file for the number of iterations of the LEQ operation')
    parser.add_argument('--leq-calls-threshold', default=100000, dest='leq_calls_threshold', help='Number of recursive calls to make before giving up in the LEQ computation')
    parser.add_argument('--print-file-info', default=False, dest='print_file_info', help='Print the file information', action='store_true')
    parser.add_argument('--no-structural-change', default=False, dest='no_structural_change', help="Don't use structural change.", action='store_true')

    parser.add_argument('--group-size-distribution', default=None, dest='group_size_distribution', help='Dump the group size distribution to this file')
    parser.add_argument('--dump-nodes-and-edges', default=None, dest='dump_nodes_and_edges', help='Dump nodes and edges for each CC into a file')
    parser.add_argument('--graph-size-threshold', default=2000, dest='graph_size_threshold', help="Exclude graphs larger than this value (deal with python recursion limit)")
    parser.add_argument('--max-branching-factor', default=150, dest='max_branching_factor', type=int, help='Maximum branching factor for graphs before omitting.')
    parser.add_argument('--algebra-size-threshold', default=2000, dest='algebra_size_threshold', type=int)
    parser.add_argument('--cross-compilation-threading', default=0,dest='cross_compilation_threading', help='How many threads should be used for genreating comparisons. 0 disables the thread pool entirely.', type=int)
    parser.add_argument('--size-difference-cutoff-factor', default=5.0, dest='size_difference_cutoff', help='If algebra X is this many times larger than algebra Y, then assume that X </= Y, 0 disables', type=float)
    parser.add_argument('--no-leq-heuristics', default=False, dest='no_leq_heuristics', action='store_true', help='Use heuristics to skip some of the comparisons that seem likely to fail anyway')
    parser.add_argument('--modification-limit', default=10, dest='modification_limit', type=int, help='Limit the number of modifications to each accelerator')
    parser.add_argument('--no-unification-heuristics', default=False, dest='no_unification_heuristics', action='store_true', help='Use heuristcs to help skip unifications likely to fail anyway.')
    parser.add_argument('--no-inline-unification-heuristics', default=False, dest='no_inline_unification_heuristics', action='store_true', help='Use no inline heuristics')
    parser.add_argument('--compile-only', default=False, dest='compile_only', action='store_true', help="Don't  run any cross-compilation commands, just compile the algebras")
    parser.add_argument('--max-unifiers', default=20, dest='max_unifiers', type=int, help='Maximum number of unifiers to consider for any particular pair of equations')
    
    # Debug flags.
    parser.add_argument('--debug-leq', default=False, dest='debug_leq', action='store_true')
    parser.add_argument('--debug-terms', default=False, dest='debug_terms', action='store_true', help='Print all information about terms when printing')
    parser.add_argument('--no-skip-on-fail', default=False, dest='no_skip_on_fail', action='store_true', help="Normally, we skip the algebra if a tool fails.  With this flag, we do not skip")
    parser.add_argument('--debug-alg', default=False, dest='debug_alg', action='store_true')
    parser.add_argument('--debug-unification', default=False, dest='debug_unification', action='store_true')
    parser.add_argument('--debug-compute-compat-matrix', default=False, dest='debug_compute_compat_matrix', action='store_true')
    parser.add_argument('--debug-generate-base', default=False, dest='debug_generate_base', action='store_true')
    parser.add_argument('--memory-debug', default=False, dest='memory_debug', action='store_true')
    parser.add_argument('--print-compile-time', action='store_true', dest='print_compile_time', default=False)
    parser.add_argument('--print-unification-statistics', action='store_true', dest='print_unification_statistics', default=False)
    parser.add_argument('--print-leq-failure-reasons', default=False, dest='print_leq_failure_reasons', action='store_true', help='Print counters indicating why various equations failed the LEQ phase')
    parser.add_argument('--print-unification-failure-reasons', default=False, dest='print_unification_failure_reasons', action='store_true', help='Print reasons that unifiers fail within the single state unification method.')
    parser.add_argument('--print-successful-conversions', default=False, dest='print_successful_conversions', action='store_true', help='Print successful conversions between algebras.')
    parser.add_argument('--line-profile', default=False, dest='line_profile', action='store_true', help='Profile the LEQ structure.')
    parser.add_argument('--time', default=False, dest='time', action='store_true', help='Print the compilation time')
    parser.add_argument('--verify', default=False, dest='verify', help='Do a verification --- the inputs are line-by-line in the file that is provided as argument.')
    parser.add_argument('--no-cache', default=False, dest='no_cache', action='store_true', help='Disable the computation caches --- this makes things /much/ shlower.')
    parser.add_argument('--no-size-limits', default=False, dest='no_size_limits', help="Disable all size limits on input graphs (Not recommended!)")

    # Target flags
    parser.add_argument('--target', choices=['single-state', 'symbol-only-reconfiguration', 'perfect-unification'], default='single-state')

    # Intermediate output flags
    parser.add_argument('--dump-comparison-cache', default=None, dest='dump_comparison_cache', help='Dump a conversion map in a file --- this can be used to speedup subsequent runs by caching comparison results.')
    parser.add_argument('--comparison-cache', default=None, dest='comparison_cache', help='Takes as input a comparison cache file (as generated by the --dump-comparison-cache flag).  This makes the unifier generation /much/ faster when repeatedly running over the same file')

EmptyOptions = Options()
