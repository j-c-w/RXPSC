class Options(object):
    def __init__(self):
        self.tail_approximation = False
        self.assume_perfect_unification = False
        self.disabled_edges_approximation = False
        self.print_algebras = False
        self.profile = False

        self.leq_iterations_file = None
        self.leq_calls_threshold = 100000

def create_from_args(args):
    opts = Options()
    opts.tail_approximation = args.tail_approximation
    opts.assume_perfect_unification = args.assume_perfect_unification
    opts.disabled_edges_approximation = args.disabled_edges_approximation
    opts.print_algebras = args.print_derived
    opts.profile = args.profile
    opts.leq_iterations_file = args.leq_iterations_file
    opts.leq_calls_threshold = args.leq_calls_threshold

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

EmptyOptions = Options()

