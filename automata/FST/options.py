class Options(object):
    def __init__(self):
        self.tail_approximation = False
        self.assume_perfect_unification = False
        self.disabled_edges_approximation = False
        self.print_algebras = False
        self.profile = False

def create_from_args(args):
    opts = Options()
    opts.tail_approximation = args.tail_approximation
    opts.assume_perfect_unification = args.assume_perfect_unification
    opts.disabled_edges_approximation = args.disabled_edges_approximation
    opts.print_algebras = args.print_derived
    opts.profile = args.profile

    return opts

def add_to_parser(parser):
    parser.add_argument('--tail-approximation', default=False, dest='tail_approximation', action='store_true', help='Use the Tail Cutoff approximation in conversions.')
    parser.add_argument('--assume-perfect-unification', default=False, dest='assume_perfect_unification', action='store_true', help='Assume that the unifier always works')
    parser.add_argument('--disabled-edges-approximation', default=False, dest='disabled_edges_approximation', action='store_true', help='Do not require edges to be disabled.')
    parser.add_argument('--print-algebras', default=False, dest='print_derived',
            help='Print Derived Algebras', action='store_true')
    parser.add_argument('--profile', action='store_true', default=False, dest='profile', help='Run the profiler')

EmptyOptions = Options()

