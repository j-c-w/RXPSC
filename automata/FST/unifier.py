import FST
import compilation_statistics
# We need a value to indicate that an edges should not match
# anything.
NoMatchSymbol = None
DEBUG_UNIFICATION = False
PRINT_UNIFICATION_FAILURE_REASONS = True

class Unifier(object):
    def __init__(self, algebra_from=None, algebra_to=None, cost=0):
        self.from_edges = []
        self.to_edges = []
        self.disabled_edges = []
        self.algebra_from = algebra_from
        self.algebra_to = algebra_to
        self.cost = cost
        self.ununified_terms = []

    def add_edges(self, from_edges, to_edges):
        assert len(from_edges) == len(to_edges)

        self.from_edges += from_edges
        self.to_edges += to_edges

    def set_ununified_terms(self, terms):
        assert self.ununified_terms == []
        self.ununified_terms.appen(terms)

    # Add some edges to make sure are never active.
    def add_disabled_edges(self, edges):
        self.disabled_edges += edges

    def add_cost(self, cost):
        self.cost += cost

    def get_disabled_edges(self):
        return self.disabled_edges

    def unify_with(self, other):
        self.from_edges += other.from_edges
        self.to_edges += other.to_edges
        self.cost += other.cost
        self.ununified_terms.append(other.ununified_terms)

    # There may be some issues surrounding the naming convention
    # of what is 'from' and what is 'to' in this function (and
    # elsewhere tbh). Anyway, any issues with that should be aparent
    # even if a pain.
    def unify_single_state(self, symbol_lookup_1, symbol_lookup_2):
        # This unifies into a single state.
        state_lookup = {}
        matching_symbol = {}

        if DEBUG_UNIFICATION or PRINT_UNIFICATION_FAILURE_REASONS:
            print "Starting new unification between "
            print self.algebra_from
            print self.algebra_to
            if self.algebra_from.equals(self.algebra_to, symbol_lookup_2, symbol_lookup_1):
                print "Algebras are actually exactly the same..."
                compilation_statistics.exact_same_compilations += 1

        for i in range(len(self.from_edges)):
            if DEBUG_UNIFICATION:
                print "Trying to unify edges: "
                print self.from_edges[i]
                print self.to_edges[i]
                print symbol_lookup_1
                print symbol_lookup_2
                print "Symbol sets are:"
                print symbol_lookup_1[self.from_edges[i]]
                print symbol_lookup_2[self.to_edges[i]]
            from_chars = symbol_lookup_1[self.from_edges[i]]
            if self.to_edges[i]:
                to_chars = symbol_lookup_2[self.to_edges[i]]
            else:
                if DEBUG_UNIFICATION or PRINT_UNIFICATION_FAILURE_REASONS:
                    print "Found an edge with disabling required"
                to_chars = None

            for from_char in from_chars:
                if from_char in state_lookup:
                    overlap_set = []
                    for character in state_lookup[from_char]:
                        if character in to_chars:
                            overlap_set.append(character)

                    if len(overlap_set) == 0:
                        if DEBUG_UNIFICATION or PRINT_UNIFICATION_FAILURE_REASONS:
                            print "Unification failed due to double-mapped state"
                            print "(" + str(from_char) + ") already mapped to " + str(state_lookup[from_char]) + " when something in " + str(to_chars) + " is required"
                        compilation_statistics.single_state_unification_double_map_fails += 1
                        return None
                    else:
                        # The overlap set is non-zero, but may
                        # still be smaller than it was before.
                        state_lookup[from_char] = overlap_set
                else: # Fromchar not in state lookup.
                    if to_chars:
                        state_lookup[from_char] = to_chars
                    else:
                        assert False # I'm pretty sure that this
                        # should never happen...

            if to_chars:
                for char in to_chars:
                    matching_symbol[char] = True

           
        # Now, go through and map any None values, which mean
        # that symbols should be mapped to things that don't match
        # any edges.
        non_matching = None
        for i in range(0, 256):
            if i not in matching_symbol:
                non_matching = i

        disabled_edges = self.get_disabled_edges()
        if non_matching is None and len(disabled_edges) != 0 and not options.disabled_edges_approximation:
            if DEBUG_UNIFICATION:
                print "Unification failed due to required edge disabling that cannot be achieved"
            compilation_statistics.single_state_unification_non_matching ++ 1
            return None
        elif non_matching is not None:
            for disable in disabled_edges:
                state_lookup[disable] = non_matching

        # Translate everything else to itself.  This is an
        # relatively arbitrary decision I'm pretty sure.
        for i in range(0, 256):
            if i in state_lookup:
                # Pick the first available option arbitrarily.
                state_lookup[i] = state_lookup[i][0]
            else:
                state_lookup[i] = i

        if DEBUG_UNIFICATION or PRINT_UNIFICATION_FAILURE_REASONS:
            print "Returning a real result"

        compilation_statistics.single_state_unification_success += 1
        return FST.SingleStateTranslator(state_lookup)
