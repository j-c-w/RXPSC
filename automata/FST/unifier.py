# We need a value to indicate that an edges should not match
# anything.
NoMatchSymbol = None
DEBUG_UNIFICATION = False

class Unifier(object):
    def __init__(self):
        self.from_edges = []
        self.to_edges = []

    def add_edges(self, from_edges, to_edges):
        assert len(from_edges) == len(to_edges)

        self.from_edges += from_edges
        self.to_edges += to_edges

    def unify_with(self, other):
        self.from_edges += other.from_edges
        self.to_edges += other.to_edges

    def unify_single_state(self, symbol_lookup_1, symbol_lookup_2):
        # This unifies into a single state.
        state_lookup = {}
        matching_symbol = {}
        # List of characters that need to not trigger any edges
        # in the accelerated state machine.
        disabled_states = []

        for i in range(len(self.from_edges)):
            if DEBUG_UNIFICATION:
                print "Trying to unify edges: "
                print self.from_edges[i]
                print self.to_edges[i]
                print "Symbol sets are:"
                print symbol_lookup_1[self.from_edges[i]]
                print symbol_lookup_2[self.to_edges[i]]
            from_chars = symbol_lookup_1[self.from_edges[i]]
            if self.to_edges[i]:
                to_chars = symbol_lookup_2[self.to_edges[i]]
            else:
                if DEBUG_UNIFICATION:
                    print "Found an edge with disabling required"
                to_chars = None

            for from_char in from_chars:
                if from_char in state_lookup:
                    overlap_set = []
                    for character in state_lookup[from_char]:
                        if character in to_chars:
                            overlap_set.append(character)

                    if len(overlap_set) == 0:
                        if DEBUG_UNIFICATION:
                            print "Unification failed due to double-mapped state"
                            print "(" + str(from_char) + ") already mapped to " + str(state_lookup[from_char]) + " when something in " + str(to_chars) + " is required"
                        return None
                    else:
                        # The overlap set is non-zero, but may
                        # still be smaller than it was before.
                        state_lookup[from_char] = overlap_set
                else: # Fromchar not in state lookup.
                    if to_chars:
                        state_lookup[from_char] = to_chars
                    else:
                        state_lookup[from_char] = None
                        disabled_states.append(from_char)

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

        if non_matching is None and len(disabled_states) != 0:
            if DEBUG_UNIFICATION:
                print "Unification failed due to required edge disabling that cannot be achieved"
            return None
        else:
            for disable in disabled_states:
                state_lookup[disable] = non_matching

        # Translate everything else to itself.  This is an
        # relatively arbitrary decision I'm pretty sure.
        for i in range(0, 256):
            if i in state_lookup:
                # Pick the first available option arbitrarily.
                state_lookup[i] = state_lookup[i][0]
            else:
                state_lookup[i] = i

        return state_lookup
