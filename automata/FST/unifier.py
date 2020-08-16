# We need a value to indicate that an edges should not match
# anything.
NoMatchSymbol = None

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
            from_chars = symbol_lookup_1[self.from_edges[i]]
            if self.to_edges[i]:
                to_chars = symbol_lookup_2[self.to_edges[i]]
            else:
                to_chars = None

            for from_char in from_chars:
                if from_char in state_lookup:
                    if state_lookup[from_char] not in to_chars:
                        return None
                else:
                    # Not already mapped:
                    # COuld do this better, because this commits
                    # the mapping to be to_chars[0] earlier rather
                    # than later.
                    if to_chars:
                        state_lookup[from_char] = to_chars[0]
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
            return None
        else:
            for disable in disabled_states:
                state_lookup[disable] = non_matching

        # Translate everything else to itself.  This is an
        # relatively arbitrary decision I'm pretty sure.
        for i in range(0, 256):
            if i not in state_lookup:
                state_lookup[i] = i

        return state_lookup
