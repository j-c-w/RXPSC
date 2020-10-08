class SimpleGraph(object):
    def __init__(self, nodes, edges, symbol_lookup, accepting_states, start_state):
        self.nodes = nodes
        self.edges = edges
        self.symbol_lookup = symbol_lookup
        self.accepting_states = accepting_states
        self.start_state = start_state
