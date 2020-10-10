import sjss

class SimpleGraph(object):
    def __init__(self, nodes, edges, symbol_lookup, accepting_states, start_state):
        assert type(symbol_lookup) == type({})
        assert type(start_state) != type([])
        self.nodes = nodes
        self.edges = edges
        self.symbol_lookup = symbol_lookup
        self.accepting_states = accepting_states
        self.start_state = start_state

    def clone(self):
        return SimpleGraph(
                self.nodes[:],
                self.edges[:],
                dict(self.symbol_lookup),
                self.accepting_states[:],
                self.start_state)

    def __str__(self):
        return 'nodes: ' + str(self.nodes) + '\nedges: ' + str(self.edges) + \
                '\nsymbols: ' + str(self.symbol_lookup) + '\naccepting: ' + str(self.accepting_states) + \
                '\nstart: ' + str(self.start_state)

def fromatma(atma):
    return sjss.automata_to_nodes_and_edges(atma)
