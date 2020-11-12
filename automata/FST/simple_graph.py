import sjss

class SimpleGraph(object):
    def __init__(self, nodes, edges, symbol_lookup, accepting_states, start_state):
        assert type(symbol_lookup) == type({})
        assert type(start_state) != type([])
        assert type(nodes) == type(set())
        assert type(edges) == type(set())
        assert type(accepting_states) == type(set())

        self.nodes = nodes
        self.edges = edges
        self.symbol_lookup = symbol_lookup
        self.accepting_states = accepting_states
        self.start_state = start_state

    def clone(self):
        return SimpleGraph(
                set(self.nodes),
                set(self.edges),
                dict(self.symbol_lookup),
                set(self.accepting_states),
                self.start_state)

    def __str__(self):
        return 'nodes: ' + str(self.nodes) + '\nedges: ' + str(self.edges) + \
                '\nsymbols: ' + str(self.symbol_lookup) + '\naccepting: ' + str(self.accepting_states) + \
                '\nstart: ' + str(self.start_state)

    def to_python_string(self):
        return "{" + "'nodes': " + str(self.nodes) + \
                ",\n 'edges':" + str(self.edges) + \
                ",\n 'symbol_lookup':" + str(self.symbol_lookup) + \
                ",\n 'end_states':" + str(self.end_states()) + \
                ",\n 'accepting_states':" + str(self.accepting_states) + \
                ",\n 'start_state':" + str(self.start_state) + \
                "}"

    def neighbors_lookup(self):
        lookup = {}
        for n in self.nodes:
            lookup[n] = set()

        for s, e in self.edges:
            lookup[s].add(e)
        return lookup

    def end_states(self, neighbors=None):
        if neighbors is None:
            neighbors = self.neighbors_lookup()
        end_states = set()

        for n in self.nodes:
            if len(neighbors[n]) == 0:
                end_states.add(n)
        return end_states

def fromatma(atma):
    return sjss.automata_to_nodes_and_edges(atma)
