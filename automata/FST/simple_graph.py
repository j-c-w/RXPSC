import sjss


# Use negative IDs to not conflict with Grapefruit IDs, whcih are
# psotivie, and sometimes passed over.
simple_graph_id_counter = -1
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

        global simple_graph_id_counter
        self.id = simple_graph_id_counter
        simple_graph_id_counter -= 1

        for edge in edges:
            assert edge in symbol_lookup

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
        try:
            loop_groups = sjss.compute_loop_groups(self.nodes, self.edges, self.start_state)
        except:
            # This is a terrible hack.  It leads to some nodes not
            #being marked as end for some graphs that they should be.
            # I don't believe it significantly impacts
            # any results --- we just need to pick an arbitrary
            # node from the final loop anyway.
            # This just omitts picking that node.
            loop_groups = {}
            for n in self.nodes:
                loop_groups[n] = []

        for n in self.nodes:
            if len(neighbors[n]) == 0:
                end_states.add(n)
            elif len(neighbors[n]) == len(loop_groups[n]):
                # If this is the first node for a bunch of loops
                # that also go nowhere else, then this is effectively
                # the end state of the graph (i.e., it's where
                # any attached graphs should hook in).
                end_states.add(n)
        return end_states

    def __hash__(self):
        # This is not a truly unique hash, but is designed to
        # be unique-enough.. Think we could do better if required.
        node_count = len(self.nodes)
        node_sum = sum(self.nodes)
        edges_count = len(self.edges)
        symbs = 0
        for edge in self.edges:
            symbs += edge[0] * sum(self.symbol_lookup[edge])
        return symbs * node_sum + edges_count * node_count

def fromatma(atma):
    g = sjss.automata_to_nodes_and_edges(atma)
    g.id = atma.id
    return g
