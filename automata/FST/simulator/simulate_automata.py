import automata.FST.simple_graph

# This is not a class designed to be complete, but rather to allow
# simple testing of the automata to make sure that the conversion
# machines behave correctly.

# Pretend this is a start at the beginning automata and see what happens.
def accepts(graph, string, trace=False):
    active_states = set([graph.start_state])
    neighbors = generate_neighbors_lookup(graph)
    
    for character in string:
        if trace:
            print "Read symbol ", character
            print "In states ", active_states
        next_states = set()
        for current_state in active_states:
            neighbor_states = neighbors[current_state]

            for next_state in neighbor_states:
                if character in graph.symbol_lookup[(current_state, next_state)]:
                    next_states.add(next_state)
        active_states = next_states

    for state in next_states:
        if state in graph.accepting_states:
            return True
    return False

def generate_neighbors_lookup(graph):
    lookup = {}
    for n in graph.nodes:
        lookup[n] = set()

    for s, e in graph.edges:
        lookup[s].add(e)
    return lookup
