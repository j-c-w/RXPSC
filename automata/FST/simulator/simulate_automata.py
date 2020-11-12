import automata.FST.simple_graph

# This is not a class designed to be complete, but rather to allow
# simple testing of the automata to make sure that the conversion
# machines behave correctly.

def accepts(graph, string, trace=False):
    active_states = set([graph.start_state])
    neighbors = graph.neighbors_lookup()
    end_states = graph.end_states(neighbors)
    if str(type(list(graph.symbol_lookup[list(graph.edges)[0]])[0])) == "<type 'int'>" and str(type(string)) == "<type 'str'>":
        # Translate the char stream to an int stream:
        string = [ord(x) for x in string]

    for character in string:
        if trace:
            print "Read symbol ", character
            print "In states ", active_states
        next_states = set([graph.start_state])
        for current_state in active_states:
            # See note about end states hack below.
            if current_state in end_states or current_state in graph.accepting_states:
                return True

            neighbor_states = neighbors[current_state]

            for next_state in neighbor_states:
                assert type(character) == type(list(graph.symbol_lookup[(current_state, next_state)])[0])

                if character in graph.symbol_lookup[(current_state, next_state)]:
                    next_states.add(next_state)
        active_states = next_states

    for state in next_states:
        if state in graph.accepting_states:
            return True
        # This is a terrible dirty hack that is done to 
        # make handling prefix extractions easier.
        # What really needs to happen is prefixes need to
        # have accept appended to them, the problem with
        # that is that it depends what kind of implementation
        # you want (i.e. do you want to connect the prefixes
        # directly to the postfixes, or deal with it through
        # combined reporting)
        if state in end_states:
            return True
    return False
