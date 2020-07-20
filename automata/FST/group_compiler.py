import single_compiler as sc

def compile(automata_components):
    # Takes a list of list of CCs, and returns a master
    # automata that can run each group + a set of FSTs
    # to load into each table.

    # For now, compile just one to one other as a demo.
    sc.compile(automata_components[0][0], automata_components[1][0])

    return None, None
