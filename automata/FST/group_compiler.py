import single_compiler as sc

def compile(automata_components):
    # Takes a list of list of CCs, and returns a master
    # automata that can run each group + a set of FSTs
    # to load into each table.

    # For now, compile just one to one other as a demo.
    for i in range(len(automata_components[0])):
        sc.compile(automata_components[0][i], automata_components[1][i])

    return None, None


# Estimate the amount of cross-compatability within
# a suite of automata.
def compile_statistics(connected_components):
    algebras = [None] * len(connected_components)

    print "Starting Compilation"
    for i in range(len(connected_components)):
        algebras[i] = sc.compute_depth_equation(connected_components[i], dump_output=True)

    # Now, check for cross compatability:
    cross_compilations = 0
    for j in range(len(algebras)):
        for i in range(len(algebras)):
            if i != j:
                result = sc.compare(algebras[i], algebras[j])
                if result:
                    print "Compiled ", algebras[i], " to ", algebras[j]
                    cross_compilations += 1

    print "Total cross compilations is ", cross_compilations
