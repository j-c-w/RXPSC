import automata as atma
import sjss
import algebra
import time
import generate_fst
import compilation_statistics
import simple_graph
import FST
import automata.FST.simulator.simulate_automata as simulator
try:
    from guppy import hpy
except:
    # Fails using pypy because the module
    # is not supported --- only used for memory
    # footprint debugging anyway
    pass


def compile(from_atma, to_atma, options):
    # Compare the shape of the autmata.  If the aren't roughly
    # the same shape, compiling won't work very well.
    depth_eqn_from = compute_depth_equation(from_atma, options)
    depth_eqn_to = compute_depth_equation(to_atma, options)

    if not depth_eqn_from:
        return None

    if not depth_eqn_to:
        return None

    if options.print_algebras:
        print "Compiling from ", depth_eqn_from
        print "Compiling to ", depth_eqn_to

    return compile_from_algebras(depth_eqn_from, from_atma, depth_eqn_to, to_atma, options)

def compare(eqn_from, eqn_to):
    return algebra.leq(eqn_from, eqn_to)


def compile_from_algebras(eqn_from, automata_from, eqn_to, automata_to, options):
    unification = algebra.leq_unify(eqn_from, eqn_to, options)

    if unification is None:
        return None
    else:
        # Use the unification to generate an FST if possible.
        result = generate_fst.generate(unification, automata_to, automata_from, options)
        if result:
            # Debugging issues with which accelerator is which.
            # for option in result.modifications.all_modifications():
            #     for edge in option.edges_after:
            #         assert edge in simple_graph.fromatma(automata_to).edges
            compilation_statistics.unification_successes += 1

            if options.verify:
                verify_fst(automata_to, automata_from, result, options)
        return result


# Ensure that the expected outputs are the same as the
# post-translation results.
def verify_fst(accelerator, automata, translator, options):
    # Don't verify automata that require structural modification,
    # a lot more work is needed before an 'actual' FST for
    # those automata is generated.
    if translator.has_structural_additions():
        return None

    # Don't currently support verification for other targets.
    assert options.target == 'single-state'

    with open(options.verify, 'r') as f:
        for input in f.readlines():
            input = input[:-1] # Delete the newline at the end of each input.
            original = simulator.accepts(simple_graph.fromatma(automata), input)
            translated_input = translator.apply(input)
            accelerated = simulator.accepts(simple_graph.fromatma(accelerator), translated_input)

            print "For input ", input
            print "Original answer is", original
            print "Accelerated answer is", accelerated

            assert original == accelerated


depth_equation_computation_index = 0
def compute_depth_equation(atma, options, dump_output=False):
    # Get the nodes, edges, starting and accepting states.
    # Convert them to straight up integers to make everything
    # else faster.
    nodes = atma.node_ids_list()
    edges = atma.edge_ids_list()
    start = atma.fake_root.id
    accepting_states = atma.reporting_states_list()
    if dump_output:
        print "Generating from :" + str(nodes)
        print "and: " + str(edges)
        print "with start: " + str(start)
        print "and end: " + str(accepting_states)
    if options.dump_nodes_and_edges:
        global depth_equation_computation_index

        with open(options.dump_nodes_and_edges, 'a') as f:
            f.write("nodes" + str(depth_equation_computation_index) + " = " + str(nodes) + "\n")
            f.write("edges" + str(depth_equation_computation_index) + " = " + str(edges) + "\n")
            f.write("""
def draw_{num}(fname):
    import automata.FST.debug as debug
    res = debug.to_graphviz(nodes{num}, edges{num})
    with open(fname, 'w') as f:
        f.write(res)
""".format(num=depth_equation_computation_index))

        depth_equation_computation_index += 1

    if len(nodes) > options.graph_size_threshold:
        print "Graph is too large for current implementation --- skipping"
        print "Graph is ", len(nodes), "and limit is ", options.graph_size_threshold, "nodes"
        print "Increase with --graph-size-limit <N>"
        compilation_statistics.graph_too_big += 1
        return None
 
    start_time = time.time()
    if options.skip_on_fail:
        try:
            alg = algebra.generate(nodes, edges, start, accepting_states, options)
        except Exception as e:
            compilation_statistics.failed_algebra_computations += 1
            print "Compilation of algebra failed!"
            print "Error was:"
            print e
            return None
    else:
        alg = algebra.generate(nodes, edges, start, accepting_states, options)
    end = time.time()
    if dump_output:
        print "Time taken is " + str(end - start_time)
        if alg.size() > 1000:
            print "Ommitting Algebra --- very long"
            print "(Size was) " + str(alg.size())
        else:
            print "Generated Algebra: " + str(alg)

    return alg
