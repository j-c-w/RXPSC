import automata as atma
import sjss
import algebra
import time
import generate_fst

def compile(from_atma, to_atma):
    # Compare the shape of the autmata.  If the aren't roughly
    # the same shape, compiling won't work very well.
    depth_eqn_from = compute_depth_equation(from_atma)
    depth_eqn_to = compute_depth_equation(to_atma)

    print "From Eqn:"
    print str(depth_eqn_from)
    print "To Eqn:"
    print str(depth_eqn_to)

    return compile_from_algebras(depth_eqn_from, from_atma, depth_eqn_to, to_atma)

def compare(eqn_from, eqn_to):
    return algebra.leq(eqn_from, eqn_to)


def compile_from_algebras(eqn_from, automata_from, eqn_to, automata_to):
    unification = algebra.leq_unify(eqn_from, eqn_to)

    if unification is None:
        return None
    else:
        # Use the unification to generate an FST if possible.
        return generate_fst.generate(unification, automata_from, automata_to)


def compute_depth_equation(atma, dump_output=False):
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
 
    start_time = time.time()
    alg =  algebra.generate(nodes, edges, start, accepting_states)
    end = time.time()
    if dump_output:
        print "Time taken is " + str(end - start_time)
        if alg.size() > 1000:
            print "Ommitting Algebra --- very long"
            print "(Size was) " + str(alg.size())
        else:
            print "Generated Algebra: " + str(alg)

    return alg
