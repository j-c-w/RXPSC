import automata as atma
import sjss

def compile(from_atma, to_atma):
    # Compare the shape of the autmata.  If the aren't roughly
    # the same shape, compiling won't work very well.
    depth_eqn_from = compute_depth_equation(from_atma)
    depth_eqn_to = compute_depth_equation(to_atma)

    # TODO -- Do something with those depth equations.


def compute_depth_equation(atma):
    # Construct a lookup table from the node that inidicates
    # the current depth.

    # Convert the automata so we can apply the path algebra.
    depth_eqn_automata, unification_constraints = atma.get_SJSS()

    current_node = atma.root()
    nodes_to_explore = []
    algebra_terms = []

    while len(nodes_to_explore) > 0:
        # First, compute how many edges are in the current
        # node, and if the current node is accepting.
        # TODO

        # Next, check for loops from this node.
        # TODO

        # If there are loops, compute the equations
        # for those loops.
        # TODO

        # If there are branches, compute the equations
        # for those branches
        # TODO

        # Progress onto the next node linearlly if there 
        # is one.
        # TODO
