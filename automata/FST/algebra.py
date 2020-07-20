# Get the algebtra terms
from terms import *
import sjss

ALG_DEBUG = True


def generate(nodes, edges, start, accept_states):
    result = generate_internal(nodes, edges, start, accept_states)
    print(result)
    return result.normalize()


def generate_internal(nodes, edges, start, accept_states):
    output_lookup = sjss.generate_output_lookup(nodes, edges)

    # Get the branches too.
    branches = sjss.compute_branches(nodes, edges, start)
    if ALG_DEBUG:
        print("Branches:")
        print(branches)
    end_states = sjss.compute_end_states(nodes, edges)

    # Now, go through each branch, and figure out
    # what should happen after it.  This is a DFS.
    nodes_list = [start]
    branch_lookup = generate_branches_by_start_lookup(branches)
    algebra = None
    algebra_stack = [[None]]
    algebra_stack_counts = [1]

    while len(nodes_list) != 0:
        node = nodes_list[0]
        nodes_list = nodes_list[1:]
        algebra = algebra_stack[-1][algebra_stack_counts[-1] - 1]
        if ALG_DEBUG:
            print("For node " + str(node))
            print("Algebra stack is: ")
            print([[str(i) for i in x] for x in algebra_stack])
            print("Algebra stack indexes are:")
            print(algebra_stack_counts)
            print("Current Algebra is:")
            print(algebra)

        branches_starting_here = branch_lookup[node]

        ends = []
        for branch in branches_starting_here:
            # Add the nodes to consider to the nodes list.
            end = branch[-1]

            if end == node:
                # This is a loop, do not push
                pass
            else:
                ends.append(end)
        nodes_list = ends[::-1] + nodes_list

        # If this is an accept state, add that.
        # I don't think I need to do this here.
        # if len(branches_starting_here) == 0:
        #     algebra = Sum(algebra, Accept()) if algebra else Accept()

        # If this is a loop, need to deal with that: TODO

        # If this branch is a dead-end, add that
        # to the result equation:
        if len(branches_starting_here) == 0:
            algebra = Sum(algebra, End()) if algebra else End()

            # Save the algebra onto the stack.
            algebra_stack[-1][algebra_stack_counts[-1] - 1] = algebra

            # This loop 'builds' the algebra stack by adding things
            # together and creating branches where there are multiple
            # options.
            while algebra_stack_counts[-1] == 1:
                if ALG_DEBUG:
                    print "Compressing at depth " + str(len(algebra_stack_counts))
                # We need to build the element before deleteing it.
                # But we know this just contains a 1.
                del algebra_stack_counts[-1] # delete the last algebra stack.

                # The tail of the algebra stack becomes a branch.
                algebra = algebra_stack[-1]

                if len(algebra) == 1:
                    algebra = algebra[0]
                else:
                    algebra = Branch(algebra)
                del algebra_stack[-1]

                if len(algebra_stack_counts) == 0:
                    # we are done :)
                    return algebra

                # We have 'completed' a branch here.
                # Combine the algebra we computed with the last algebra
                # on the stack.
                new_head_of_stack = algebra_stack[-1][algebra_stack_counts[-1] - 1]
                if new_head_of_stack is None:
                    # This case is to deal with the shitty instantiation
                    # for the first iteration that leaves a 'None'
                    # at the start of the algebra stack.
                    new_head_of_stack = algebra
                else:
                    new_head_of_stack = Sum(new_head_of_stack, algebra)
                algebra_stack[-1][algebra_stack_counts[-1] - 1] = new_head_of_stack

            # Decrement the stack counts.
            algebra_stack_counts[-1] = algebra_stack_counts[-1] - 1

        # Dunno how often this case comes up, since things should
        # be compressed.  Might happen for the  start node anyway
        if len(branches_starting_here) == 1:
            algebra = Sum(algebra, linear_algebra_for(branches_starting_here[0], accept_states)) if algebra else linear_algebra_for(branches_starting_here[0], accept_states)
            # Persist that algebra into the stack.
            algebra_stack[-1][algebra_stack_counts[-1] - 1] = algebra

        # If this is a branch, go deeper in the algebra stack.
        if len(branches_starting_here) > 1:
            branches_count = len(branches_starting_here)

            # This keeps track of how many branches there
            # are to explore down this path.
            # Get the algebra for each branch
            branch_algebras = []
            added_count = 0
            for branch in branches_starting_here:
                if branch[-1] == node:
                    # We don't want to handle loops, those
                    # aren't considered as things to walk down
                    # I think this is probably a bug and will have
                    # to be removed later, but the concept
                    # is that this has to mirror the modifications
                    # to the next node stack.
                    continue
                alg = linear_algebra_for(branch, accept_states)
                branch_algebras.append(alg)
                added_count += 1

            algebra_stack_counts.append(added_count)
            algebra_stack.append(branch_algebras)


def generate_branches_by_start_lookup(branches):
    # Generate a lookup table that indexes the list
    # of branches by the start node.
    lookup = {}

    for branch in branches:
        for node in branch:
            lookup[node] = []
    for branch in branches:
        lookup[branch[0]].append(branch)

    return lookup


# Given a set of states where we know that there
# are no branches, compute some equation
# for them, e.g. 3 + a + 2 + a + e
def linear_algebra_for(branch, accept_states):
    algebra = []
    edges = []
    single_state_count = 0
    last_state = None

    if ALG_DEBUG:
        print "Computing algebra for branch: " + str(branch)

    last_state = branch[0]
    for state in branch[1:]:
        single_state_count = single_state_count + 1
        edges.append((last_state, state))
        last_state = state

        if state in accept_states:
            const = Const(single_state_count, edges)
            algebra.append(const)
            single_state_count = 0
            edges = []
            algebra.append(Accept())

    if single_state_count > 0:
        algebra.append(Const(single_state_count, edges))

    if single_state_count == 0:
        algebra.append(Const(0, []))

    if len(algebra) == 1:
        sum_elt = algebra[0]
    else:
        # Join them together with Sum operations.
        algebra = algebra[::-1]
        sum_elt = Sum(algebra[1], algebra[0])
        for item in algebra[2:]:
            sum_elt = Sum(item, sum_elt)

    if ALG_DEBUG:
        print "Computed Algebra is " + str(sum_elt)

    return sum_elt
