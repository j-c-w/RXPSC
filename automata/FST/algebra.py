# Get the algebtra terms
from terms import *
import sjss

ALG_DEBUG = True

def generate(nodes, edges, start, accept_states):
    branches = sjss.compute_branches(nodes, edges, start)
    loops = sjss.compute_loops(nodes, edges, start)
    end_states = sjss.compute_end_states(nodes, edges)

    result = generate_internal(nodes, edges, start, accept_states, end_states, branches, loops)
    return result.normalize()


def generate_internal(nodes, edges, start, accept_states, end_states, branches_analysis, loops_analysis):
    output_lookup = sjss.generate_output_lookup(nodes, edges)
    if ALG_DEBUG:
        print("Branches:")
        print(branches_analysis)
        print("Loops:")
        print(loops_analysis)
        print("End states:")
        print(end_states)

    # Now, go through each branch, and figure out
    # what should happen after it.  This is a DFS.
    nodes_list = [start]
    branch_lookup = generate_branches_by_start_lookup(branches_analysis)
    algebra = None
    algebra_stack = [[None]]
    algebra_stack_counts = [1]

    while len(nodes_list) != 0:
        node = nodes_list[0]
        nodes_list = nodes_list[1:]
        if ALG_DEBUG:
            print("For node " + str(node))
            print("Algebra stack is: ")
            print([[str(i) for i in x] for x in algebra_stack])
            print("Algebra stack indexes are:")
            print(algebra_stack_counts)
            print("Current Algebra is:")
            algebra = algebra_stack[-1][algebra_stack_counts[-1] - 1]
            print(algebra)

        branches_starting_here = branch_lookup[node]

        ends = []
        loops_from_this_node = []
        non_loops_from_this_node = []
        for branch in branches_starting_here:
            # Add the nodes to consider to the nodes list.
            end = branch[-1]

            if isloopstart(branch, loops_analysis):
                # This is a loop, do not push
                loops_from_this_node.append(branch)
            else:
                ends.append(end)
                non_loops_from_this_node.append(branch)
        nodes_list = ends[::-1] + nodes_list

        # If this node has loops, then we need to deal with those.
        if len(loops_from_this_node) > 0:
            # This is a loop.  We only handle loops that
            # have start and end equivalence, transformations
            # to this type of automata are handled in the sjss
            # pass.
            # We need to compute the algebra of the loop, then
            # put it in a product.

            # Get the loop that the branch is part of.
            if ALG_DEBUG:
                print "Recursing"

            sub_nodes, sub_edges, removed_edges, sub_branches_analysis, sub_loops_analysis = sjss.compute_loop_subregion(nodes, edges, node, loops_analysis, branches_analysis)
            # This is done by recursively considering a much smaller
            # graph with the loop in it.

            # Only recurse with the branches and loops we
            # want to recurse with.  These are
            # branches and loops that have nodes only within
            # the set of branches and loops being used
            # within the loop.

            # Put those on a stack and recurse, we could iterate,
            # but that seems messier, and we kinda figure, 
            # how deep will this stack really go? i.e. for real
            # automata, how many nested loops will there be?
            # I figure not many.
            loop_algebra = generate_internal(sub_nodes, sub_edges, branch[0], accept_states, end_states, sub_branches_analysis, sub_loops_analysis)

            if ALG_DEBUG:
                print "Recursion done"
                print "Got " + str(loop_algebra) + " back"

            # There will be a removed edge at the end of the loop
            # algebra linking it back to the start node to allow
            # the analysis to finish.  We need to replace that:
            # This should be able to handle more cases, just need
            # to figure out how to push the removed edge constant
            # into the loop algebra to the appropriate term.
            assert len(removed_edges) == 1
            loop_algebra = Sum(loop_algebra, Const(1, [removed_edges[0]]))
            if ALG_DEBUG:
                print "Generated loop algebra is:"
                print(loop_algebra)

            algebra = algebra_stack[-1][algebra_stack_counts[-1] - 1]
            algebra_stack[-1][algebra_stack_counts[-1] - 1] = Sum(algebra, Product(loop_algebra))

        # If this branch is a dead-end, add that
        # to the result equation:
        if len(non_loops_from_this_node) == 0:
            if node in end_states:
                # Save an end state onto the stack.
                algebra = algebra_stack[-1][algebra_stack_counts[-1] - 1]
                algebra = Sum(algebra, End()) if algebra else End()
                algebra_stack[-1][algebra_stack_counts[-1] - 1] = algebra
            else:
                pass
                # This case happens in subcalls where edges have been trimmed,
                # so the algorithm doesn't see the end of the graph
                # even if it exists.

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
        if len(non_loops_from_this_node) == 1:
            new_linalg = linear_algebra_for(non_loops_from_this_node[0], accept_states)
            # Persist that algebra into the stack.
            current_stack = algebra_stack[-1][algebra_stack_counts[-1] - 1]
            new_alg = new_linalg if current_stack is None else Sum(current_stack, new_linalg)
            algebra_stack[-1][algebra_stack_counts[-1] - 1] = new_alg

        # If this is a branch, go deeper in the algebra stack.
        if len(non_loops_from_this_node) > 1:
            branches_count = len(non_loops_from_this_node)

            # This keeps track of how many branches there
            # are to explore down this path.
            # Get the algebra for each branch
            branch_algebras = []
            added_count = 0
            for branch in non_loops_from_this_node:
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


# This basically looks to see if the branch 'branch'
# is a prefix to any of the loops analysis loops.
def isloopstart(branch, loops_analysis):
    branchstart = branch[0]

    for loop in loops_analysis[branchstart]:
        # We don't need total equality, just equality
        # in the second node in the loop.
        if len(loop) == 0:
            return True

        # If the branch is a single node large,
        # then all the loops starting from this node
        # count.
        if len(branch) == 1 or loop[1] == branch[1]:
            return True

    return False

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
