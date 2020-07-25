# Get the algebtra terms
from terms import *
import sjss
from repoze.lru import lru_cache
import itertools

ALG_DEBUG = False
# This should probably be enabled for most things, it
# drastically helps avoid exponential blowup for non-SJSS
# graphs.
CACHE_ENABLED = True

# How many permutations should we explore when trying to
# find branch equality?
PERMUTATION_THRESHOLD = 10000

def generate(nodes, edges, start, accept_states):
    # Clear the results cache for 'generate_internal'
    clean_caches()

    branches = sjss.compute_branches(nodes, edges, start)
    loops = sjss.compute_loops(nodes, edges, start)
    end_states = sjss.compute_end_states(nodes, edges)

    result = generate_internal(nodes, edges, start, accept_states, end_states, branches, loops)
    result = result.normalize()

    return result


def clean_caches():
    global results_cache
    global computed_algebras

    results_cache = {}
    computed_algebras = {}


# See the reasoning about caches.  Hopefully this approximation
# of argument equality doesn't cause any super hard to track
# down bugs.
def unique_cache_index_for(edges, start):
    edges = tuple(edges)
    return hash(hash(start) + hash(edges))


# This is a results cache.  In theory, for a truly SJSS
# automata, this isn't needed --- the following function won't
# recurse with arguments that is has already been called with.
# However, in practice, there are structures that result in
# exponential blowup of algebra size wrt. automata size.
# If we converted to SJSS, we would face the brunt of this
# cost, so we don't do that.
# This conversion algorithm handles those case correctly,
# but to avoid an exponential number of calls, we cache
# partial results.
# This is obviously best cleared after running this.
# There are two types of cache, the 'computed_algebras'
# cache, which caches results /within/ a single invocation
# of generate_internal, and 'results_cache', which
# caches recursive results.
computed_algebras = {}
results_cache = {}
hits = 0
node_cache_hits = 0
recursions = 0
def generate_internal(nodes, edges, start, accept_states, end_states, branches_analysis, loops_analysis):
    global computed_algebras
    global node_cache_hits

    if CACHE_ENABLED:
        unique_cache_index = unique_cache_index_for(edges, start)
        if unique_cache_index in results_cache:
            global hits
            hits += 1
            return results_cache[unique_cache_index]

    output_lookup = sjss.generate_output_lookup(nodes, edges)
    if ALG_DEBUG:
        global recursions
        recursions += 1
        print("Recursions:")
        print(recursions)
        print("Results cache has: " + str(len(results_cache)) + " entries")
        print("(Hits: " + str(hits))
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
    algebra_stack = [[Const(0, [])]]
    # This keeps track of the node that each stack element represents.
    algebra_stack_nodes = [None]
    algebra_stack_counts = [1]

    while len(nodes_list) != 0:
        node = nodes_list[0]
        nodes_list = nodes_list[1:]
        try_to_compress = False

        if CACHE_ENABLED and node in computed_algebras:
            # Set the top of stack to this node
            # and try to compress
            algebra = computed_algebras[node]
            current_tail = algebra_stack[-1][algebra_stack_counts[-1] - 1]
            algebra_stack[-1][algebra_stack_counts[-1] - 1] = Sum([current_tail, algebra])

            try_to_compress = True


            if ALG_DEBUG:
                node_cache_hits += 1
                print "Hit cache for node " + str(node)
                print "Found result " + str(algebra)
        else:
            # Check if we already have a cached result
            # for 'node'
            if ALG_DEBUG:
                print("For node " + str(node))
                print("Algebra stack is: ")
                print([[str(i) for i in x] for x in algebra_stack])
                print("Algebra stack indexes are:")
                print(algebra_stack_counts)
                print("Node stack is: ")
                print(algebra_stack_nodes)
                print("Node cache hits so far is:")
                print(node_cache_hits)
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
                elif end != branch[0]:
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

                # We need to construct a branch --- critically,
                # we need to apply the right edge to the end 
                # of each algebra.

                if loop_algebra.isbranch():
                    branch_elems = loop_algebra.e1
                else:
                    branch_elems = [loop_algebra]

                # We know that each removed edge will be of
                # the form [last_node_in_loop, loop_start]
                # So we can just add a sum to the end of the loop:
                for i in range(len(branch_elems)):
                    elemn = branch_elems[i]

                    last_edge = elemn.get_last_node()
                    if last_edge:
                        branch_elems[i] = Sum([branch_elems[i], Const(1, [(last_edge, node)])])
                    else:
                        # This is likely due to a branch within
                        # the branch --- we really need some
                        # complicated recusion here to 
                        # then go down every
                        # branch and add the + 1 to
                        # every end.
                        pass

                for (src, dst) in removed_edges:
                    if src == dst:
                        # THis means there is a 1-loop,
                        # which was removed from the underlying
                        # analysis, so add it back in.
                        branch_elems.append(Const(1, [(src, dst)]))

                loop_algebra = Sum([loop_algebra, Const(1, [removed_edges[0]])])
                if ALG_DEBUG:
                    print "Generated loop algebra is:"
                    print(loop_algebra)

                algebra = algebra_stack[-1][algebra_stack_counts[-1] - 1]
                algebra_stack[-1][algebra_stack_counts[-1] - 1] = Sum([algebra, Product(loop_algebra)])

            # If this branch is a dead-end, add that
            # to the result equation:
            if len(non_loops_from_this_node) == 0:
                try_to_compress = True
                if node in end_states:
                    # Save an end state onto the stack.
                    algebra = algebra_stack[-1][algebra_stack_counts[-1] - 1]
                    algebra = Sum([algebra, End()]) if algebra else End()
                    algebra_stack[-1][algebra_stack_counts[-1] - 1] = algebra
                else:
                    pass
                    # This case happens in subcalls where edges have been trimmed,
                    # so the algorithm doesn't see the end of the graph
                    # even if it exists.


            # Dunno how often this case comes up, since things should
            # be compressed.  Might happen for the  start node anyway
            if len(non_loops_from_this_node) == 1:
                new_linalg = linear_algebra_for(non_loops_from_this_node[0], accept_states)
                # Persist that algebra into the stack.
                current_stack = algebra_stack[-1][algebra_stack_counts[-1] - 1]
                new_alg = new_linalg if current_stack is None else Sum([current_stack, new_linalg])
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
                algebra_stack_nodes.append(node)

        if try_to_compress:
            # This loop 'builds' the algebra stack by adding things
            # together and creating branches where there are multiple
            # options.
            # There are two cases where we try to compress ---
            # if we have just loaded something from the cache,
            # or if we have reached a dead-end earlier on.
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

                    if CACHE_ENABLED:
                        # (Put it in the cache too
                        results_cache[unique_cache_index] = algebra

                    return algebra


                if CACHE_ENABLED:
                    # Cache the resutls in the local cache 
                    # if we are not returning.
                    node_we_finished = algebra_stack_nodes[-1]
                    if ALG_DEBUG:
                        print "Node finished :", node_we_finished
                        print "Has algebra", str(algebra)
                    computed_algebras[node_we_finished] = algebra
                del algebra_stack_nodes[-1]
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
                    new_head_of_stack = Sum([new_head_of_stack, algebra])
                algebra_stack[-1][algebra_stack_counts[-1] - 1] = new_head_of_stack


            # Decrement the stack counts.
            algebra_stack_counts[-1] = algebra_stack_counts[-1] - 1

    # We should not reach the end here.
    assert False


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
        sum_elt = Sum(algebra)

    if ALG_DEBUG:
        print "Computed Algebra is " + str(sum_elt)

    return sum_elt


# This is an implementation of the comparison operator ---
# the concept is that if A < B, then  we can use the circuit
# for B to implement A.  This does not have to do with
# the generation of a translator, which will likely
# use effectively the same algorithm.  See the theory
# paper for a description of this algorithm in a sensible
# manner.
# The problem is that some of these things are massive,
# so a recursive implementation would exceed stack limits
# in python.
comparison_cache = {}
def leq(A, B):
    global comparison_cache
    comparison_cache = {}

    return leq_internal(A, B)

# Trim equality means we can use <= instead of == to compare
# integers --- only applies at the end of statements.
def leq_internal(A, B, trim_equality=False):
    cache_pointer = (A.id, B.id)

    # See if we have a cached comparison for these.
    if CACHE_ENABLED and cache_pointer in comparison_cache:
        return comparison_cache[cache_pointer]

    result = None
    if A.isconst():
        if B.isconst():
            if trim_equality and A.val <= B.val:
                result = True
            elif A.val == B.val:
                result = True
            else:
                result = False
        elif B.isproduct():
            result = True
        else:
            result = False
    elif B.isend():
        result = True
    elif A.isaccept() and B.isaccept():
        result = True
    elif A.isproduct() and B.isproduct():
        result = leq_internal(A.e1, B.e1)
    elif A.issum() and B.issum():
        element_equality = []

        if len(A.e1) > len(B.e1):
            result = False
        elif len(A.e1) < len(B.e1) and not A.e1[-1].isend():
            # The part about the end statement is applying
            # the (trim) rule.
            result = False
        elif len(A.e1) == len(B.e1):
            # There are some optimizations to be done
            # here, to give more equality.  Not too
            # sure what the are ATM.
            still_equal = True
            for i in range(len(A.e1)):
                if i == len(A.e1) - 3:
                    # This is the third to last element ---
                    # it is likley to be often that the last elements
                    # are a + e, so do a comparison
                    # that allows for unequal integers at the end.
                    if A.e1[-1].isend() and B.e1[-1].isend() and A.e1[-2].isaccept() and B.e1[-2].isaccept():
                        still_equal = still_equal and leq_internal(A.e1[-3], B.e1[-3], trim_equality=True)
                else:
                    still_equal = still_equal and leq_internal(A.e1[i], B.e1[i])
            result = still_equal
        else:
            #assme the A ends with End()
            # are the same.
            # Compute equality of all the elements.
            still_equal = True
            for i in range(len(A.e1) - 1):
                if A.e1[-1].isend() and B.e1[-1].isend() and A.e1[-2].isaccept() and B.e1[-2].isaccept():
                    still_equal = still_equal and leq_internal(A.e1[-3], B.e1[-3], trim_equality=True)

                still_equal = still_equal and leq_internal(A.e1[i], B.e1[i])

            result = still_equal

        # TODO --- This is where we can use the loop rolling
        # property.
    elif A.isbranch() and B.isbranch():
        elements_A = A.options
        elements_B = B.options

        if len(elements_A) > len(elements_B):
            # No way to map these things.
            result = False
        else:
            # Try all matches --- likely need some kind
            # of cutoff here.
            matches = [None] * len(elements_A)
            for i in range(len(elements_A)):
                matches[i] = [None] * len(elements_B)

                for j in range(len(elements_B)):
                    matches[i][j] = leq_internal(elements_A[i], elements_B[j])

            result = False
            # Now, try and find some match for each i element in A.
            perm_count = 0

            # This could be made smarter, by making some sensible guesses
            # about which permutations might work --- there are definitely
            # constraints here, e.g. we expect the matches matrix
            # to be quite sparse.
            for combination in permutations(len(elements_A), range(len(elements_B))):
                perm_count += 1
                if perm_count > PERMUTATION_THRESHOLD:
                    print "Warning: Permutation fail due to excessive numbers"
                    break
                # check if the combination is good.
                is_match = True
                for i in range(len(combination)):
                    if not matches[i][combination[i]]:
                        is_match = False
                if is_match:
                    result = True
                    break
    else:
        result = False

    comparison_cache[cache_pointer] = result
    if result is None:
        print "Failed to produce a comparison for:"
        print A
        print B
    assert result is not None
    return result


# Yield every cpermustations of i numbers up to j.
def permutations(i, j):
    return itertools.permutations(j, i)
