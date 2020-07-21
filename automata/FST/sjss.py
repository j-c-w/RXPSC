# This file computes an SJSS graph from a non-SJSS input graph.

# I think I actually don't need to do this.  I think we
# can compute the accepting path algebra as if the automata
# were SJSS.
def compute_SJSS_graph(input_nodes, input_edges, start_node):
    # Aim (1) is to make sure that branches all join
    # at the same spot.
    # Do a DFS to find the branches.
    # Aim is to (2) look for loops, and duplicate nodes
    # on duplicate joins.
    # Algorithm here is: (1) find any loops.
    # (2) make note of all edges in loops.
    # (3) find any paths diverging /within/ the loop.
    # (4) Duplicate so that the loop doesn't diverge
    # unless the divergence rejoins before the end.
    pass


def loopify(nodes, edges, start_node, branches_analysis, loops_analysis):
    for loop in loops_analysis:
        outputlookup = generate_output_lookup(nodes, edges)
        inputlookup = generate_input_lookup(nodes, edges)

        # Go through, and make sure that there is (a)
        # only one exit from each node, except for
        # a single node, which is allowed to have
        # two exits.  If a node has two exits, it must
        # also have multiple entrances.
        # There must also only be a single place where
        # this can happen -- if there it not, this loop
        # has two entrances.
        loopstart_node = None
        for node in loop:
            if len(output_lookup[node]) > 1:
                if len(input_lookup[node]) > 1:
                    # Say this is the loop start.
                    if loopstart_node is None:
                        loopstart_node = node
                    else:
                        # This is a second 'entry'
                        # area to the loop.  We need
                        # to duplicate this loop, and 
                        # remove all edges into and out
                        # of this.
                        pass
                else: # len(input_lookup) <= 1:
                    # This is a place where something sp
                    pass


def branchify(nodes, edges, start_node, branches_analysis, loops_analysis):
    # Current thinking is that this doesn't need to happen.
    # Skipping it for now.
    # Aim here is to come up with a set of nodes that have
    # to be duplicated.
    for branch in branches_analysis:
        start = branch[0]
        end = branch[len(branch) - 1]
        branches_starting_at_same_place = []
        duplication_regions = []

        # Check if there are any branches that start
        # at the same place.


def has_path(nodes, edges, outputlookup, node_from, node_to):
    pass


def generate_input_lookup(nodes, edges):
    input_lookup = {}
    for n in nodes:
        input_lookup[n] = []

    for (src, dst) in nodes:
        input_lookup[dst].append(src)

    return input_lookup

def generate_output_lookup(nodes, edges):
    # Compute an efficient output lookup for the edges.
    output_lookup = {}
    for n in nodes:
        output_lookup[n] = []

    for (src, dst) in edges:
        output_lookup[src].append(dst)

    return output_lookup


# Given an automata, compute any "regions" that
# are loops, where a region may contain subloops.
# Each region is some number of branches, along
# with a start and end node.    We assume single
# entry loops only.
def compute_loop_subregions(nodes, edges, start_node):
    branches = compute_branches(nodes, edges, start_node)
    loops = compute_loops_deduplicated(nodes, edges, start_node)

    # We define the start of a subregion
    pass
    # TODO


# Given a graph, compute the loops in it and group them
# by entry point on the shortest path from the start node.
def compute_loop_groups(nodes, edges, start):
    dists, paths = compute_sssp(nodes, edges, start)
    loops = compute_loops(nodes, edges)

    # Group the loops by entry-point.  Once they
    # are grouped by entry point, then it is easy
    # to rotate and deduplicate them.
    groups = {}
    for node in nodes:
        groups[node] = []

    for loop in loops:
        loop_start = None
        start_dist = 1000000000000

        for node in loop:
            if dists[node] < start_dist:
                loop_start = node
                start_dist = dists[node]

        # We don't support loops with multiple
        # entrances to them ATM.  I think that
        # an SJSS property enforcement should
        # remove this stuff.
        for node in loop:
            assert loop_start in paths[node]

        groups[loop_start].append(loop)
    return groups


def compute_loops_deduplicated(nodes, edges, start):
    groups = compute_loop_groups(nodes, edges, start)

    # Now, we have all loops grouped by starting point.
    # Remove the ones that are just duplicates.
    for node in nodes:
        loops = groups[node]
        to_delete = []

        for i in range(len(loops)):
            # We want to preserve the loops that
            # start at teh node on the shortest path.
            if loops[i][0] != node:
                # This is the version of the loop where
                # the start isn't on the shortest path, so
                # ignore it.
                continue

            if i not in to_delete:
                # Every loop path starts and ends with the
                # same node, so delete the one on the front (i.e.
                # [n, m, ..., k, n]
                checking_loop = loops[i][1:]
                for j in range(len(loops)):
                    if i == j:
                        continue

                    if rotation(checking_loop, loops[j][1:]):
                        # Don't want to deal with a list changing
                        # under the lop iterators.
                        to_delete.append(j)

        # Delete things from greates to smallest
        # to avoid indexing problems.
        for index in sorted(to_delete)[::-1]:
            del groups[node][index]

    return groups


# nabbed from stackoverflow (https://stackoverflow.com/questions/31000591/check-if-a-list-is-a-rotation-of-another-list-that-works-with-duplicates/31000695)
def rotation(u, v):
    n, i, j = len(u), 0, 0
    if n != len(v):
        return False
    while i < n and j < n:
        k = 1
        while k <= n and u[(i + k) % n] == v[(j + k) % n]:
            k += 1
        if k > n:
            return True
        if u[(i + k) % n] > v[(j + k) % n]:
            i += k
        else:
            j += k
    return False


# Compute the shortest path to every node.  Returns
# a dictionary indexed by node that returns the shortest
# path to that node.
def compute_sssp(nodes, edges, source):
    node_dists = {}
    node_paths = {}

    for node in nodes:
        node_dists[node] = 10000000000
        node_paths[node] = []

    node_dists[source] = 0
    output_lookup = generate_output_lookup(nodes, edges)

    # I feel like there was some optimization I can't quite
    # remember here to avoid some loop iterations.
    changing = True
    while changing:
        changing = False
        for j in range(len(nodes)):
            node = nodes[j]
            outs = output_lookup[node]
            for dst in outs:
                if node_dists[dst] > node_dists[node] + 1:
                    node_dists[dst] = node_dists[node] + 1
                    node_paths[dst] = node_paths[node] + [node]
                    changing = True

    for node in nodes:
        node_paths[node].append(node)

    return (node_dists, node_paths)

# Given an automata, return a list of all loops.
def compute_loops(nodes, edges):
    output_lookup = generate_output_lookup(nodes, edges)
    # Algorithm is basically to do a DFS for each node, and
    # see if we get back to it.  (Every time we do, we note
    # that as a loop to that node).
    loops = []
    for search_node in nodes:
        next_node = [search_node]
        visited = [0] * len(nodes)
        # This is a stack that keeps track of the last
        # place the brances diverged.  It report any full
        # path to any node by joining all the parts
        # of this path together.
        paths = [[]]

        while len(next_node) > 0:
            current = next_node[0]
            next_node = next_node[1:]
            paths[len(paths) - 1].append(current)

            if current == search_node and visited[current] == 1:
                # We have been here before, this is a loop.
                # We can get the loop path by appending all
                # the elements of the paths list.
                loop_stops = []
                for partial_path in paths:
                    loop_stops += partial_path

                loops.append(loop_stops)
            elif visited[current] == 1:
                # We need to not recurse because we#e already been here.
                paths = paths[:-1]
            else:
                # Add the nodes to the list of what comes next.
                next_nodes = output_lookup[current]
                next_node += next_nodes

                visited[current] = 1

                if len(next_nodes) == 0:
                    # This is a dead-end.  Pop off the
                    # paths stack.
                    paths = paths[:-1]
                elif len(next_nodes) >= 1:
                    # This is a branch --- go down both
                    # options:
                    for i in range(len(next_nodes)):
                        paths.append([])
    return loops


def compute_end_states(nodes, edges):
    # Assume this is connected.
    end_states = []

    outputs = generate_output_lookup(nodes, edges)
    for node in nodes:
        if len(outputs[node]) == 0:
            end_states.append(node)
    return end_states

# Given an automata, return a list of all branches.
def compute_branches(nodes, edges, start_node):
    next_node = [start_node]
    output_lookup = generate_output_lookup(nodes, edges)
    visited = [0] * len(nodes)
    loops = []
    # Every path that is currently being walked.
    branches = [[]]
    branchdepth = 0

    # All the completely-walked paths.
    completed_branches = []

    while (len(next_node) > 0):
        node = next_node[0]
        next_node = next_node[1:]

        # Check if visited:
        if visited[node] > 0:
            # We have found a loop.  We don't need to do
            # anything as we are simply looking for branches
            # here.
            # Do append this to the current branch though so
            # we know where it ends.
            branches[branchdepth].append(node)

            # We should reduce the number of branches
            # by one -- this branch is complete.
            completed_branches.append(branches[branchdepth])
            del branches[branchdepth]

            branchdepth -= 1
        else:
            visited[node] = 1
            # Enqueue all the next:
            next_nodes = output_lookup[node]
            next_node = next_nodes + next_node
            branches[branchdepth].append(node)

            if len(next_nodes) > 1:
                # We are entering another split:

                # Make all the other branches too.
                for i in range(len(next_nodes)):
                    # This node is the first element in each
                    # of those branches
                    branches.append([node])
                # And move the current brnach to the completed list.
                completed_branches.append(branches[branchdepth])
                del branches[branchdepth]
            elif len(next_nodes) == 0:
                # Move the branch to the completed list.
                completed_branches.append(branches[branchdepth])
                del branches[branchdepth]

            branchdepth += len(next_nodes) - 1

    return completed_branches
