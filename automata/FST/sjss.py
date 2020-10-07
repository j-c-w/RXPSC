# Due to poor choices, we largely don't use this.
import networkx as nx
import time

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

    TODO


def has_path(nodes, edges, outputlookup, node_from, node_to):
    TODO
    pass


def generate_input_lookup(nodes, edges):
    input_lookup = {}
    for n in nodes:
        input_lookup[n] = []

    for (src, dst) in edges:
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


# Given an automata, compute a "region" that is a
# loop, where a region may contain subloops.
# Each region is some number of branches, along
# with a start and end node.    We assume single
# entry loops only.
# This returns the nodes, edges, branches and loops
# within that region.
def compute_loop_subregion(nodes, edges, loop_start_node, loops, branches):
    if loops[loop_start_node] == []:
        # There is no subregion here.
        return []
    else:
        search_nodes = []
        new_edges = []

        existing_edges_lookup = generate_output_lookup(nodes, edges)

        for loop in loops[loop_start_node]:
            # Add every node and every edge that
            # can be reached from the loop_start_node,
            # /provided/ they are in the loop. i.e.,
            # we do not add all edges that can be reached
            # from the start node, but after the start
            # node, we do add all edges.
            search_nodes.append(loop[1])

        # Now, go through and find every node that
        # can be reached from each of these nodes.
        # Store the edges in a dictionary.
        new_edges_dict = {}

        for node in search_nodes:
            visited = {}
            # Do not want to go past the start node.
            visited[loop_start_node] = 1
            to_visit = [node]

            while len(to_visit) > 0:
                node = to_visit[0]
                del to_visit[0]

                if node not in visited:
                    to_visit += existing_edges_lookup[node]
                    new_edges_dict[node] = [(node, dst) for dst in existing_edges_lookup[node]]

                    visited[node] = 1

        # Now, we have a list of nodes and edges, so construct
        # the new graph, and the new branches:
        new_nodes = new_edges_dict.keys()
        new_edges = []
        for node in new_nodes:
            new_edges += new_edges_dict[node]

        # We also have to add the edges from the loop start node
        # to the start of each loop at this point.
        for node in search_nodes:
            new_edges.append((loop_start_node, node))

        # Finally, the loop start node should also be in the
        # list of nodes.
        new_nodes.append(loop_start_node)
        removed_edges = []

        # However, the loop should not be completed,
        # so we also need to delete all edges coming
        # back into the starting node.
        i = 0
        while i < len(new_edges):
            (src, dst) = new_edges[i]
            if dst == loop_start_node:
                del new_edges[i]
                i -= 1

                removed_edges.append((src,dst))
            i += 1

        return (new_nodes, new_edges, removed_edges,
                compute_branches(new_nodes, new_edges, loop_start_node),
                compute_loops(new_nodes, new_edges, loop_start_node))


# Given a graph, compute the loops in it and group them
# by entry point on the shortest path from the start node.
# This returns a dictionary from [node] -> [list of loops
# whose shortest path from the start is on node]
# This does not support loops with multiple entrances,
# and neither does the path algebra.  Those loops should
# be cleaned up somewhere else.
def compute_loop_groups(nodes, edges, start):
    dists, paths = compute_sssp(nodes, edges, start)
    loops = compute_loops_with_duplicates(nodes, edges)

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


# This returns a dictionary from [node] -> [loops starting at that node]
# The loops are 'deduplicated', i.e. are not cyclic rotations
# of each other.
def compute_loops(nodes, edges, start):
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

counts = 0
# Given an automata, return a list of all loops that
# don't involved a repeated node.
def compute_loops_with_duplicates(nodes, edges):
    if edges == []:
        return []
    global counts
    counts += 1
    # print counts
    # Use NX to do large graphs because it has
    # an asymptotically faster algorithm.
    if len(edges) > 50:
        nx_graph = nx.MultiDiGraph()
        nx_graph.add_edges_from(edges)
        nx_cycles = list(nx.cycles.simple_cycles(nx_graph))
        # Convert these to the format we expect.
        for i in range(len(nx_cycles)):
            nx_cycles[i].append(nx_cycles[i][0])

        return nx_cycles


    output_lookup = generate_output_lookup(nodes, edges)
    # Algorithm is basically to do a DFS for each node, and
    # see if we get back to it.  (Every time we do, we note
    # that as a loop to that node).
    # This could definitely be more efficient, see
    # "finding all simple cycles in a digraph" on google.
    loops = []
    for search_node in nodes:
        next_node = [search_node]
        visited = {}
        for node in nodes:
            visited[node] = 0
        # This is a stack that keeps track of the last
        # place the brances diverged.  It report any full
        # path to any node by joining all the parts
        # of this path together.
        paths = [[]]

        while len(next_node) > 0:
            current = next_node[0]
            del next_node[0]
            paths[-1].append(current)

            if current == search_node and visited[current] == 1:
                # We have been here before, this is a loop.
                # We can get the loop path by appending all
                # the elements of the paths list.
                loops.append(paths[-1])
                # Clear the difference between the two paths.
                if len(paths) > 1:
                    prev_length = len(paths[-2])
                else:
                    prev_length = 0
                for i in range(prev_length, len(paths[-1]) - 1):
                    visited[paths[-1][i]] = 0
                paths = paths[:-1]
            elif visited[current] == 1:
                # We need to not recurse because we#e already been here.
                # Clear the difference between the two paths.
                if len(paths) > 1:
                    prev_length = len(paths[-2])
                else:
                    prev_length = 0
                for i in range(prev_length, len(paths[-1]) - 1):
                    visited[paths[-1][i]] = 0
                paths = paths[:-1]
            else:
                # Add the nodes to the list of what comes next.
                next_nodes = output_lookup[current]
                next_node = next_nodes + next_node

                visited[current] = 1

                if len(next_nodes) == 0:
                    # This is a dead-end.  Pop off the
                    # paths stack.
                    # Clear the difference between the two paths.
                    if len(paths) > 1:
                        prev_length = len(paths[-2])
                    else:
                        prev_length = 0
                    for i in range(prev_length, len(paths[-1]) - 1):
                        visited[paths[-1][i]] = 0
                    paths = paths[:-1]
                elif len(next_nodes) >= 1:
                    # This is a branch --- go down both
                    # options:
                    thispath = paths[-1]
                    del paths[-1]
                    for i in range(len(next_nodes)):
                        paths.append(thispath[:])

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
    input_lookup = generate_input_lookup(nodes, edges)
    visited = {}
    for node in nodes:
        visited[node] = 0
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
            prev_nodes = input_lookup[node]
            next_node = next_nodes + next_node
            branches[branchdepth].append(node)

            if len(next_nodes) > 1:
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
            elif len(prev_nodes) > 1: # We assume len(next_nodes) == 1
                # If there is more than one node coming in here,
                # also end the branch analysis, so that it
                # doesn't interfere with the loop analysis.
                completed_branches.append(branches[branchdepth])
                del branches[branchdepth]
                # Add an empty item to the list to continue the walk
                # this way.
                branches.append([node])

            branchdepth += len(next_nodes) - 1

    return completed_branches

# Return a list of nodes that preceed the edges in target edges.
# (i.e. the union of all the first elements to all the edges.
def get_node_before_edges(target_edges):
    result = set()
    for edge in target_edges:
        if edge[0] not in result:
            result.add(edge[0])
    return result

# Splice a graph into another graph at a given node --- this works
# by (1) renumbering all the new graph nodes so they are unique
# then (2) by replacing the start node of the input graph
# with the starting-point node of the graph to be splied onto.
# It also updates the symbol lookup and the accepting states
# for the first graph.
# Returns: the new nodes, the new edges, the new accepting states
# and the
def splice(nodes, edges, symbol_lookup, accepting_states, target_node,
        new_nodes, new_edges, new_start_states, new_symbol_lookup, new_accepting_states):
    # First, get the max node number, and just add onto that:
    max_node_number = max(nodes)

    # Relabel the graph:
    new_nodes, new_edges, new_start_states, new_symbol_lookup, new_accepting_states = relabel_from(max_node_number + 1, new_nodes, new_edges, new_start_states, new_symbol_lookup, new_accepting_states)

    assert len(new_start_states) == 1 # Only support inserts
    # with single entries.

    # Now, we need to go through all the above fields, and replace
    # any reference to the start state with references to the
    # node.
    new_start_state = new_start_states[0]
    # nodes
    for i in range(len(new_nodes) - 1, -1, -1):
        # Just delete the node --- it won't exist anymore.
        if new_start_state == new_nodes[i]:
            del new_nodes[i]
    # Edges and the edge-symbol lookup.
    for i in range(len(new_edges)):
        (from_node, to_node) = new_edges[i]
        if from_node == new_start_state:
            from_node = target_node
        if to_node == new_start_state:
            to_node = target_node
        # Reconfigure the edge-symbol lookup
        new_edge = (from_node, to_node)
        # Only change things if the edge changes.
        if new_edge != new_edges[i]:
            new_symbol_lookup[new_edge] = new_symbol_lookup[new_edges[i]]
            del new_symbol_lookup[new_edges[i]]
            new_edges[i] = new_edge

    # Accept states
    for i in range(len(new_accepting_states)):
        if new_accepting_states[i] == new_start_state:
            new_accepting_states[i] = target_node

    # Now that we have the rebuilt graph, we just need to splice it into the current graph:
    result_nodes = nodes + new_nodes
    result_edges = edges + new_edges
    result_symbol_lookup = dict(symbol_lookup)
    for edge in new_edges:
        result_symbol_lookup[edge] = new_symbol_lookup[edge]

    result_accepting_states = accepting_states + new_accepting_states

    return result_nodes, result_edges, result_symbol_lookup, result_accepting_states

def relabel_from(new_lowest_number, nodes, edges, start_states, symbol_lookup, accepting_states):
    node_label_mapping = {}
    # Construct a mapping for the old numbers to the new ones.
    for node in nodes:
        node_label_mapping[node] = new_lowest_number
        new_lowest_number += 1

    # Rebuild every component
    for i in range(len(nodes)):
        nodes[i] = node_label_mapping[nodes[i]]
    for i in range(len(edges)):
        old_edge = edges[i]
        edges[i] = (node_label_mapping[edges[i][0]],
                        node_label_mapping[edges[i][1]])
        symbol_lookup[edges[i]] = symbol_lookup[old_edge]
    for i in range(len(accepting_states)):
        accepting_states[i] = node_label_mapping[accepting_states[i]]
    for i in range(len(start_states)):
        start_states[i] = node_label_mapping[start_states[i]]

    return nodes, edges, start_states, symbol_lookup, accepting_states


# Given an AutomataNetwork object from grapefruit, get
# back a list of nodes and edges.
def automata_to_nodes_and_edges(automata):
    return automata.node_ids_list(), automata.edge_ids_list()
