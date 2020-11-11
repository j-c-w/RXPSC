# Due to poor choices, we largely don't use this.
import simple_graph
from automata.elemnts.element import StartType
from automata.elemnts.element import FakeRoot
from automata.elemnts.ste import list_to_packed_set 
from automata.elemnts.ste import S_T_E
from unifier import FastSet
import generate_fst
import automata.elemnts.ste
import networkx as nx
import time
import generate_fst
from automata.automata_network import Automatanetwork

DEBUG_COMPUTE_BRANCHES = False

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
    if DEBUG_COMPUTE_BRANCHES:
        print "Starting compute branches..."
        print "Edges are, ", edges
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
        if DEBUG_COMPUTE_BRANCHES:
            print "Next Iteration Started:"
            print "Next node stack is ", next_node
            print "Current node is ", node
            print "Visisted?", visited[node]

        # Check if visited:
        if visited[node] > 0:
            # We have found a loop.  We don't need to do
            # anything as we are simply looking for branches
            # here.
            # Do append this to the current branch though so
            # we know where it ends.
            assert (branches[-1][-1], node) in edges

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

    if DEBUG_COMPUTE_BRANCHES:
        print "Completed branches are"
        print completed_branches
    return completed_branches

# Return a list of nodes that preceed the edges in target edges.
# (i.e. the union of all the first elements to all the edges.
def get_node_before_edges(target_edges):
    result = set()
    for edge in target_edges:
        if edge[0] not in result:
            result.add(edge[0])
    return result

def splice_between(source_graph, before_node, after_node, splice_graph, last_nodes_in_splice_graph):
    max_node_number = max(source_graph.nodes)

    new_splice_graph, node_mapping = relabel_from(max_node_number + 1, splice_graph.clone(), return_mapping=True)

    # Convert the last nodes list to the same numbering system.
    for i in range(len(last_nodes_in_splice_graph)):
        last_nodes_in_splice_graph[i] = node_mapping[last_nodes_in_splice_graph[i]]

    # Now, we go through, and replace any reference to the start node with before_node,
    # and any reference to any last_nodes_in_splice_graph with after_node
    # Then we combine all the fields.
    for i in range(len(new_splice_graph.nodes) - 1, -1, -1):
        # Just delete the node --- it won't exist anymore.
        if new_splice_graph.start_state == new_splice_graph.nodes[i]:
            del new_splice_graph.nodes[i]
        elif new_splice_graph.nodes[i] in last_nodes_in_splice_graph:
            del new_splice_graph.nodes[i]

    # Edges and the edge-symbol lookup.
    for i in range(len(new_splice_graph.edges)):
        (from_node, to_node) = new_splice_graph.edges[i]
        if from_node == new_splice_graph.start_state:
            from_node = before_node
        elif from_node in last_nodes_in_splice_graph:
            from_node = after_node
        if to_node == new_splice_graph.start_state:
            to_node = before_node
        elif to_node in last_nodes_in_splice_graph:
            to_node = after_node

        # Reconfigure the edge-symbol lookup
        new_edge = (from_node, to_node)
        # Only change things if the edge changes.
        if new_edge != new_splice_graph.edges[i]:
            new_splice_graph.symbol_lookup[new_edge] = new_splice_graph.symbol_lookup[new_splice_graph.edges[i]]
            del new_splice_graph.symbol_lookup[new_splice_graph.edges[i]]
            new_splice_graph.edges[i] = new_edge

    # Accept states
    for i in range(len(new_splice_graph.accepting_states)):
        if new_splice_graph.accepting_states[i] == new_splice_graph.start_state:
            new_splice_graph.accepting_states[i] = before_node
        if new_splice_graph.accepting_states[i] in last_nodes_in_splice_graph:
            new_splice_graph.accepting_states[i] = after_node

    # Now that we have the rebuilt graph, we just need to splice it into the current graph:
    result_nodes = source_graph.nodes + new_splice_graph.nodes
    result_edges = source_graph.edges + new_splice_graph.edges
    result_symbol_lookup = dict(source_graph.symbol_lookup)
    for edge in new_splice_graph.edges:
        result_symbol_lookup[edge] = new_splice_graph.symbol_lookup[edge]

    result_accepting_states = source_graph.accepting_states + new_splice_graph.accepting_states

    graph = simple_graph.SimpleGraph(result_nodes, result_edges, result_symbol_lookup, result_accepting_states, source_graph.start_state)

    assert is_homogenous(graph)
    return graph

# Splice a graph into another graph at a given node --- this works
# by (1) renumbering all the new graph nodes so they are unique
# then (2) by replacing the start node of the input graph
# with the starting-point node of the graph to be splied onto.
# It also updates the symbol lookup and the accepting states
# for the first graph.
# Returns: the new nodes, the new edges, the new accepting states
# and the
def splice_after(source_graph, target_node, splice_graph):
    # First, get the max node number, and just add onto that:
    max_node_number = max(source_graph.nodes)

    # Relabel the graph:
    new_splice_graph = relabel_from(max_node_number + 1, splice_graph.clone())

    # Now, we need to go through all the above fields, and replace
    # any reference to the start state with references to the
    # node.
    # nodes
    for i in range(len(new_splice_graph.nodes) - 1, -1, -1):
        # Just delete the node --- it won't exist anymore.
        if new_splice_graph.start_state == new_splice_graph.nodes[i]:
            del new_splice_graph.nodes[i]
    # Edges and the edge-symbol lookup.
    for i in range(len(new_splice_graph.edges)):
        (from_node, to_node) = new_splice_graph.edges[i]
        if from_node == new_splice_graph.start_state:
            from_node = target_node
        if to_node == new_splice_graph.start_state:
            to_node = target_node
        # Reconfigure the edge-symbol lookup
        new_edge = (from_node, to_node)
        # Only change things if the edge changes.
        if new_edge != new_splice_graph.edges[i]:
            new_splice_graph.symbol_lookup[new_edge] = new_splice_graph.symbol_lookup[new_splice_graph.edges[i]]
            del new_splice_graph.symbol_lookup[new_splice_graph.edges[i]]
            new_splice_graph.edges[i] = new_edge

    # Accept states
    for i in range(len(new_splice_graph.accepting_states)):
        if new_splice_graph.accepting_states[i] == new_splice_graph.start_state:
            new_splice_graph.accepting_states[i] = target_node

    # Now that we have the rebuilt graph, we just need to splice it into the current graph:
    result_nodes = source_graph.nodes + new_splice_graph.nodes
    result_edges = source_graph.edges + new_splice_graph.edges
    result_symbol_lookup = dict(source_graph.symbol_lookup)
    for edge in new_splice_graph.edges:
        result_symbol_lookup[edge] = new_splice_graph.symbol_lookup[edge]

    result_accepting_states = source_graph.accepting_states + new_splice_graph.accepting_states

    graph = simple_graph.SimpleGraph(result_nodes, result_edges, result_symbol_lookup, result_accepting_states, source_graph.start_state)
    assert is_homogenous(graph)
    return graph

def relabel_from(new_lowest_number, graph, return_mapping=False):
    node_label_mapping = {}
    # Construct a mapping for the old numbers to the new ones.
    for node in graph.nodes:
        node_label_mapping[node] = new_lowest_number
        new_lowest_number += 1

    # Rebuild every component
    for i in range(len(graph.nodes)):
        graph.nodes[i] = node_label_mapping[graph.nodes[i]]
    for i in range(len(graph.edges)):
        old_edge = graph.edges[i]
        graph.edges[i] = (node_label_mapping[graph.edges[i][0]],
                        node_label_mapping[graph.edges[i][1]])
        graph.symbol_lookup[graph.edges[i]] = graph.symbol_lookup[old_edge]
    for i in range(len(graph.accepting_states)):
        graph.accepting_states[i] = node_label_mapping[graph.accepting_states[i]]
    graph.start_state = node_label_mapping[graph.start_state]

    if return_mapping:
        return graph, node_label_mapping
    else:
        return graph


# Given an input graph, determine whether it is homogenous
def is_homogenous(simple_graph):
    input_lookup = generate_input_lookup(simple_graph.nodes, simple_graph.edges)

    for node in simple_graph.nodes:
        input_edges = input_lookup[node]

        for i in range(len(input_edges)):
            if simple_graph.symbol_lookup[(input_edges[0], node)] != simple_graph.symbol_lookup[(input_edges[i], node)]:
                print "Edges are"
                print input_edges[0], node
                print input_edges[i], node
                print simple_graph.symbol_lookup[(input_edges[0], node)]
                print simple_graph.symbol_lookup[(input_edges[i], node)]
                return False

    return True


# Given an AutomataNetwork object from grapefruit, get
# back a list of nodes and edges.
def automata_to_nodes_and_edges(automata):
    edges = automata.edge_ids_list()
    accepting_states = automata.reporting_states_list()
    start_state = automata.fake_root.id
    assert start_state == 0
    return simple_graph.SimpleGraph(automata.node_ids_list(), edges, generate_fst.edge_label_lookup_generate(automata), accepting_states, start_state)

# Convert a nodes and edges representation to an automata network
# object.
def nodes_and_edges_to_automata(simple_graph):
    id = 'testid' # Does this need to be insightful?

    # Need to reconstruct the STEs for each node:
    stes = []
    symbol_sets = []
    node_lookup = {}
    starts = set() # Keep track of the real start nodes.
    for start, finish in simple_graph.edges:
        if start == 0:
            starts.add(finish)

    for node in simple_graph.nodes:
        reporting = node in simple_graph.accepting_states
        # Get the symbol table from the symbol lookup.
        for edge in simple_graph.edges:
            start, finish = edge
            # Assume homogenous, so all inputs are the same.
            if finish == node:
                symbols = list_to_packed_set(simple_graph.symbol_lookup[edge], ints=True)
                break
        assert len(symbols) > 0

        # 0 is always the identity of the fake root. --- skip it.
        if node == 0:
            continue
        else:
            if node in starts:
                start_type = StartType.all_input
            else:
                start_type = StartType.non_start

            # Ignore the report code because that isn't actually
            # what we are interested in?  Just research software yo
            stes.append(S_T_E(start_type, reporting, False, node, symbols, None, (0 if reporting else -1), report_code=1))

            node_lookup[node] = stes[-1]
            symbol_sets.append(symbols)

    graph = nx.MultiDiGraph()
    for i in range(len(stes)):
        graph.add_nodes_from(stes, data=stes[i])
    for from_n, to_n in simple_graph.edges:
        if from_n == 0:
            continue
        assert to_n != 0 # Can't have an edge going /to/ the
        # start no
        from_ste = node_lookup[from_n]
        to_ste = node_lookup[to_n]

        # Need to make sure that these are actually the same...
        reslookup = FastSet(generate_fst.expand_ranges(automata.elemnts.ste.list_to_packed_set(set(simple_graph.symbol_lookup[(from_n, to_n)]), ints=True)))
        assert set(reslookup) ==  set(simple_graph.symbol_lookup[(from_n, to_n)])

        graph.add_edge(from_ste, to_ste, symbol_set=automata.elemnts.ste.list_to_packed_set(simple_graph.symbol_lookup[(from_n, to_n)], ints=True))
    max_node_val = max(graph.nodes)
    # Assume homogenous, and stride of 1.
    # Also assume that we are using byte representation
    # of inputs, which means 255 I think.
    result = Automatanetwork._from_graph(id, True, graph, 1, max_node_val, 255, add_fake_root=True)
    return result

# This is intended to be a shitty, 'good enough' hash rather than
# some sure thing...
def hash_graph(simple_graph):
    node_count = len(simple_graph.nodes)
    edge_count = len(simple_graph.edges)
    nodes = sorted(simple_graph.nodes)
    edges = sorted(simple_graph.edges)

    return str(node_count) + "-" + str(edge_count) + "-" + ",".join([str(x) for x in nodes[max(0, len(nodes) - 20):]]) + "-" + ",".join([str(x) for x in edges[max(0, len(edges) - 40):]])
