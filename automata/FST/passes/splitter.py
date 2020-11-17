import rxp_pass

from automata.FST.terms import *

# TODO --- This needs to take acceptance rate and the acceptance
# rate threshold into account.

# This is a pass that splits graphs into smaller components.
# It uses various heuristics to do so.
class SplitterPass(rxp_pass.Pass):
    def __init__(self):
        super(SplitterPass, self).__init__("SplitterPass")

    def execute(self, groups, options):
        from automata.FST.group_compiler import AutomataContainer
        import automata.FST.compilation_statistics as compilation_statistics
        if not options.use_splitter:
            return groups

        splits_at_start = compilation_statistics.splits_made
        new_groups = []
        for i in range(len(groups)):
            new_group = []
            for j in range(len(groups[i])):
                # Compute the new graph using one of the splitting methods.
                new_graphs = size_split(groups[i][j].automata, groups[i][j].algebra, options)

                # The new graphs have the same extra groups
                # as the old graphs, but we need to nullify
                # the algebras.
                for graph in new_graphs:
                    new_container = AutomataContainer(graph, None)
                    new_container.other_groups = groups[i][j].other_groups
                    new_group.append(new_container)

            new_groups.append(new_group)

        if options.print_split_stats:
            print "SPLIT STATS: Splitting finished, introducing", compilation_statistics.splits_made - splits_at_start, "Splits"

        return new_groups


# In this function, we split graphs into smaller graphs based on sizes.
def size_split(graph, algebra, options):
    from automata.FST.terms import Sum
    import automata.FST.compilation_statistics as compilation_statistics
    # We try to make sure that we only size-split at the top-level, ie.
    # that we don't create a size-split that means we need to go back
    # into the same graph again.
    # Get the top-level sum-statement only as a result.
    if algebra.issum():
        index = 0
        last_used_index = -1

        size_so_far = 0
        result_algebras = []

        # Go through each element of the sum, and split it based
        # on size.
        while index < len(algebra.e1):
            size_so_far += algebra.e1[index].size()
            
            # Need to check the next element is not accept or end --- if so, we need to continue on one more iteration
            # to make sure that the start of the next algebra is not invalid.
            if index < len(algebra.e1) - 1 and (algebra.e1[index + 1].isend() or algebra.e1[index + 1].isaccept()):
                index += 1
                continue

            if size_so_far > options.split_size_threshold:
                result_algebras.append(Sum(algebra.e1[last_used_index + 1: index + 1]).normalize())
                last_used_index = index
                size_so_far = 0

                compilation_statistics.splits_made += 1

            index += 1
        
        # Add anything that is left over to the results.
        if last_used_index < len(algebra.e1) - 1:
            result_algebras.append(Sum(algebra.e1[last_used_index + 1: index + 1]).normalize())
    else:
        result_algebras = [algebra]

    return algebras_to_graphs(result_algebras, graph.symbol_lookup)


def algebras_to_graphs(algebras, lookup):
    import automata.FST.algebra as alg
    result = []
    for algebra in algebras:
        result.append(alg.full_graph_for(algebra, lookup))

    return result


# Given a list of algebras, compute the occurances of various
# splits
def hash_counts(graphs, algebras, options):
    # Recursively go through all the terms in an algebra and
    # add them to the table.
    def hash_walk(algebra, table, options):
        # Don't hash things smaller than the splitting threshold.
        if algebra.size() < options.split_size_threshold:
            return

        structural_hash = algebra.structural_hash()
        if structural_hash in table:
            table[structural_hash] = table[structural_hash] + 1
        else:
            table[structural_hash] = 0

        if algebra.isconst():
            # Don't structural hash a const into the table --- too
            # simple to be worth extracting.
            pass
        elif algebra.isproduct():
            hash_walk(algebra.e1, table, options)
        elif algebra.isbranch():
            for opt in algebra.options:
                hash_walk(opt, table, options)
        elif algebra.issum():
            # Walk every entry from here until the end.
            # Use that to compute the number of indexes.
            for elt_index in range(len(algebra.e1)):
                hash_walk(algebra.e1[elt_index], table, options)

                sub_hash = Sum(algebra.e1[elt_index:]).normalize().structural_hash()
                if sub_hash in table:
                    table[sub_hash] += 1
                else:
                    table[sub_hash] = 0

        elif algebra.isend():
            pass
        elif algebra.isaccept():
            pass

    hash_table = {}
    for algebra in algebras:
        # Walk through the algebra terms.
        hash_walk(algebra, hash_table, options)

    return threshold(hash_table, options)


def threshold(table, options):
    # Go through the table and remove entries less than
    # a certain value.
    for entry in table.keys():
        if table[entry] < options.split_threshold_frequency:
            del table[entry]
    return table


# In this function we try to identify common sub-structures
# in graphs.  We split based on the sub-structures (hashed
# by their algebras).
def hash_split(graph, algebra, hash_counts, options):
    # Split an algebra up into smaller sections --- preserving
    # the property that each section must rejoin the
    # original algebra in exactly one place, making
    # computing the overall acceptance easier.
    # This condition could be relaxed, and would likely
    # result in much greater success.  However, this would
    # require additional complexity in handling the graphs.

    # The internal function returns two arguments --- the
    # list of algebras that has been split off, and a flag
    # indicating whether it should be split off.
    def split_algebra(algebra, hash_table, options, nosplit_here=False):
        if algebra.size() < options.split_size_threshold:
            return algebra, [], False

        structural_hash = algebra.structural_hash()
        if algebra.isconst():
            # Can't split inside a const.
            return algebra, [], False
        elif algebra.isproduct():
            # Can't split inside a product.
            return algebra, [], False
        elif algebra.isaccept():
            return algebra, [], False
        elif algebra.isend():
            return algebra, [], False
        elif algebra.isbranch():
            split_options = []
            kept_options = []

            for opt in algebra.options:
                kept, sub_splits, split_here = split_algebra(opt, hash_table, options, nosplit_here=True)
                assert not split_here # Cant split immediately after a branch, because we need to have a way to go into the thing
                # we just split off.
                if not kept.isend():
                    kept_options.append(kept)
                split_options += sub_splits

            if len(kept_options) == 0:
                new_base_algebra = End()
            else:
                new_base_algebra = Branch(kept_options).normalize()

            structural_hash = new_base_algebra.structural_hash()

            if structural_hash in hash_table and not nosplit_here and algebra.ends_with_end():
                return End(), split_options + [algebra], True
            else:
                return new_base_algebra, split_options, False
        elif algebra.issum() and algebra.ends_with_end():
            # This is really a postfix extractor rather than anything else.
            # This algorithm could easily be adapted to an infix algorithm, but needs
            # to deal with edge cases and restarting automata.
            # Go through and get the postfix out.
            tail_kept, tail_splits, split_at_tail = split_algebra(algebra.e1[-1], hash_table, options)
            algebra.e1[-1] = tail_kept

            # Go through each elt of the sum, and see if it is in
            # the structural hash.
            consider_from_index = 0
            if nosplit_here:
                consider_from_index = 1

            splits = []
            index = len(algebra.e1)

            while index >= consider_from_index:
                index -= 1

                sub_term = Sum(algebra.e1[index:]).normalize()
                structural_hash = sub_term.structural_hash()

                # Only actually do the split if it's big enough.
                if sub_term.size() > options.split_size_threshold and structural_hash in hash_table:
                    # Add this split to the list of splits, or the
                    splits.append(sub_term)
                    algebra = Sum(algebra.e1[:index] + [End()]).normalize()
            return algebra, splits, False
        elif algebra.issum() and not algebra.ends_with_end():
            # In this case, we can't really extract anything :(
            # I think there are algorithms to do this, but they'll require
            # running on a different form of the IR where the branches aren't
            # merged together.
            pass

    res_alg, splits, _ = split_algebra(algebra, hash_counts, options, nosplit_here=True)

    splits.append(res_alg)
    splits = [x for x in splits if not x.isend()]
    return splits
