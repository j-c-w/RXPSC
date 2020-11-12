import rxp_pass

class PrefixSplitPass(rxp_pass.Pass):
    def __init__(self):
        super(PrefixSplitPass, self).__init__("PrefixSplit")

    def execute(self, groups, options):
        return prefix_split(groups, options)

# This function does a prefix splitting for the automata
# components.  It does this in the accepting path algebra,
# because that was just a bit easier to write.
# After doing prefix splitting, it separates out the prefixes
# and the postfixes to enable better cross compilation.
# IT IS NOT INTENDED FOR FULL USE IN THE CURRENT IMPLEMENTATION,
# because it is not correctly hooked in with the backend.
# The backend would ideally generate fake start states
# for the prefix-split automata, but instead they are
# treated as real start states.
# In other words, the current implementation provides the
# numbers, but not a real implementation of prefix-split
# automata.
def prefix_split(groups, options):
    import automata.FST.algebra as alg
    from automata.FST.group_compiler import AutomataContainer
    import automata.FST.single_compiler as sc

    # keep track of the new automata that we should add.
    prefixes_to_add = []
    automata_to_remove = set()

    # Now, compute the prefixes of every pair of automata:
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            if (i, j) in automata_to_remove:
                # This algebra has been completely replaced
                # by a prefix --- move onto next.
                continue

            shared_prefixes = []
            # Aim is to only iterate over the /subsequent/
            # automata so to avoid prefix splitting in loops.
            for i2 in range(i, len(groups)):
                if i2 == i:
                    jrange = range(j + 1, len(groups[i2]))
                else:
                    jrange = range(len(groups[i2]))
                for j2 in jrange:
                    if i2 == i and j2 == j:
                        continue

                    shared_prefix, tail_first, tail_second = alg.prefix_merge(groups[i][j].algebra, groups[i][j].automata.symbol_lookup, groups[i2][j2].algebra, groups[i2][j2].automata.symbol_lookup, options)

                    if shared_prefix is not None and shared_prefix.size() > options.prefix_size_threshold:
                        # Replace the two algebras if the shared
                        # prefix is big enough.
                        shared_prefixes.append((i2, j2, shared_prefix, tail_first, tail_second))

            # Now, compute the prefix split --- find the biggest
            # possible prefix and split that.
            # There are other good heuristics we could use here, 
            # like find the prefix with the most possible states.
            split_size = -1
            splits = []
            prefix_groups_required = set()
            for (i2, j2, prefix, tail_first, tail_second) in shared_prefixes:
                if prefix.size() > split_size:
                    prefix_groups_required = groups[i][j].other_groups.union(groups[i2][j2].other_groups)
                    split_size = prefix.size()
                    splits = [(i2, j2, prefix, tail_first, tail_second)]
                # This equality might not be right -- it computes 
                # structural equality, and clearly we are after
                # strict equality.  That said, I think they are
                # the same here, because we know that these are prefixes
                # for this automata, so they must be equal to the existing
                # prefix.
                elif prefix.size() == split_size and prefix.equals(splits[0][2]):
                    splits.append((i2, j2, prefix, tail_first, tail_second))
                    prefix_groups_required = prefix_groups_required.union(groups[i2][j2].other_groups)
                else:
                    continue
            # Execute the splits
            group1_set = False
            old_symbol_lookup = groups[i][j].automata.symbol_lookup
            for (i2, j2, prefix, tail_first, tail_second) in splits:
                prefix_groups_required.add(i2)
                if not group1_set:
                    if tail_first is None:
                        automata_to_remove.add((i, j))
                    else:
                        new_graph = alg.full_graph_for(tail_first, groups[i][j].automata.symbol_lookup)
                        wrapper = AutomataContainer(new_graph, sc.compute_depth_equation(new_graph, options))
                        wrapper.other_groups = groups[i][j].other_groups
                        groups[i][j] = wrapper

                    # Only set the source group the first time.
                    group1_set = True

                if tail_second is None:
                    automata_to_remove.add((i2, j2))
                else:
                    new_graph = alg.full_graph_for(tail_second, groups[i2][j2].automata.symbol_lookup)

                    old_groups = groups[i2][j2].other_groups
                    groups[i2][j2] = AutomataContainer(
                            new_graph,
                            sc.compute_depth_equation(new_graph, options)
                        )
                    groups[i2][j2].other_groups = old_groups
            # The prefixes_to_add keeps track of what needs to be added to the automata list.
            # This makes sure that we don't add things to multiple
            # lists.
            if len(splits) > 0:
                prefixes_to_add.append((i, j, prefix, prefix_groups_required, old_symbol_lookup))

    # Now add the prefixes to the groups that need it.
    for i, j, prefix, other_groups, old_symbol_lookup in prefixes_to_add:
        new_graph = alg.full_graph_for(prefix, old_symbol_lookup)
        wrapper = AutomataContainer(new_graph, None)
        wrapper.other_groups = other_groups
        groups[i].append(wrapper)

    automata_to_remove = sorted(list(automata_to_remove))[::-1]
    for (i, j) in automata_to_remove:
        del groups[i][j]

    return groups


