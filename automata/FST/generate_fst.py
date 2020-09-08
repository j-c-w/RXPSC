def expand_ranges(ranges):
    ranges = list(ranges)
    result = []
    for char_range in ranges:
        left = char_range.left.point
        right = char_range.right.point

        # Not sure I understand the case where these fail.
        # The packed input objects (see ste.py) seem excessively
        # complicated.
        assert len(left) == 1
        assert len(right) == 1
        result += range(left[0], right[0] + 1)
    return result

def edge_label_lookup_generate(atma):
    edges = list(atma.get_edges(data=True))
    lookup = {}
    for (to, from_ind, data) in edges:
        lookup[(to, from_ind)] = expand_ranges(data['symbol_set'])
    return lookup

def generate(unification, to_atma, from_atma, options):
    assert unification is not None
    # Get the lookup tables.

    to_edge_lookup = edge_label_lookup_generate(to_atma)
    from_edge_lookup = edge_label_lookup_generate(from_atma)

    return unification.unify_single_state(to_edge_lookup, from_edge_lookup, options)
