import sys
import FST

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
        symbols = expand_ranges(data['symbol_set'])
        lookup[(to, from_ind)] = symbols
    return lookup

def generate(unification, to_atma, from_atma, options):
    assert unification is not None
    # Get the lookup tables.

    to_edge_lookup = edge_label_lookup_generate(to_atma)
    from_edge_lookup = edge_label_lookup_generate(from_atma)

    best_result = None
    best_structural_modification_count = 1000000
    for unifier in unification:
        if not unifier:
            continue

        if options.target == 'single-state':
            result = unifier.unify_single_state(from_edge_lookup, to_edge_lookup, options)
        elif options.target == 'symbol-only-reconfiguration':
            result = unifier.unify_symbol_only_reconfigutaion(to_edge_lookup, from_edge_lookup, options)
        elif options.target == 'perfect-unification':
            result = FST.AllPowerfulUnifier()
        else:
            print "Unkown target " + options.target
            sys.exit(1)

        # We want to return results with 0 structural modification
        # where possible.  So, if we find one with structural
        # modification, then keep going.
        structural_modification_count = unifier.structural_modification_count()
        if result and structural_modification_count == 0:
            # Auto-return if we get an answer with 0 modification
            return result
        elif result and structural_modification_count < best_structural_modification_count:
            best_result = result
            best_structural_modification_count = structural_modification_count

    return best_result
