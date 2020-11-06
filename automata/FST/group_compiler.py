import single_compiler as sc
import compilation_statistics
from multiprocessing import Pool
import generate_fst
import tqdm
import time
import sjss
import unifier
from cache import ComparisonCache
import algebra as alg
import FST

try:
    import line_profiler
    from guppy import hpy
    from memory_profiler import profile
except:
    # Fails using pypy because the module
    # is not supported --- only used for memory
    # footprint debugging anyway
    pass


MODIFICATIONS_LIMIT = 10
DEBUG_COMPUTE_HARDWARE = False
DEBUG_COMPUTE_COMPAT_MATRIX = True
DEBUG_GENERATE_BASE = False
DEBUG_COMPILE_TO_EXISTING = True

# This is a class that contains a set of accelerated
# regular expressions.  We can use it to find which
# expressions can be accelerated.
group_id_counter = 0

class CompiledObject(object):
    def __init__(self, algebras, automatas):
        global group_id_counter
        assert len(algebras) == len(automatas)
        self.algebras = algebras
        self.automatas = automatas
        self.group_id = group_id_counter
        self.has_assignment = False
        group_id_counter += 1

    def set_assignment(assignment_indexes):
        self.assignment_indexes = assignment_indexes
        self.has_assignment = True

    def add_pattern(algebra, automata):
        assert self.has_assignment # Must already have assignment
        # to add to hardware.

        # Plan here is to find a hardware componenet
        # that this can be added to.
        # TODO
        assert False

# This is a class that contains the set of expressions
# that /are/ being accelerated on the hardware.
class HardwareAccelerators(object):
    def __init__(self, groups):
        # select the hardware we need, and compute
        # the utilization.
        self.hardware = compute_hardware_for(groups)
        self.groups = groups

# This is just a wrapper that stores the location and conversion
# machine required for a particular regular expression.
class CompilationIndex(object):
    def __init__(self, i2, j2, conversion, modifications):
        self.i = i2
        self.j = j2
        self.conversion_machine = conversion
        self.modifications = modifications

class AutomataContainer(object):
    def __init__(self, automata, algebra):
        self.automata = automata
        self.algebra = algebra

        self.symbol_lookup = generate_fst.edge_label_lookup_generate(self.automata.component)

class AutomataComponentWrapper(object):
    def __init__(self, ac):
        self.component = ac
        # Store a list of other groups that this automat is activated
        # within.
        self.other_groups = set()

# Store an automata to be implemented, and a set of translators
# to link up to it as a CCGroup (ConnectedComponent Group)
class CCGroup(object):
    def __init__(self, physical_automata, physical_algebra=None):
        self.physical_automata = physical_automata
        self.physical_algebra = physical_algebra
        self.supported_automata = []
        self.supported_algebras = []
        self.translators = []

    def add_automata(self, automata, algebra, translator):
        self.supported_automata.append(automata)
        self.supported_algebras.append(algebra)
        self.translators.append(translator)

# Given a group, compute a 3D array representing the cross
# compilability of that matrix.
def compute_cross_compatibility_matrix_for(group, options):
    read_comparison_cache, dump_comparison_cache = get_comparison_caches(options)

    results = [None] * len(group)
    compilation_list = [None] * len(group)
    # The compilation_list should be a 3D array that
    # is the inverse the results array, i.e. it contains
    # the list of all things that compile to a particular
    # automata.
    for i in range(len(group)):
        contents = [None] * len(group[i])
        for j in range(len(group[i])):
            contents[j] = []
        compilation_list[i] = contents

    # This flattens the tasks so that they can be
    # executed by a thread pool.
    tasks = []

    for i in range(len(group)):
        for j in range(len(group[i])):
            # Now, compile everything that is /not/ in this
            # group to this.
            tasks.append((group, i, j, options, read_comparison_cache, dump_comparison_cache))

    if options.line_profile:
        alg.profiler = line_profiler.LineProfiler()

    # Compute all the results:
    flat_results = []
    if options.cross_compilation_threading == 0:
        print "Executing single threaded"
        progress = tqdm.tqdm(total=(len(tasks)))
        # Compute traditionally:
        for (group_ref, i, j, options_ref, rcomparison_cache, wcomparison_cache) in tasks:
            if options.memory_debug:
                print "Memory Usage before during cross compat"
                h = hpy()
                print(h.heap())
            flat_results.append(compute_compiles_for((group_ref, i, j, options_ref, rcomparison_cache, wcomparison_cache)))
            progress.update(1)
        progress.close()
    else:
        pool = Pool(options.cross_compilation_threading)
        flat_results = [None] * len(tasks)
        index = 0
        for i, j, res, compilation_list_res in tqdm.tqdm(pool.imap_unordered(compute_compiles_for, tasks), total=len(tasks)):
            flat_results[index] = (i, j, res, compilation_list_res)
            index += 1

    if options.line_profile:
        alg.profiler.print_stats()

    # Now, expand the flat results back out:
    for i in range(len(group)):
        results[i] = [0] * len(group[i])

    for i, j, res, compilation_list_results in flat_results:
        results[i][j] = res

        # And also get the compilation_list setup:
        for i2, j2, comp_item in compilation_list_results:
            compilation_list[i2][j2].append(comp_item)

    # Dump the write comparison cache if it exists:
    if options.dump_comparison_cache:
        dump_comparison_cache.dump_to_file(options.dump_comparison_cache)

    if DEBUG_COMPUTE_COMPAT_MATRIX:
        print "Results of cross compatability"
        print results
        print "Compile from list is "
        print compilation_list
    return results, compilation_list

# This function compues the compatability of a single
# expression to every other expression --- it is designed
# to support multithreaded behaviour to help speed up this
# slow-ass python code.
def compute_compiles_for(args):
    # Expand the args, which are compressed to be passable
    # by the pool.map.
    group, i, j, options, rcomparison_cache, wcomparison_cache = args
    successful_compiles = []
    compilation_list = []
    source_automata = group[i][j]
    if DEBUG_COMPUTE_COMPAT_MATRIX:
        print "Comparing to automata: ", source_automata.algebra

    for i2 in range(len(group)):
        if i2 == i:
            # Don't want to count cross compilations
            # within the same group.
            continue

        # Some automata belong to more than one group, e.g. prefix
        # automata.  If this is one of those automata, then we
        # can't cross-compile within the group.
        if i2 in group[i][j].automata.other_groups:
            continue

        for j2 in range(len(group[i2])):
            if DEBUG_COMPUTE_COMPAT_MATRIX:
                print "Pair is ", i2, j2
                print "Comparied to ", i, j
            target_automata = group[i2][j2]

            # We can skip the comparison if it's not
            # in the comparison cache --- we wanted
            # to make a negative comparison cache to make
            # it a bit more flexible, but the problem is that
            # the size of that comparison cache would be too big.
            # We have a positive comparison cache instead, so we
            # skip if there is no entry in the cache.
            if options.comparison_cache:
                source_hash = source_automata.algebra.structural_hash()
                dest_hash = target_automata.algebra.structural_hash()

                if not rcomparison_cache.compilesto(source_hash, dest_hash):
                    continue

            # We could make this faster by not generating
            # the conversion machine here --- it could
            # use just the depth equation equality
            # here and only generate the conversion
            # machine where needed (i.e. later when
            # things are actually assigned).
            if DEBUG_COMPUTE_COMPAT_MATRIX:
                print "Comparing ", str(source_automata.algebra)
                print "(Hash)", str(source_automata.algebra.structural_hash())
                print " and, ", str(target_automata.algebra)
                print "(Hash)", str(target_automata.algebra.structural_hash())
            conversion_machine, failure_reason = sc.compile_from_algebras(source_automata.algebra, source_automata.automata.component, target_automata.algebra, target_automata.automata.component, options)

            if conversion_machine:
                if options.dump_comparison_cache:
                    wcomparison_cache.add_compiles_to(source_automata.algebra.structural_hash(), target_automata.algebra.structural_hash())
                if DEBUG_COMPUTE_COMPAT_MATRIX:
                    print "Successfully found a conversion between ", i2, j2
                successful_compiles.append(CompilationIndex(i2, j2, conversion_machine, conversion_machine.modifications))
                compilation_list.append((i2, j2, CompilationIndex(i, j, conversion_machine, conversion_machine.modifications)))

    return i, j, successful_compiles, compilation_list


# Given a list of CompiledOjectGroup objects,
# compute a set of regular expressions to
# put on hardware, assignments for everything,
# and conversion machines.
def compute_hardware_assignments_for(groups, options):
    if options.memory_debug:
        print "Memory Usage before cross compat matrix"
        h = hpy()
        print(h.heap())

    if options.use_cross_compilation:
        compiles_to, compiles_from = compute_cross_compatibility_matrix_for(groups, options)
    else:
        # Do not do cross compilation, so generate a compiles to/from list that just lets everything compile to itself.
        compiles_to = []
        compiles_from = []
        for i in range(len(groups)):
            compiles_to.append([])
            compiles_from.append([])

            for j in range(len(groups[i])):
                compiles_to[i].append([])
                compiles_from[i].append([])

    if options.memory_debug:
        print "Memory Usage after cross compat matrix"
        h = hpy()
        print(h.heap())

    print "Generated cross-compilation list"
    result =  assign_hardware(compiles_from, compiles_to, options)
    return result

def assign_hardware(compiles_from, compiles_to, options):
    # Greedy algorithm is to find the regex with the
    # most coverage, and use that to 
    # We could make this a lot faster with a heap or something.
    
    # Keep a list of indexes into the groups that indicate
    # which ones we intend to put into hardware.
    assigned_hardware = [None] * len(compiles_from)
    for i in range(len(compiles_from)):
        assigned_hardware[i] = [None] * len(compiles_from[i])

    while True:
        # Get the index of the most used compilation too.
        max_compiles = 0
        index = None
        # Greedy step 1: pick the automata with the greatest
        # coverage.
        unassigned_hardware = False
        for i in range(len(compiles_from)):
            for j in range(len(compiles_from[i])):
                if assigned_hardware[i][j] is not None:
                    continue

                unassigned_hardware = True
                compiles = len(compiles_from[i][j])
                if compiles >= max_compiles:
                    max_compiles = compiles
                    index = (i, j)

        if DEBUG_COMPUTE_HARDWARE:
            print "Next Interation:"
            print "Found index with most overlap:"
            print index

        # All hardware is compiled to, return it.
        if not unassigned_hardware:
            return assigned_hardware

        # Assign this peice of hardware to itself:
        hardware_i, hardware_j = index
        assigned_hardware[hardware_i][hardware_j] = CompilationIndex(hardware_i, hardware_j, None, unifier.Modifications([], []))

        # Now, we (a) put that in hardware, we need to
        # assign other automata that are part of different
        # groups to it.  Greedy step 2: pick the automata
        # with the fewest other options first.
        # We also try and pick the automata with the fewest
        # number of structural modifications.
        # Cap the total number of structural modifications ---
        # Automata undergoing a huge number of structural
        # modifications just causes problems with the reunification
        # tasks.
        modifications_required = 0
        for i in range(len(compiles_to)):
            min_assigns = 100000000
            min_num_modifications = 10000000000
            min_assigns_index = None
            min_assings_object = None
            # find the min and min index.
            for j in range(len(compiles_to[i])):
                if assigned_hardware[i][j] is not None:
                    # this has already been assigned a thing,
                    # so doesnt need another one.
                    continue
                num_options = len(compiles_to[i][j])
                # Check if this is a match:
                is_match = False
                match_obj = None
                for option in compiles_to[i][j]:
                    if option.i == index[0] and option.j == index[1]:
                        if DEBUG_COMPUTE_HARDWARE:
                            print "Match Found"
                        is_match = True
                        match_obj = option
                        num_modifications = len(option.modifications)
                if is_match and num_options < min_assigns and num_modifications <= min_num_modifications:
                    min_num_mofications = num_modifications
                    min_assigns = num_options
                    min_assigns_index = j
                    min_assigns_object = match_obj

            # Now, create that particular assignment for group
            # j.  Can only assign one per group.
            # Could make this heuristic a bit better by deleting
            # entries from this list.
            if min_assigns_index is not None and MODIFICATIONS_LIMIT > modifications_required + num_modifications:
                modifications_required += num_modifications

                assigned_hardware[i][min_assigns_index] = CompilationIndex(index[0], index[1], min_assigns_object.conversion_machine, min_assigns_object.modifications)

    return assigned_hardware


# Given a set of groups, generate the set of automata needed ---
# using structural modification where required.
# In addition to returning the automata for structural assignments,
# it returns a hash table lookup for all the successful compilations.
def generate_base_automata_for(groups, assignments, options):
    result_automata_group = []
    structural_additions = []
    automata_mapping = {}

    for i in range(len(groups)):
        for j in range(len(groups[i])):
            # See if this is assigned to itself:
            assignment = assignments[i][j]
            if assignment.i == i and assignment.j == j:
                # This is assigned to itself --- need to
                # add it to the group of automata
                result_automata_group.append(groups[i][j].automata)
                # Add the empty structural additions --- these
                # are added in the next loop.
                structural_additions.append([])
                automata_mapping[(i, j)] = len(result_automata_group) - 1

    # Now, generate the mapping: (and include any structural
    # changes required.
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            # The automata in position i, j maps to the automata
            # in position target.i, target.j
            target = assignments[i][j]
            mapping = automata_mapping[(target.i, target.j)]
            automata_mapping[(i, j)] = mapping
            
            # Add the required structural mappings:
            structural_additions[mapping].append(assignments[i][j].modifications)

    # Now, generate the structurally changed automata:
    result = []
    for i in range(len(result_automata_group)):
        automata = result_automata_group[i]

        if DEBUG_GENERATE_BASE:
            print "Pre modification", sc.compute_depth_equation(automata.component, options)
            print "Modifications are:"
            mod_count = 0
            for addition in structural_additions[i]:
                for mod in addition.all_modifications():
                    mod_count += 1
                    print mod.algebra
            print "Making ", mod_count, "modifications"

        result.append(AutomataComponentWrapper(alg.apply_structural_transformations(automata.component, structural_additions[i], options)))
        if DEBUG_GENERATE_BASE:
            print "Post-mofication", sc.compute_depth_equation(result[-1].component, options)

    return result, automata_mapping


def wrap_automata(automata_components, options):
    for i in range(len(automata_components)):
        for j in range(len(automata_components[i])):
            automata_components[i][j] = AutomataComponentWrapper(automata_components[i][j])

    if options.no_groups:
        # Flatten the automata compoenents into a single list of componenets.
        new_acs = []
        for group in automata_components:
            for elt in group:
                new_acs.append([elt])
        automata_components = new_acs

    return automata_components


# if we want to use the comparison cache from a file, then load
# it in.  likewise, if we want to dump the comparison cache,
# then create the dump comparison cache.
def get_comparison_caches(options):
    if options.comparison_cache:
        read_comparison_cache = comparisoncache(options.target)
        read_comparison_cache.from_file(options.comparison_cache)
    else:
        read_comparison_cache = None

    if options.dump_comparison_cache:
        dump_comparison_cache = comparisoncache(options.target)
    else:
        dump_comparison_cache = None

    return read_comparison_cache, dump_comparison_cache

def remove_prefixes(addition_components, group_components, options):
    all_prefix_machines = []
    remaining_components = []
    used_accelerators = set()
    prefix_reduced_machine_indexes = set()

    # Because we are using (translation-free) prefix machines
    # here, we can compile all components to the same prefix
    # machines.
    for comp_index in range(len(addition_components)):
        component = addition_components[comp_index]
        removed_prefix = True
        found_prefix = None
        found_prefix_i = None
        found_prefix_j = None
        found_tail_component = None
        found_tail_group_component = None
        last_prefix_size = -1

        prefix_machines = []
        # Keep removing prefixes as long as there is a match.
        while removed_prefix:
            removed_prefix = False

            # Find the biggest prefix remaining.  There should still
            # be a threshold to the prefix size applied.
            for i in range(len(group_components)):
                for j in range(len(group_components[i])):
                    shared_prefix, tail_first, tail_second = alg.prefix_merge(component.algebra, component.symbol_lookup, group_components[i][j].algebra, group_components[i][j].symbol_lookup, options)

                    # We are looking for more than just a splitting of
                    # two automata here, which is what we are looking for
                    # in shared_prefixes.  We are looking for automata
                    # where the conversion uses the entireity of the automata in
                    # the group_components

                    if tail_second is not None:
                        continue # second component didn't get used
                    # up completely.

                    if shared_prefix is not None and shared_prefix.size() > options.prefix_size_threshold and shared_prefix.size() > last_prefix_size:
                        last_prefix_size = shared_prefix.size()
                        found_prefix = shared_prefix
                        found_prefix_i = i
                        found_prefix_j = j
                        found_tail_component = tail_first
                        found_tail_group_component = tail_second

            if found_prefix is not None:
                removed_prefix = True
                prefix_reduced_machine_indexes.add(comp_index)

                # Add the prefix that we found from the matching
                # automata.
                prefix_machines.append(group_components[found_prefix_i][found_prefix_j])
                # And note that the accelerator is in use.
                used_accelerators.add((found_prefix_i, found_prefix_j))

                # And also update the component we are trying to match.
                if found_tail_group_component is None:
                    # This prefix ate up the whole regex match.
                    # There is no more component to deal with :)
                    component = None
                    break
                else:
                    # There is still more, update the component.
                    component = AutomataContainer(
                            alg.full_graph_for(found_tail_component, component.symbol_lookup),
                            found_tail_component)
        #endwhile
        all_prefix_machines.append(prefix_machines)
        if component is not None:
            # Component will be None if we got an entire-component
            # prefix match --- in that case, no further work is
            # needed on it, so we don't append.
            remaining_components.append(component)

    return all_prefix_machines, remaining_components, used_accelerators, prefix_reduced_machine_indexes


def find_match_for_addition(components, group_components, used_group_components, prefix_reduced_machine_indexes, options):
    conversions = []
    # Keep track of the conversion machines for each component.
    for comp_index in range(len(components)):
        component = components[comp_index]
        conversions.append([])

        for i in range(len(group_components)):
            for j in range(len(group_components[i])):
                if (i, j) in used_group_components:
                    # We can't use this to target any accelerator, because it's being used by a prefix
                    # merge.
                    continue
                target = group_components[i][j]

                conversion_machine, failure_reason = sc.compile_from_algebras(component.algebra, component.automata.component, target.algebra, target.automata.component, options)

                if conversion_machine is not None:
                    conversions[comp_index].append((i, j, target, conversion_machine))

    # Now, go through and pick the conversion machine for each
    # component, picking from the smallest component first.
    # We could sort this etc, but I don't expec the components
    # list to ever be very long, so we can just go through it
    # every time.
    assignment_count = 0
    targets = [None] * len(components)
    conversion_machines = [None] * len(components)
    assigned_accelerators = set()

    # Use the same two-phase greedy algorithm used in the compression
    # routines, but it is a bit simpler because it is a one-way
    # process (i.e. you can only compile from the accelerators
    # that are being added in this
    while assignment_count < len(components):
        min_conversions = 100000000
        min_index = None
        # Find index with fewest conversions
        for i in range(len(components)):
            if len(conversions[i]) == 0:
                # There are no conversion opportunties for
                # this automata, so we can't do this.
                return None, None

            if len(conversions[i]) < min_conversions:
                min_conversions = len(conversions[i])
                min_index = i

        # Now, go through and try and find an assignment
        # in each assigned list.
        found_assignment = False
        for (i, j, target, conversion_machine) in conversions[min_index]:
            if (i, j) not in assigned_accelerators:
                found_assignment = True

                targets[min_index] = target
                conversion_machines[min_index] = conversion_machine

                assigned_accelerators.add((i, j))

        if not found_assignment:
            if options.use_prefix_estimation and min_index in prefix_reduced_machine_indexes:
                # Didn't find anything, but that doesn't matter, because we had a prefix extracted from this
                # regex, so we can still use it to eliminate a large
                # number of prospective packets.
                targets[min_index] = None
                conversion_machines[min_index] = None
            else:
                # Failed because all the potential hardware
                # assignments for that particular accelerator
                # are already in use.
                return None, None

    return targets, conversion_machines


def build_cc_list(targets, conversion_machines, prefix_machines, options):
    if conversion_machines is None:
        return None
    assert len(targets) == len(conversion_machines)

    # Create the conversion machines that
    cc_list = []
    for i in range(len(targets)):
        target = targets[i]
        conversion_machine = conversion_machines[i]
        if target is None and conversion_machines is None:
            # This is a prefix reduced automata for which we
            # have a partial match.
            continue

        cc_group = CCGroup(target.automata, target.algebra)
        cc_group.add_automata(postfix_component.automata, postfix_component.algebra, conversion_machine)
        cc_list.append(cc_group)

    if options.use_prefix_splitting:
        # Also need to return the null translators for the new
        # algebra.  These can all be empty, because we know
        # that these are exact prefixes.  Note that there is
        # a lot more potential here for /inexact/ prefixes.
        for prefix_machine_set in prefix_machines:
            for prefix_machine in prefix_machine_set:
                resmachine = CCGroup(prefix_machine.automata, prefix_machine.algebra)
                # We can use the empty conversion here -- the algebras
                # will be the same.  May need slight modification if
                # we end up doing some input stream translation for
                # each component.
                resmachine.add_automata(prefix_machine.automata, prefix_machine.algebra, FST.EmptySingleStateTranslator())
                cc_list.append(resmachine)

    return cc_list


def find_conversions_for_additions(addition_components, existing_components, options):
    all_conv_machines = []
    # Keep track of whether we have seen a partial match from
    # prefix matching, which still allows for partial pattern
    # recognition.
    has_partial_match = [False] * len(addition_components)
    for i in range(len(addition_components)):
        prefix_machines = None
        used_existing_components = set()
        if options.use_prefix_splitting:
            # We do need to split the incoming regexp into any prefix components
            # that might match the prefix-merged lists however.  That
            # will mean we have to unify a few smaller chunks rather than
            # one large one, so we make it it's own group (recall we
            # are assuming independence from any of the underlying regexps) --- any non-independent regexps should have been removed
            # already.

            # This is not an exhaustively correct approach, but it does
            # allow for much better coverage.  Approach
            # is to split off the longest prefix with an exact match,
            # then try again.
            prefix_machines, postfix_components, used_existing_components, prefix_reduced_machine_indexes = remove_prefixes(addition_components[i], existing_components, options)
            # Only need to deal with the postfix component, because the
            # underlying component will already have been matched.
            # NOTE: The length of the postfix_components may not
            # be the same as the addition components
            addition_components[i] = postfix_components

            if len(prefix_machines) > 0:
                has_partial_match[i] = True

        # Now, work out the number of compiles from each source
        # component to each dest component, and try to find at
        # least one.
        if not options.prefix_merging_only:
            targets, conversion_machines = find_match_for_addition(addition_components[i], existing_components, used_existing_components, prefix_reduced_machine_indexes, options)
        else:
            if not options.use_prefix_splitting:
                print "To use --prefix-merging-only, you also need to have --use-prefix-splitting"
                assert False

            # However, we still need to fail if any one of these
            # didn't get /any/ prefix extracted.
            # Admittedly, this is a bit shit because it doesn't
            # consider the rate at which the CPU will have
            # to check the others (i.e. how /long/ the prefix is)
            # Anyway.  It gives a good idea.
            targets = []
            conversion_machines = []

            for comp_index in addition_components[i]:
                if comp_index not in prefix_reduced_machine_indexes:
                    targets = None
                    conversion_machines = None

        group_conv_machines = build_cc_list(targets, conversion_machines, prefix_machines, options)

        all_conv_machines.append(group_conv_machines)
    return all_conv_machines


# This function takes a list of existing components, and a list
# of components that we wish to /add/ to the list of existing
# components.
# It tries to add the source componenets to the existing componenets
# list.
def compile_to_existing(addition_components, existing_components, options):
    # Various options don't make much sense when compiling to a single
    # target.
    assert not options.use_structural_change
    assert options.target == 'single-state'
    if options.print_compile_time:
        start_time = time.time()

    # Wrap the automata in the wrappers I use to keep track of
    # metadata.
    addition_components = wrap_automata(addition_components, options)
    existing_components = wrap_automata(existing_components, options)

    # In this case, we have had the opportunity to specify
    # the behaviour of the accelerators that were placed into hardware.
    # We chose previously to prefix merge (and prefix split) them, so do so
    # again here.
    if options.use_prefix_splitting:
        if DEBUG_COMPILE_TO_EXISTING:
            print "Assuming that underlying accelerators were prefix-split, splitting..."
            initial_count = sum([len(x) for x in existing_components])
        existing_components = prefix_merge(existing_components, options)

        if DEBUG_COMPILE_TO_EXISTING:
            print "Total introduced prefixes is "
            print sum([len(x) for x in existing_components]) - initial_count


    # Now, we can turn these into algebras :)
    addition_algebras = compile_to_fixed_structures(addition_components, options)
    existing_algebras = compile_to_fixed_structures(existing_components, options)

    if DEBUG_COMPILE_TO_EXISTING:
        print "Addition Algebras are:"
        for group in addition_algebras:
            print "---New Group---"
            for a in group:
                print a.algebra

        print "Existing Algebras are:"
        for group in existing_algebras:
            for e in group:
                print e.algebra


    all_conv_machines = find_conversions_for_additions(addition_algebras, existing_algebras, options)

    if options.print_compile_time:
        total_time = time.time() - start_time
        print "Total taken is:", total_time, "seconds"

    return all_conv_machines


# This method takes a set of automata components and tries
# to compress them, i.e. generate the smallest number of automata
# that can perform the same function given the inter-group
# constraints.
def compile(automata_components, options):
    assert not options.prefix_merging_only # This is an option only
    # when compiling to existing -- could easily be made an option
    # for this though.
    if options.print_compile_time:
        start_time = time.time()

    automata_components = wrap_automata(automata_components, options)

    # Compilation does not support 'true' prefix merging right now,
    # it supports its closely related cousin, prefix splitting
    # however.
    assert not options.use_prefix_merging
    if options.use_prefix_splitting:
        # Do prefix merging on the automata, generate
        # the new prefixes, and the continue on to the
        # rest of the optimizations.
        automata_components = prefix_merge(automata_components, options)

    if options.group_size_distribution:
        group_sizes = []
        for group in automata_components:
            group_sizes.append(len(group))

        print "Dumping group size distributions to file ", options.group_size_distribution
        with open(options.group_size_distribution, 'w') as f:
            f.write(",".join(group_sizes))

    # Do the normal compilation pass:
    #   1: comptue the algebras for each automata.
    #   2: compute all the unifications.
    #   3: compute the hardware assignments.
    #   4: recompute the translators for all the chosen automata.
    # (1)
    groups = compile_to_fixed_structures(automata_components, options)
    if options.compile_only:
        return None

    if options.print_regex_injection_stats:
        print_regex_injection_stats(groups, options)
        return None

    # (2)
    assignments = compute_hardware_assignments_for(groups, options)
    # (3)
    base_automata_components, mapping = generate_base_automata_for(groups, assignments, options)
    
    # If we are using structural modification, then we need to
    # regenerate the groups form the /new/ base_automata_components
    # but /without/ the same old structural modification flags.
    if options.use_structural_change:
        options.use_structural_change = False

    # (4) - regenerate the base automata algebras in case these changed.
    options.use_size_limits = False # Need to disable limits --- the graphs may have grown when they
    # were being structurally modified.
    base_automata_algebras = compile_to_fixed_structures([base_automata_components], options)[0]
    assert len(base_automata_algebras) == len(base_automata_components)
    result = generate_translators(base_automata_algebras, groups, mapping, assignments, options)

    if options.print_successful_conversions:
        for group in result:
            if len(group.supported_algebras) > 1:
                print "Group with physical algebra "
                print group.physical_algebra
                print "Supports algebras:"
                print '\n'.join([str(x) for x in group.supported_algebras])
                print '\n'.join([x.str_with_lookup(generate_fst.edge_label_lookup_generate(y)) for (x, y) in zip(group.supported_algebras, group.supported_automata)])
                print "Translators are:"
                print '\n'.join([str(t) for t in group.translators])
                

    if options.print_compile_time:
        total_time = time.time() - start_time
        print "Total taken is:", total_time, "seconds"

    return result

# Given a list of accelerators that are going to be implemented,
# and a large group of automata, and a mapping on which automata
# to try on the list, compute the translators that these automata
# are going to use.  The original assignments list is taken as
# a debug crutch.
def generate_translators(base_accelerators, groups, mapping, assignments, options):
    translators = []
    for accelerator in base_accelerators:
        translators.append(CCGroup(accelerator.automata, accelerator.algebra))

    for i in range(len(groups)):
        for j in range(len(groups[i])):
            # See what i, j compiles to:
            target_accel_index = mapping[(i, j)]
            target = base_accelerators[target_accel_index]
            source = groups[i][j]
            
            # Now, generate the unifier for this compilation:
            conversion_machine, failure_reason = sc.compile_from_algebras(source.algebra, source.automata.component, target.algebra, target.automata.component, options)
            # I think that this is going to have to succeed.
            # There are ways around it, but it suggests that
            # some approximation was used if it fails.
            if conversion_machine is None:
                print "Suprisise! Failed to convert machines"
                # print source.algebra.str_with_lookup(generate_fst.edge_label_lookup_generate(source.automata))
                print "(lookup)"
                print source.algebra
                # print generate_fst.edge_label_lookup_generate(source.automata)
                # print target.algebra.str_with_lookup(generate_fst.edge_label_lookup_generate(target.automata))
                print "(lookup)"
                print target.algebra
                # print generate_fst.edge_label_lookup_generate(target.automata)
                print "The failure reason was", failure_reason.reason
                # These are ommitted by default because they
                # might be really big..
                # print "They have graphs"
                # sgraph = sjss.automata_to_nodes_and_edges(source.automata)
                # dgraph = sjss.automata_to_nodes_and_edges(target.automata)
                print "WHen we were promised to be able to"
                print "Original assignment was from:"
                print source.algebra
                assign = assignments[i][j]
                print "Required ", len(assign.modifications), "modifications"
                ti, tj = assign.i, assign.j
                print groups[ti][tj].algebra
            # assert conversion_machine is not None
            # This can't be the case --- we are just about to
            # create a final hardware assignment, if there are
            # structural additions, they should be dealt with
            # before this function.
            # assert not conversion_machine.has_structural_additions()

            translators[target_accel_index].add_automata(source.automata, source.algebra, conversion_machine)

    return translators


# This runs a compilation pass that regenerates some structures
# for much greater automata coverage.
# It takes the automata_components, rejenerates a new set
# of automata componenets that we think are likely to have
# a lot broader support at the cost of minor modifications.
def recompile_structures(automata_components, options):
    groups = groups_from_components(automata_components, options)
    group_index = 0

# This function does a prefix merging for the automata
# components.  It does this in the accepting path algebra,
# because that was just a bit easier to write.
# After doing prefix merging, it separates out the prefixes
# and the postfixes to enable better cross compilation.
# IT IS NOT INTENDED FOR FULL USE IN THE CURRENT IMPLEMENTATION,
# because it is not correctly hooked in with the backend.
# The backend would ideally generate fake start states
# for the prefix-merged automata, but instead they are
# treated as real start states.
# In other words, the current implementation provides the
# numbers, but not a real implementation of prefix-merged
# automata.
def prefix_merge(automata_components, options):
    # Get the groups:
    groups = compile_to_fixed_structures(automata_components, options)
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
            # automata so to avoid prefix merging in loops.
            for i2 in range(i, len(groups)):
                if i2 == i:
                    jrange = range(j + 1, len(groups[i2]))
                else:
                    jrange = range(len(groups[i2]))
                for j2 in jrange:
                    if i2 == i and j2 == j:
                        continue

                    shared_prefix, tail_first, tail_second = alg.prefix_merge(groups[i][j].algebra, groups[i][j].symbol_lookup, groups[i2][j2].algebra, groups[i2][j2].symbol_lookup, options)

                    if shared_prefix is not None and shared_prefix.size() > options.prefix_size_threshold:
                        # Replace the two algebras if the shared
                        # prefix is big enough.
                        shared_prefixes.append((i2, j2, shared_prefix, tail_first, tail_second))

            # Now, compute the prefix merges --- find the biggest
            # possible prefix and merge that.
            # There are other good heuristics we could use here, 
            # like find the prefix with the most possible states.
            merge_size = -1
            merges = []
            prefix_groups_required = set()
            for (i2, j2, prefix, tail_first, tail_second) in shared_prefixes:
                if prefix.size() > merge_size:
                    prefix_groups_required = groups[i][j].automata.other_groups.union(groups[i2][j2].automata.other_groups)
                    merge_size = prefix.size()
                    merges = [(i2, j2, prefix, tail_first, tail_second)]
                # This equality might not be right -- it computes 
                # structural equality, and clearly we are after
                # strict equality.  That said, I think they are
                # the same here, because we know that these are prefixes
                # for this automata, so they must be equal to the existing
                # prefix.
                elif prefix.size() == merge_size and prefix.equals(merges[0][2]):
                    merges.append((i2, j2, prefix, tail_first, tail_second))
                    prefix_groups_required = prefix_groups_required.union(groups[i2][j2].automata.other_groups)
                else:
                    continue
            # Execute the merges
            group1_set = False
            old_symbol_lookup = groups[i][j].symbol_lookup
            for (i2, j2, prefix, tail_first, tail_second) in merges:
                prefix_groups_required.add(i2)
                if not group1_set:
                    if tail_first is None:
                        automata_to_remove.add((i, j))
                    else:
                        new_graph = sjss.nodes_and_edges_to_automata(alg.full_graph_for(tail_first, groups[i][j].symbol_lookup))
                        wrapper = AutomataComponentWrapper(new_graph)
                        wrapper.other_groups = groups[i][j].automata.other_groups

                        groups[i][j] = AutomataContainer(
                                wrapper,
                                sc.compute_depth_equation(new_graph, options)
                            )

                    # Only set the source group the first time.
                    group1_set = True

                if tail_second is None:
                    automata_to_remove.add((i2, j2))
                else:
                    new_graph = sjss.nodes_and_edges_to_automata(alg.full_graph_for(tail_second, groups[i2][j2].symbol_lookup))
                    wrapper = AutomataComponentWrapper(new_graph)
                    wrapper.other_groups = groups[i2][j2].automata.other_groups

                    groups[i2][j2] = AutomataContainer(
                            wrapper,
                            sc.compute_depth_equation(new_graph, options)
                        )
            # The prefixes_to_add keeps track of what needs to be added to the automata list.
            # This makes sure that we don't add things to multiple
            # lists.
            if len(merges) > 0:
                prefixes_to_add.append((i, j, prefix, prefix_groups_required, old_symbol_lookup))

    # Now, go back to a list of CCs because that's what we said we would
    # return.
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            groups[i][j] = groups[i][j].automata

    # Now add the prefixes to the groups that need it.
    for i, j, prefix, other_groups, old_symbol_lookup in prefixes_to_add:
        new_graph = sjss.nodes_and_edges_to_automata(alg.full_graph_for(prefix, old_symbol_lookup))
        wrapper = AutomataComponentWrapper(new_graph)
        wrapper.other_groups = other_groups
        groups[i].append(wrapper)

    automata_to_remove = sorted(list(automata_to_remove))[::-1]
    for (i, j) in automata_to_remove:
        del groups[i][j]

    return groups


def compile_to_fixed_structures(automata_components, options):
    # Takes a list of lists of CCs, and computes a
    # set of CCs that should go in hardware, and a list
    # that can be translated.

    # Generate the group.
    groups = groups_from_components(automata_components, options)

    if options.memory_debug:
        print "Memory Usage before computing depth equations"
        h = hpy()
        print(h.heap())

    if options.memory_debug:
        print "Memory Usage after computing depth equations"
        h = hpy()
        print(h.heap())

    return groups


# Store algebra computations between calls to groups_from_components
algebra_cache = {}

def groups_from_components(automata_components, options):
    groups = []
    group_index = 0
    for cc_list in automata_components:
        group = []
        equation_index = 0
        for cc in cc_list:
            if options.print_file_info:
                print "Compiling equation from group ", group_index
                print "Equation index", equation_index

            global algebra_cache
            graph_hash = sjss.hash_graph(cc.component)
            if options.use_algebra_cache and graph_hash in algebra_cache:
                depth_eqn = algebra_cache[graph_hash]

                if depth_eqn is not None:
                    depth_eqn = depth_eqn.clone()
            else:
                depth_eqn = sc.compute_depth_equation(cc.component, options)
                if depth_eqn is None:
                    algebra_cache[graph_hash] = None
                else:
                    algebra_cache[graph_hash] = depth_eqn.clone()

            simple_graph = sjss.automata_to_nodes_and_edges(cc.component).edges

            if not depth_eqn:
                # Means that the graph was too big for the current
                # setup.
                continue

            edges_not_in_graph = False
            for edge in depth_eqn.all_edges():
                if edge not in simple_graph:
                    edges_not_in_graph = True
                    print "Edge", edge, "not in graph"
            if edges_not_in_graph:
                print "Graph", simple_graph
                print "Equation", depth_eqn
                assert False

            if options.print_algebras:
                print depth_eqn
                print "Hash: ", depth_eqn.structural_hash()
            if options.algebra_size_threshold and depth_eqn.size() > options.algebra_size_threshold:
                print "Omitting equation due to size"
            else:
                group.append(AutomataContainer(cc, depth_eqn))
                equation_index += 1

        groups.append(group)
        group_index += 1

    return groups


def print_regex_injection_stats(groups, options):
    compiles_from, compiles_to = compute_cross_compatibility_matrix_for(groups, options)

    non_compiling_opts = 0
    compiling_opts = 0
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            if len(compiles_to[i][j]) == 0:
                non_compiling_opts += 1
            else:
                compiling_opts += 1

    print "Number of regexes we can compile to existing regexes: ", compiling_opts
    print "Number of regexes we can't compile to existing regexes", non_compiling_opts
