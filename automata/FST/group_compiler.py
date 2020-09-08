import single_compiler as sc
from guppy import hpy
import compilation_statistics
from multiprocessing import Pool
from memory_profiler import profile
import tqdm

DEBUG_COMPUTE_HARDWARE = False
DEBUG_COMPUTE_COMPAT_MATRIX = True

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
    def __init__(self, i2, j2, conversion):
        self.i = i2
        self.j = j2
        self.conversion_machine = conversion

class AutomataContainer(object):
    def __init__(self, automata, algebra):
        self.automata = automata
        self.algebra = algebra

# Given a group, compute a 3D array representing the cross
# compilability of that matrix.
def compute_cross_compatibility_matrix_for(group, options):
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
            tasks.append((group, i, j, options))


    # Compute all the results:
    flat_results = []
    if options.cross_compilation_threading == 0:
        print "Executing single threaded"
        progress = tqdm.tqdm(total=(len(tasks)))
        # Compute traditionally:
        for (group_ref, i, j, options_ref) in tasks:
            if options.memory_debug:
                print "Memory Usage before during cross compat"
                h = hpy()
                print(h.heap())
            flat_results.append(compute_compiles_for((group_ref, i, j, options_ref)))
            progress.update(1)
        progress.close()
    else:
        pool = Pool(options.cross_compilation_threading)
        flat_results = [None] * len(tasks)
        index = 0
        for i, j, res, compilation_list_res in tqdm.tqdm(pool.imap_unordered(compute_compiles_for, tasks), total=len(tasks)):
            flat_results[index] = (i, j, res, compilation_list_res)
            index += 1

    # Now, expand the flat results back out:
    for i in range(len(group)):
        results[i] = [0] * len(group[i])

    for i, j, res, compilation_list_results in flat_results:
        results[i][j] = res

        # And also get the compilation_list setup:
        for i2, j2, comp_item in compilation_list_results:
            compilation_list[i2][j2].append(comp_item)

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
    group, i, j, options = args
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

        for j2 in range(len(group[i2])):
            if DEBUG_COMPUTE_COMPAT_MATRIX:
                print "Pair is ", i2, j2
                print "Comparied to ", i, j
            target_automata = group[i2][j2]

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
            conversion_machine = sc.compile_from_algebras(source_automata.algebra, source_automata.automata, target_automata.algebra, target_automata.automata, options)

            if conversion_machine:
                if DEBUG_COMPUTE_COMPAT_MATRIX:
                    print "Successfully found a conversion between ", i2, j2
                # Do not store the conversion machines --- recompte
                # the ones we pick.
                conversion_machine = None
                successful_compiles.append(CompilationIndex(i2, j2, conversion_machine))
                compilation_list.append((i2, j2, CompilationIndex(i, j, conversion_machine)))

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

    compiles_from, compiles_to = compute_cross_compatibility_matrix_for(groups, options)
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
        assigned_hardware[hardware_i][hardware_j] = CompilationIndex(hardware_i, hardware_j, None)

        # Now, we (a) put that in hardware, we need to
        # assign other automata that are part of different
        # groups to it.  Greedy step 2: pick the automata
        # with the fewest other options first.
        for i in range(len(compiles_to)):
            min_assigns = 100000000
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
                if is_match and num_options < min_assigns:
                    min_assigns = num_options
                    min_assigns_index = j
                    min_assigns_object = match_obj

            # Now, create that particular assignment for group
            # j.  Can only assign one per group.
            # Could make this heuristic a bit better by deleting
            # entries from this list.
            if min_assigns_index is not None:
                assigned_hardware[i][min_assigns_index] = CompilationIndex(index[0], index[1], min_assigns_object.conversion_machine)

    return assigned_hardware

def compile(automata_components, options):
    # Takes a list of lists of CCs, and computes a
    # set of CCs that should go in hardware, and a list
    # that can be translated.
    
    # Generate the group.
    groups = []
    group_index = 0
    if options.group_size_distribution:
        group_sizes = []

    if options.memory_debug:
        print "Memory Usage before computing depth equations"
        h = hpy()
        print(h.heap())

    for cc_list in automata_components:
        group = []
        equation_index = 0
        for cc in cc_list:
            if options.print_file_info:
                print "Compiling equation from group ", group_index
                print "Equation index", equation_index
            depth_eqn = sc.compute_depth_equation(cc, options)
            if not depth_eqn:
                # Means that the graph was too big for the current
                # setup.
                continue
            if options.print_algebras:
                print depth_eqn
                print "Hash: ", depth_eqn.structural_hash()
            if options.algebra_size_threshold and depth_eqn.size() > options.algebra_size_threshold:
                print "Omitting equation due to size"
            else:
                group.append(AutomataContainer(cc, depth_eqn))
                equation_index += 1

        if options.group_size_distribution:
            group_sizes.append(str(len(group)))

        groups.append(group)
        group_index += 1

    if options.memory_debug:
        print "Memory Usage after computing depth equations"
        h = hpy()
        print(h.heap())

    if options.group_size_distribution:
        print "Dumping group size distributions to file ", options.group_size_distribution
        with open(options.group_size_distribution, 'w') as f:
            f.write(",".join(group_sizes))

    return groups, compute_hardware_assignments_for(groups, options)


# Estimate the amount of cross-compatability within
# a suite of automata.
def compile_statistics(connected_components, options):
    algebras = [None] * len(connected_components)

    if options.memory_debug:
        print "Memory Usage before computing depth equations"
        h = hpy()
        print(h.heap())

    print "Starting Compilation"
    for i in range(len(connected_components)):
        algebras[i] = sc.compute_depth_equation(connected_components[i], options, dump_output=True)

    if options.memory_debug:
        print "Memory usage after computed depth equations"
        h = hpy()
        print(h.heap())

    print "Computed Algebras: checking for cross-compatability now"

    # Now, check for cross compatability:
    cross_compilations = 0
    exact_duplicates = 0
    for j in range(len(algebras)):
        for i in range(len(algebras)):
            if i != j:
                result = sc.compare(algebras[i], algebras[j])
                if result:
                    print "Compiled ", algebras[i], " to ", algebras[j]
                    cross_compilations += 1

    print "Total cross compilations is ", cross_compilations
    print "Total successes is ", compile_statistics.single_state_unification_success
    print "Of those, there were ", compile_statistics.exact_same_compilations, " exact duplicates"
