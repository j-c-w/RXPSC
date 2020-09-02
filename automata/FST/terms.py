# This file contains a class structure for representing NFA
# depth.  See notebook "Deriving an FST Compilation Algorithm"
# 11 Jul 2020.

TERMS_DEBUG = False
terms_id_counter = 0

class DepthEquation(object):
    def __init__(self):
        global terms_id_counter
        self.id = terms_id_counter
        terms_id_counter += 1
        self._cached_first_edge = None
        self._cached_has_accept = None
        self._cached_has_accept_before_first_edge = None
        self._cached_accepting_distances_approximation = None

    def isproduct(self):
        return False

    def issum(self):
        return False

    def isconst(self):
        return False

    def isbranch(self):
        return False

    def isaccept(self):
        return False

    def isend(self):
        return False

    def has_accept(self):
        if self._cached_has_accept:
            return self._cached_has_accept
        else:
            result = self._has_accept()
            self._cached_has_accept = result
            return result

    def has_accept_before_first_edge(self):
        if self._cached_has_accept_before_first_edge:
            return self._cached_has_accept_before_first_edge
        else:
            result = self._has_accept_before_first_edge()
            self._cached_has_accept_before_first_edge = result
            return result

    def _has_accept(self):
        assert False

    def _has_accept_before_first_edge(self):
        assert False

    def equals(self, other, selflookup=None, otherlookup=None):
        assert False

    # The idea of this is an approximation of the lengths
    # of strings that the regex accepts.  We anticipate
    # that the approximation for the Product case
    # is the most interseting (as that is the only non-trivial
    # case).
    def accepting_distances_approximation(self):
        if self._cached_accepting_distances_approximation is not None:
            return self._cached_accepting_distances_approximation
        else:
            result = self._accepting_distances_approximation()
            self._cached_accepting_distances_approximation = result
            return result

    def _accepting_distances_approximation(self):
        assert False

    # The concept here is to return the N last
    # edges from this term, split from the current
    # item.
    def split_last(self, n):
        assert False

    # The concept here is to return the number
    # of edges that overlap exactly within
    # the two terms.  It's usecase is for deduplication
    # of edges within the accepting path algebra at
    # the end of branches.
    def overlap_distance(self, other):
        assert False

    # Return the first edge from the current term.
    def _first_edge_internal(self):
        assert False

    def first_edge(self):
        if self._cached_first_edge is None:
            self._cached_first_edge = self._first_edge_internal()
        return self._cached_first_edge

    def has_first_edge(self):
        if self.first_edge() is None:
            return False
        else:
            return True

    def structural_hash(self):
        assert False


class Product(DepthEquation):
    def __init__(self, e1):
        super(Product, self).__init__()

        self.e1 = e1
        self.isnormal = False
        self._size = None
        self._last_node = None

    # Approximate this as just the length of the
    # subexpression.
    def _accepting_distances_approximation(self):
        result = self.e1.accepting_distances_approximation()
        result.append(0)
        return result

    def type(self):
        return "Product"

    def _first_edge_internal(self):
        return self.e1.first_edge()

    def _has_accept(self):
        return self.e1.has_accept()

    def _has_accept_before_first_edge(self):
        return self.e1.has_accept_before_first_edge()

    def equals(self, other, selflookup=None, otherlookup=None):
        if other.isproduct():
            return self.e1.equals(other.e1, selflookup, otherlookup)
        else:
            return False

    def overlap_distance(self, other):
        # I think we can do this because the caching
        # algorithm that constructs these trees
        # ensures that any duplicate items are really
        # at the same point in memory.
        if self == other:
            return self.size()
        else:
            return 0

    def split_last(self, n):
        if n > 0:
            # We can't split within a product.
            assert n == self.size()
            return Const(0, []), self
        else:
            return self, Const(0, [])

    def get_last_node(self):
        if self._last_node:
            return self._last_node

        self._last_node = self.e1.get_last_node()
        return self._last_node

    def isproduct(self):
        return True

    def __str__(self):
        return "(" + str(self.e1) + ")*"

    def normalize(self):
        # We do not want to normalize things
        # more than once, becuase that creates
        # mass duplication of algebras, when we
        # can already represent immensely complex
        # algebras with shared pointers.
        if self.isnormal:
            return self

        self.isnormal = True
        self.e1 = self.e1.normalize()

        return self

    def size(self):
        if self._size:
            return self._size
        else:
            self._size = self.e1.size()
            return self._size

    def structural_hash(self):
        return self.e1.structural_hash() * 2


class Sum(DepthEquation):
    def __init__(self, elems):
        super(Sum, self).__init__()

        self.e1 = elems
        self.isnormal = False
        self._size = None
        self._last_node = None

    def _accepting_distances_approximation(self):
        set = self.e1[0].accepting_distances_approximation()
        for elem in self.e1[1:]:
            if len(set) > 100:
                # Abort the approximation if it is too big.
                return set
            accepting_distances = elem.accepting_distances_approximation()
            new_set = []
            for elem in set:
                for elem2 in accepting_distances:
                    new_set.append(elem + elem2)
            set = new_set
        return set

    def type(self):
        return "Sum"

    def equals(self, other, selflookup=None, otherlookup=None):
        if other.issum() and len(self.e1) == len(other.e1):
            for i in range(len(self.e1)):
                if not self.e1[i].equals(other.e1[i], selflookup, otherlookup):
                    return False
            return True
        else:
            return False

    def _has_accept(self):
        for elem in self.e1:
            if self.elem.has_accept():
                return True
        return False

    def _has_accept_before_first_edge(self):
        for elem in self.e1:
            if elem.has_accept_before_first_edge():
                return True
            if elem.first_edge() is not None:
                return False

        return False

    def _first_edge_internal(self):
        # Return the first item in this sum that
        # has an edge.
        for item in self.e1:
            first = item.first_edge()
            if first:
                return first
        return None

    def split_last(self, n):
        i = len(self.e1) - 1
        split_off = 0

        while split_off + self.e1[i].size() <= n:
            split_off += self.e1[i].size()

            i -= 1

        if i == -1:
            return Const(0, []), self
        else:
            if n != split_off:
                # We have to subsplit at item i:
                head, tail = self.e1[i].split_last(n - split_off)
            else:
                head = self.e1[i]
                tail = Const(0, [])

            head_match = Sum(self.e1[:i] + [head]).normalize()
            tail_match = Sum([tail] + self.e1[i + 1:]).normalize()
            return head_match, tail_match

    def overlap_distance(self, other):
        if other.issum():
            tail_overlap_index = 0
            matching = True
            # The concept is to go over every element from the end
            # and see if they are equal -- if they aren't equal,
            # just compute teh tail overlap of the last element.
            # If they are equal, keep going backwards within
            # the sum.
            tail_overlap_count = 0
            while matching and tail_overlap_index < len(self.e1) and tail_overlap_index < len(other.e1):
                this_elem = self.e1[len(self.e1) - tail_overlap_index - 1]
                other_elem = other.e1[len(other.e1) - tail_overlap_index - 1]
                if this_elem == other_elem or \
                    (this_elem.isaccept() and other_elem.isaccept()) or \
                    (this_elem.isend() and other_elem.isend()):
                    tail_overlap_index += 1
                else:
                    matching = False
                tail_overlap_count += this_elem.overlap_distance(other_elem)

            return tail_overlap_count
        else:
            return 0


    def get_last_node(self):
        if self._last_node:
            return self._last_node

        found_last_node = None
        index = len(self.e1) - 1
        while found_last_node is None and index >= 0:
            found_last_node = self.e1[-1].get_last_node()

            index -= 1

        self._last_node = found_last_node
        return self._last_node

    def issum(self):
        return True

    def __str__(self):
        return " + ".join([str(x) for x in self.e1])

    def normalize(self, flatten=True):
        if self.isnormal:
            return self
        self.isnormal = True
        if len(self.e1) == 1:
            return self.e1[0].normalize()
        self.e1 = [x.normalize() for x in self.e1]
        # We don't always have to flatten, e.g. if we know
        # that none of the subelements are sum elements.
        if flatten:
            # Now, flatten everything
            flattened = []
            for e in self.e1:
                if e.issum():
                    flattened += e.e1
                elif e.isconst() and e.val == 0:
                    pass
                else:
                    flattened.append(e)

            self.e1 = flattened
        # This is from when 'normalizing' meant putting everything
        # into as few consts as possible rather than as many as possible.
        # const_sum = 0
        # edges = []
        # new_values = []
        # for e in self.e1:
        #     if e.isconst():
        #         const_sum += e.val
        #         edges += e.edges
        #     else:
        #         if const_sum > 0:
        #             new_values.append(Const(const_sum, edges))
        #         new_values.append(e)
        #         const_sum = 0
        #         edges = []

        # if const_sum > 0:
        #     new_values.append(Const(const_sum, edges))
        #     const_sum = 0
        #     edges = []

        # self.e1 = new_values
        if len(self.e1) == 1:
            return self.e1[0]

        return self

    def size(self):
        if self._size:
            return self._size
        else:
            self._size = sum([x.size() for x in self.e1])
            return self._size

    def structural_hash(self):
        return sum([x.structural_hash() for x in self.e1]) * 3


class Const(DepthEquation):
    def __init__(self, val, edges):
        super(Const, self).__init__()
        assert(val == len(edges))

        self.val = val
        # These are the edges in the automata that this constant
        # represents coverage over.
        self.edges = edges
        self._size = None

    def _accepting_distances_approximation(self):
        # Assume there is an accept at the end...
        return [self.val]

    def type(self):
        return "Const"

    def _has_accept(self):
        return False

    def _has_accept_before_first_edge(self):
        return False

    def _first_edge_internal(self):
        if len(self.edges) > 0:
            return [self.edges[0]]
        else:
            return None

    def equals(self, other, selflookup=None, otherlookup=None):
        if other.isconst() and other.val == self.val:
            for i in range(self.val):
                if otherlookup and selflookup:
                    if otherlookup[other.edges[i]] != selflookup[self.edges[i]]:
                        return False
            return True
        else:
            return False

    def split_last(self, n):
        if n == self.size():
            return Const(0, []), self
        else:
            return Const(self.val - n, self.edges[:self.val - n]), Const(n, self.edges[self.val - n:])

    def overlap_distance(self, other):
        if not other.isconst():
            return 0
        if self == other:
            return self.size()
        tail_count = 0
        matching = True
        while matching and tail_count < len(self.edges) and tail_count < len(other.edges):
            this_elem = self.edges[len(self.edges) - tail_count - 1]
            other_elem = other.edges[len(other.edges) - tail_count - 1]
            if this_elem == other_elem:
                tail_count += 1
            else:
                matching = False
        return tail_count

    def isconst(self):
        return True

    def get_last_node(self):
        # Return the last node of the last edge if it exists
        if len(self.edges) > 0:
            return self.edges[-1][1]
        else:
            return None

    def __str__(self):
        if TERMS_DEBUG:
            return str(self.val) + " " + str(self.edges) + ""
        else:
            return str(self.val)

    def normalize(self):
        if self.val <= 1:
            return self
        else:
            return Sum([Const(1, [self.edges[i]]) for i in range(self.val)])

    def size(self):
        return self.val

    def structural_hash(self):
        return self.val


class Branch(DepthEquation):
    def __init__(self, options):
        super(Branch, self).__init__()

        self.options = [opt for opt in options if opt]
        self.isnormal = False
        self._size = None

    def _accepting_distances_approximation(self):
        results = []
        for opt in self.options:
            results += opt.accepting_distances_approximation()
        return results

    def type(self):
        return "Branch"

    def equals(self, other, selflookup=None, otherlookup=None):
        if not other.isbranch():
            return False
        matches = []
        for i in range(len(self.options)):
            found_match = False
            for j in range(len(other.options)):
                if j in matches:
                    continue
                if self.options[i].equals(other.options[j], selflookup, otherlookup):
                    found_match = True
                    matches.append(j)

            if not found_match:
                return False

        return True

    def _has_accept(self):
        for opt in self.options:
            if opt.has_accept():
                return True
        return False

    def _has_accept_before_first_edge(self):
        for opt in self.options:
            if opt.has_accept_before_first_edge():
                return True
        return False

    def _first_edge_internal(self):
        first_edges = []
        for opt in self.options:
            sub_first_edge = opt.first_edge()
            if sub_first_edge:
                first_edges += sub_first_edge

        if len(first_edges) == 0:
            return None
        else:
            return first_edges

    def overlap_distance(self, other):
        if self == other:
            return self.size()
        else:
            return 0

    def split_last(self, count):
        if count > 0:
            # We can either do all or nothing on a
            # branch.
            assert count == self.size()
            return Const(0, []), self
        else:
            return self, Const(0, [])

    def isbranch(self):
        return True

    def get_last_node(self):
        return None

    def __str__(self):
        return "{" + ", ".join([str(opt) for opt in self.options]) + "}"

    def normalize(self, compress=True):
        if self.isnormal:
            return self
        self.isnormal = True
        if len(self.options) == 1:
            return self.options[0].normalize()
        self.options = [opt.normalize() for opt in self.options if opt]
        # Compression is not required if the sum was constructed
        # with a number of assumptions, e.g. that it has
        # already been partially compressed at some point in the
        # past.
        if compress:
            self.compress()
        if len(self.options) == 1:
            return self.options[0]
        else:
            assert len(self.options) != 1
            return self

    def compress(self):
        # The results of generate_internal duplicate every branch tail.
        # Excessive compilation time etc. are avoided because of the
        # caching system.  However, having the algebra in that
        # form makes comparisons hard.  This function compresses
        # it.  (e.g. {2 + 1, 2 + 1}, where the last "1" represents
        # the same edge, should be pulled out to be
        # {2, 2} + 1.

        # The aim here is to construct backward equvalences
        # that are as large as possible.

        overlap_distances = [None] * len(self.options)
        for i in range(len(self.options)):
            overlap_distances[i] = [None] * len(self.options)
            for j in range(len(self.options)):
                if i == j:
                    continue
                # compute the distance of overlap in
                # reverse.
                overlap_distances[i][j] = self.options[i].overlap_distance(self.options[j])

        # We need to build an additive phylogeny basically.
        # Shouldn't be too hard...
        # We use neightbour joining.
        branch_sets = self.options
        has_compression = True
        while has_compression:
            # Find the two elements with the most overlap
            # and merge them together.
            max_overlap = 0
            max_i = 0
            max_j = []
            for i in range(len(branch_sets)):
                for j in range(len(branch_sets)):
                    if i == j:
                        continue
                    if overlap_distances[i][j] > max_overlap:
                        max_i = i
                        max_j = [j]
                        max_overlap = overlap_distances[i][j]
                    elif overlap_distances[i][j] == max_overlap and max_i == i:
                        max_j.append(j)

            if max_overlap == 0:
                # No more merging to do :'(
                break

            # Now, merge these two.  We shouldn't have
            # to adjust the differences.
            new_i_algebra, shared_tail = branch_sets[max_i].split_last(max_overlap)
            branch_sets[max_i] = [new_i_algebra]
            for j in max_j:
                # Get the bif of each algebra that isn't shared.
                new_j_algebra, shared_tail_j = branch_sets[j].split_last(max_overlap)
                branch_sets[max_i].append(new_j_algebra)

            branch_sets[max_i] = Sum([Branch(branch_sets[max_i]), shared_tail])

            # Now, delete all the 'j' stuff that has been copied
            # across.
            for j in sorted(max_j)[::-1]:
                del branch_sets[j]
                assert j != max_i
                if j < max_i:
                    max_i -= 1
                del overlap_distances[j]

                # Also need to delete the column:
                for i in range(len(overlap_distances)):
                    del overlap_distances[i][j]


        # Finally, set the elements to the ones we computed.
        self.options = branch_sets
        self._size = None


    def size(self):
        if self._size:
            return self._size
        else:
            self._size = sum([opt.size() for opt in self.options])
            return self._size

    def structural_hash(self):
        return sum([x.structural_hash() for x in self.options]) * 5

class Accept(DepthEquation):
    def __init__(self):
        super(Accept, self).__init__()

    def _accepting_distances_approximation(self):
        return [0]

    def split_last(self):
        # Not 100% sure what to do in this case.  Pretty
        # sure this shouldn't get called.
        assert False

    def type(self):
        return "Accept"

    def equals(self, other, selflookup=None, otherlookup=None):
        return other.isaccept()

    def _has_accept(self):
        return True

    def _has_accept_before_first_edge(self):
        return True

    def isaccept(self):
        return True

    def _first_edge_internal(self):
        return None

    # Accept and End are not important things
    # to merge anyway -- the concepts they represent
    # are not data dependent, i.e. it does not
    # matter what state you end in as long as it ends
    # and is an accepting state.
    def overlap_distance(self, other):
        return 0

    def __str__(self):
        return "a"

    def get_last_node(self):
        return None

    def normalize(self):
        return self

    def size(self):
        return 0

    def structural_hash(self):
        return -1

class End(DepthEquation):
    def __init__(self):
        super(End, self).__init__()

    def split_last(self):
        # Not 100% sure what to do in this case.  Pretty
        # sure this shouldn't get called.
        assert False

    def _accepting_distances_approximation(self):
        return [0]

    def equals(self, other, selflookup=None, otherlookup=None):
        return other.isend()

    def _has_accept(self):
        return False

    def _has_accept_before_first_edge(self):
        return False

    def type(self):
        return "End"

    def isend(self):
        return True

    def _first_edge_internal(self):
        return None

    def overlap_distance(self, other):
        return 0

    def get_last_node(self):
        return None

    def __str__(self):
        return "e"

    def normalize(self):
        return self

    def size(self):
        return 0

    def structural_hash(self):
        return -2
