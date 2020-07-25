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


class Product(DepthEquation):
    def __init__(self, e1):
        super(Product, self).__init__()

        self.e1 = e1
        self.isnormal = False
        self._size = None
        self._last_node = None

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
            self._size = self.e1.size() + 1
            return self._size


class Sum(DepthEquation):
    def __init__(self, elems):
        super(Sum, self).__init__()

        self.e1 = elems
        self.isnormal = False
        self._size = None
        self._last_node = None

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

            head_match = Sum(self.e1[:i] + [head])
            tail_match = Sum([tail] + self.e1[i + 1:])
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

    def normalize(self):
        if self.isnormal:
            return self
        self.isnormal = True
        self.e1 = [x.normalize() for x in self.e1]
        # Now, flatten everything
        flattened = []
        for e in self.e1:
            if e.issum():
                flattened += e.e1
            else:
                flattened.append(e)

        self.e1 = flattened

        const_sum = 0
        edges = []
        new_values = []
        for e in self.e1:
            if e.isconst():
                const_sum += e.val
                edges += e.edges
            else:
                if const_sum > 0:
                    new_values.append(Const(const_sum, edges))
                new_values.append(e)
                const_sum = 0
                edges = []

        if const_sum > 0:
            new_values.append(Const(const_sum, edges))
            const_sum = 0
            edges = []

        self.e1 = new_values
        if len(self.e1) == 1:
            return self.e1[0]

        return self

    def size(self):
        if self._size:
            return self._size
        else:
            self._size = sum([x.size() for x in self.e1])
            return self._size


class Const(DepthEquation):
    def __init__(self, val, edges):
        super(Const, self).__init__()
        assert(val == len(edges))

        self.val = val
        # These are the edges in the automata that this constant
        # represents coverage over.
        self.edges = edges
        self._size = None

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
        return self

    def size(self):
        return self.val


class Branch(DepthEquation):
    def __init__(self, options):
        super(Branch, self).__init__()

        self.options = [opt for opt in options if opt]
        self.isnormal = False
        self._size = None

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

    def normalize(self):
        if self.isnormal:
            return self
        self.isnormal = True
        if len(self.options) == 1:
            return self.options[0].normalize()
        else:
            self.options = [opt.normalize() for opt in self.options if opt]
            self.compress()
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
                print "Joining"

            branch_sets[max_i] = Sum([Branch(branch_sets[max_i]), shared_tail]).normalize()

            # Now, delete all the 'j' stuff that has been copied
            # across.
            for j in sorted(max_j)[::-1]:
                del branch_sets[j]
                assert j != max_i
                if j < max_i:
                    max_i -= 1
                del overlap_distances[max_i][j]

        # Finally, set the elements to the ones we computed.
        self.options = branch_sets


    def size(self):
        if self._size:
            return self._size
        else:
            self._size = sum([opt.size() for opt in self.options])
            return self._size

class Accept(DepthEquation):
    def __init__(self):
        super(Accept, self).__init__()

    def split_last(self):
        # Not 100% sure what to do in this case.  Pretty
        # sure this shouldn't get called.
        assert False

    def isaccept(self):
        return True

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

class End(DepthEquation):
    def __init__(self):
        super(End, self).__init__()

    def split_last(self):
        # Not 100% sure what to do in this case.  Pretty
        # sure this shouldn't get called.
        assert False

    def isend(self):
        return True

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
