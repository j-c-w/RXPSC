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


class Product(DepthEquation):
    def __init__(self, e1):
        super(Product, self).__init__()

        self.e1 = e1
        self.isnormal = False
        self._size = None
        self._last_node = None

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

        if const_sum > 0:
            new_values.append(Const(const_sum, edges))

        self.e1 = new_values
        if len(self.e1) == 1:
            return self.e1[0]

        return self

    def size(self):
        if self._size:
            return self._size
        else:
            self._size = self.e1.size() + self.e2.size()
            return self._size


class Const(DepthEquation):
    def __init__(self, val, edges):
        super(Const, self).__init__()

        self.val = val
        # These are the edges in the automata that this constant
        # represents coverage over.
        self.edges = edges
        self._size = None

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
        return 1


class Branch(DepthEquation):
    def __init__(self, options):
        super(Branch, self).__init__()

        self.options = [opt for opt in options if opt]
        self.isnormal = False
        self._size = None

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
            return Branch([opt.normalize() for opt in self.options if opt])

    def size(self):
        if self._size:
            return self._size
        else:
            self._size = sum([opt.size() for opt in self.options])
            return self._size

class Accept(DepthEquation):
    def __init__(self):
        super(Accept, self).__init__()

    def isaccept(self):
        return True

    def __str__(self):
        return "a"

    def get_last_node(self):
        return None

    def normalize(self):
        return self

    def size(self):
        return 1

class End(DepthEquation):
    def __init__(self):
        super(End, self).__init__()

    def isend(self):
        return True

    def get_last_node(self):
        return None

    def __str__(self):
        return "e"

    def normalize(self):
        return self

    def size(self):
        return 1
