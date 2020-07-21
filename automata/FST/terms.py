# This file contains a class structure for representing NFA
# depth.  See notebook "Deriving an FST Compilation Algorithm"
# 11 Jul 2020.

TERMS_DEBUG = False

class DepthEquation(object):
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
        self.e1 = e1

    def isproduct(self):
        return True

    def __str__(self):
        return "(" + str(self.e1) + ")*"

    def normalize(self):
        self.e1 = self.e1.normalize()

        return self


class Sum(DepthEquation):
    def __init__(self, e1, e2):
        self.e1 = e1
        self.e2 = e2

    def issum(self):
        return True

    def __str__(self):
        return str(self.e1) + " + " + str(self.e2)

    def normalize(self):
        self.e1 = self.e1.normalize()
        self.e2 = self.e2.normalize()

        if (self.e1.isconst() and self.e2.isconst()):
            return Const(self.e1.val + self.e2.val, self.e1.edges + self.e2.edges)

        if (self.e1.isconst() and self.e1.val == 0):
            return self.e2.normalize()

        if (self.e2.isconst() and self.e2.val == 0):
            return self.e1.normalize()

        return self


class Const(DepthEquation):
    def __init__(self, val, edges):
        self.val = val
        # These are the edges in the automata that this constant
        # represents coverage over.
        self.edges = edges

    def isconst(self):
        return True

    def __str__(self):
        if TERMS_DEBUG:
            return str(self.val) + " " + str(self.edges) + ""
        else:
            return str(self.val)

    def normalize(self):
        return self


class Branch(DepthEquation):
    def __init__(self, options):
        self.options = [opt for opt in options if opt]

    def isbranch(self):
        return True

    def __str__(self):
        return "{" + ", ".join([str(opt) for opt in self.options]) + "}"

    def normalize(self):
        if len(self.options) == 1:
            return self.options[0].normalize()
        else:
            return Branch([opt.normalize() for opt in self.options if opt])

class Accept(DepthEquation):
    def __init__(self):
        pass

    def isaccept(self):
        return True

    def __str__(self):
        return "a"

    def normalize(self):
        return self

class End(DepthEquation):
    def __init__(self):
        pass

    def isend(self):
        return True

    def __str__(self):
        return "e"

    def normalize(self):
        return self
