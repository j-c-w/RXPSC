import unittest
import automata.FST.algebra as algebra
from automata.FST.options import EmptyOptions
from automata.FST.terms import *

class ComparisonTest(unittest.TestCase):
    def test_compare(self):
        self.assertTrue(algebra.leq(Const(1, [(0, 1)]), Const(1, [(0, 1)]), EmptyOptions))

    def test_compute_branch(self):
        self.assertTrue(algebra.leq(Branch([Const(3, [(0, 1), (1, 2), (2, 3)]), Accept()]), Branch([Accept(), Const(3, [(3, 4), (4, 5), (5, 6)])]), EmptyOptions))

    def test_compute_sum(self):
        self.assertTrue(algebra.leq(Sum([Const(1, [(0, 1)]), Accept(), End()]), Sum([Const(1, [(0, 1)]), Accept(), End()]), EmptyOptions))

        # Test the trim property.
        self.assertTrue(algebra.leq(Sum([Const(1, [(0, 1)]), Accept(), End()]),
                                    Sum([Const(1, [(0, 1)]), Accept(), Const(2, [(1, 2), (2, 3)]), End(), Accept()]), EmptyOptions))

        self.assertFalse(algebra.leq(Sum([Const(1, [(0, 1)]), End()]),
                                    Sum([Const(1, [(0, 1)]), Accept(), Const(2, [(1, 2), (2, 3)]), End(), Accept()]), EmptyOptions))


if __name__ == "__main__":
    unittest.main()
