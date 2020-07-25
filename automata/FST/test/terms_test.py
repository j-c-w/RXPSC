import unittest
from automata.FST.terms import *

class TermsTest(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(Sum([Branch([End(), End()])]).size(), 0)

    def test_tailcount(self):
        self.assertEqual(Sum([Const(2, [(0, 1), (1, 2)]), End()]).overlap_distance(Sum([Const(1, [(1, 2)]), End()])), 1)

    def test_split_last(self):
        self.assertEqual(str(Sum([Const(2, [(0, 1), (1, 2)]), End()]).split_last(1)[1]), "1 + e")

if __name__ == "__main__":
    unittest.main()
