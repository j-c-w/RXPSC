import unittest

from automata.FST.options import EmptyOptions
import automata.FST.passes.splitter as splitter
from automata.FST.terms import *

class HashSplitter(unittest.TestCase):
    # I really don't know what the automated checks should be here --- I think ideally
    # we'll provide simulation information for a number of these, and we can check
    # that the simulations line up.
    def test_split(self):
        # Make the test generation a bit easier.
        EmptyOptions.split_threshold_frequency = 0
        EmptyOptions.split_size_threshold = 1
        t1 = Sum([Const(3, [1, 2, 3]), End()]).normalize()
        t2 = Branch([Sum([Const(2, [4, 5]), End()]), Sum([Const(3, [6, 7, 8]), End()])]).normalize()

        hash_table = splitter.hash_counts([], [t1, t2], EmptyOptions)
        results = splitter.hash_split([], t2, hash_table, EmptyOptions)
        print [str(x) for x in results]
        results = splitter.hash_split([], t1, hash_table, EmptyOptions)
        print hash_table
        print [str(x) for x in results]

    def test_split_branch(self):
        t1 = Sum([Const(3, [1, 2, 3]), End()]).normalize()
        t2 = Branch([Sum([Const(2, [4, 5]), Branch([Sum([Const(2, [5, 6]), End()]), Sum([Const(3, [7, 8, 9]), End()])])]), Sum([Const(3, [6, 7, 8]), End()])]).normalize()

        hash_table = splitter.hash_counts([], [t1, t2], EmptyOptions)
        results = splitter.hash_split([], t2, hash_table, EmptyOptions)
        print t2
        print [str(x) for x in results]
        self.assertTrue(sum([x.size() for x in results]) < t2.size())
        results = splitter.hash_split([], t1, hash_table, EmptyOptions)
        print hash_table
        print t1
        print [str(x) for x in results]

        self.assertTrue(< t1.size())



if __name__ == "__main__":
    unittest.main()
