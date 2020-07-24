import unittest
from automata.FST.terms import *

class TermsTest(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(Sum([Branch([End(), End()])]).size(), 2)

if __name__ == "__main__":
    unittest.main()
