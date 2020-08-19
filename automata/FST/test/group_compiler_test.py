import unittest
from automata.FST.group_compiler import *

class GroupCompilerTest(unittest.TestCase):
    def test_compute_hardware_for(self):
        compiles_from = [
                [[], [], []],
                [[], [], []],
                [[], [], [CompilationIndex(0, 0, None)]]
                ]
        compiles_to = [
                [[CompilationIndex(2, 2, None)], [], []],
                [[], [], []],
                [[], [], []],
                ]

        assignments = compute_cross_hardware(compiles_from, compiles_to)
        print assignments

if __name__ == "__main__":
    unittest.main()
