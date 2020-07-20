import unittest
import automata.FST.algebra as alg


class AlgebraTest(unittest.TestCase):
    def test_simpleTest(self):
        simple = alg.generate([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)], 0, [3])
        self.assertEquals("3 + a + e", str(simple))

    def test_branch(self):
        simple_branches = alg.generate([0, 1, 2], [(0, 1), (0, 2)], 0, [1, 2])
        self.assertEquals("{1 + a + e, 1 + a + e}", str(simple_branches))

        delayed_branches = alg.generate([0, 1, 2, 3], [(0, 1), (1, 2), (1, 3)], 0, [2])
        self.assertEquals("1 + {1 + a + e, 1 + e}", str(delayed_branches))

    def test_complex_branch(self):
        branches = alg.generate([0, 1, 2, 3, 4, 5], [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)], 0, [2, 3, 5])
        print(branches)

    def test_generate_linear_algebras(self):
        simple = str(alg.linear_algebra_for([0, 1, 2], [2]).normalize())
        self.assertEquals("2 + a", simple)


if __name__ == "__main__":
    unittest.main()
