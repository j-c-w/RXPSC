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
        self.assertEquals(str(branches), "{1 + e, 1 + a + e, 1 + a + {1 + e, 1 + a + e}}")

    def test_generate_linear_algebras(self):
        simple = str(alg.linear_algebra_for([0, 1, 2], [2]).normalize())
        self.assertEquals("2 + a", simple)

    def test_simple_loop(self):
        loop = alg.generate([0, 1, 2, 3], [(0, 1), (1, 2), (2, 1), (1, 3)], 0, [3])
        self.assertEquals("1 + (2)* + 1 + a + e", str(loop))

    def test_simple_loop_2(self):
        loop = alg.generate([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 1), (2, 3), (3, 2), (1, 4)], 0, [4])
        self.assertEquals("1 + (1 + (2)* + 1)* + 1 + a + e", str(loop))

    def test_mono_loop(self):
        loop = alg.generate([0], [(0, 0)], 0, [0])
        self.assertEqual("(1)*", str(loop))

    # This is an edge case we don't really handle well, but
    # it isn't incorrect -- the algebra just makes the automaton
    # seem more general than it is.  That problem will get
    # fixed upon unification though.
    def test_diverging_triple(self):
        triple = alg.generate([0, 1, 2, 3, 4], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)], 0, [4])
        print str(triple)

    def test_converging_triple(self):
        triple = alg.generate([0, 1, 2, 3, 4], [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 4), (2, 4)], 0, [4])
        print triple

    def test_multiple_removed_edges(self):
        res = alg.generate([0, 1, 2, 3, 4], [(0, 1), (1, 1), (1, 2), (2, 1), (2, 3), (3, 4)], 0, [4])
        print(res)

    def test_multiple_loops(self):
        res = alg.generate([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 1), (1, 3), (3, 1), (1, 4)], 0, [4])

    def test_double_branch(self):
        res = alg.generate([0, 1, 2, 3, 4, 5], [(0, 1), (0, 2), (2, 3), (1, 3), (3, 4), (3, 5)], 0, [])
        self.assertEqual(str(res), "{{2, 2} + {1 + e, 1 + e}}")

    def test_double_branch_2(self):
        res = alg.generate([0, 1, 2, 3, 4, 5, 6], [(0, 1), (0, 2), (2, 3), (1, 3), (3, 4), (4, 5), (4, 6)], 0, [])
        self.assertEqual(str(res), "{{2, 2} + 1 + {1 + e, 1 + e}}")



if __name__ == "__main__":
    unittest.main()
