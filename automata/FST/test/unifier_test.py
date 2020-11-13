import unittest
from automata.FST.options import EmptyOptions
import automata.FST.unifier as unifier
import automata.FST.algebra as alg
from automata.FST.terms import *

simple_unifier = unifier.Unifier()
simple_unifier.add_edges([(0, 1), (1, 2)], [(0, 1), (1, 2)], EmptyOptions)

class UnifierTest(unittest.TestCase):
    def test_branch_addition(self):
        l1 = {
                (0, 1): [10],
                (1, 2): [11],
                (2, 3): [12]
        }
        l2 = {
                (0, 1): [14],
                (1, 2): [15],
        }
        u = alg.leq_unify(Sum([Const(2, [(0, 1), (1, 2)]), Accept(), End()]).normalize(), Sum([Const(3, [(0, 1), (1, 2), (2, 3)]), Accept(), End()]).normalize(), EmptyOptions)
        self.assertEqual(len(u), 1)
        generated = u[0].unify_single_state(l1, l2, EmptyOptions)
        print generated.modifications
        self.assertEqual(generated.modifications.all_modifications()[0].edges_after, [(1, 2)])

    def test_loop_addition(self):
        l1 = {
                (0, 1): [65],
                (1, 2): [66],
                (2, 3): [67]
        }
        l2 = {
                (0, 1): [68],
                (1, 1): [69],
                (1, 2): [69],
                (2, 3): [70]
        }
        u = alg.leq_unify(Sum([Const(1, [(0, 1)]), Product(Const(1, [(1, 1)])), Const(2, [(1, 2), (2, 3)]), Accept(), End()]).normalize(), Sum([Const(3, [(0, 1), (1, 2), (2, 3)]), Accept(), End()]).normalize(), EmptyOptions)
        self.assertEqual(len(u), 1)
        generated = u[0].unify_single_state(l2, l1, EmptyOptions)
        self.assertEqual(generated.modifications.all_modifications()[0].edges_after, [(1, 2)])
        print [str(x) for x in generated.modifications.all_modifications()]
        self.assertFalse((1, 2) in generated.lookup)
        self.assertTrue(generated.lookup[65] != 65)
        self.assertTrue(generated.lookup[69] == 65)

    def test_simple_sinlge_state(self):
        lookup_1 = {
                (0, 1): [65],
                (1, 2): [66]
                }
        lookup_2 = {
                (0, 1): [67],
                (1, 2): [68]
                }

        state = simple_unifier.unify_single_state(lookup_1, lookup_2, EmptyOptions)

        self.assertEqual(state[65], 67)


    def test_clashing_unification(self):
        lookup_1 = {
                (0, 1): [65],
                (1, 2): [65]
                }
        lookup_2 = {
                (0, 1): [66],
                (1, 2): [67]
                }

        state = simple_unifier.unify_single_state(lookup_1, lookup_2, EmptyOptions)
        self.assertEqual(state, None)

    def test_none_unification(self):
        lookup_1 = {
                (0, 1): [65],
                (1, 2): [66]
                }

        lookup_2 = {
                (0, 1): [65, 66],
                (1, 2): [67]
                }

        state = simple_unifier.unify_single_state(lookup_1, lookup_2, EmptyOptions)
        self.assertNotEqual(state[65], 67)

if __name__ == "__main__":
    unittest.main()
