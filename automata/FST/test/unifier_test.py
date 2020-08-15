import unittest
import automata.FST.unifier as unifier

simple_unifier = unifier.Unifier()
simple_unifier.add_edges([(0, 1), (1, 2)], [(0, 1), (1, 2)])

class UnifierTest(unittest.TestCase):
    def test_simple_sinlge_state(self):
        lookup_1 = {
                (0, 1): ["a"],
                (1, 2): ["b"]
                }
        lookup_2 = {
                (0, 1): ["c"],
                (1, 2): ["d"]
                }

        state = simple_unifier.unify_single_state(lookup_1, lookup_2)

        self.assertEqual(state["a"], "c")


    def test_clashing_unification(self):
        lookup_1 = {
                (0, 1): ["a"],
                (1, 2): ["a"]
                }
        lookup_2 = {
                (0, 1): ["c"],
                (1, 2): ["d"]
                }

        state = simple_unifier.unify_single_state(lookup_1, lookup_2)
        self.assertEqual(state, None)

    def test_none_unification(self):
        lookup_1 = {
                (0, 1): ["a"],
                (1, 2): ["b"]
                }

        lookup_2 = {
                (0, 1): None,
                (1, 2): ["d"]
                }

        state = simple_unifier.unify_single_state(lookup_1, lookup_2)
        print (state["a"])
        self.assertNotEqual(state["a"], "d")

if __name__ == "__main__":
    unittest.main()
