import unittest
from automata.FST.terms import *
from automata.FST.options import EmptyOptions
import automata.FST.unifier as unifier
from automata.FST.unifier import Modifications, Modification
import automata.FST.algebra as alg


class AlgebraTest(unittest.TestCase):
    # It is not 100% clear to me that these should nessecarily
    # be of the format a + (1 + a)*, rather than (a + 1)*.
    # I think that the later requires applying the loop rolling
    # property, but I'm not actually sure.
    def test_trailing_product_2(self):
        algebra = alg.generate([0, 1, 2, 3, 4, 5], [(0, 1), (1, 3), (3, 4), (1, 4), (4, 5), (5, 4)], 0, [4], EmptyOptions)
        self.assertEqual("1 + {2, 1} + a + (2 + a)*", str(algebra))

    def test_trailing_product(self):
        algebra = alg.generate([0, 1, 2, 3, 4], [(0, 1), (1, 3), (3, 4), (1, 4), (4, 4)], 0, [4], EmptyOptions)
        self.assertEqual("1 + {2, 1} + a + (1 + a)*", str(algebra))

    def test_trailing_product_3(self):
        nodes = [4, 5, 6, 7, 0, 8, 11]
        edges =  [(4, 5), (4, 6), (5, 6), (6, 7), (7, 8), (0, 4), (7, 11), (7, 7)]
        accepting = [11]
        start = 0
        algebra = alg.generate(nodes, edges, start, accepting, EmptyOptions)
        print algebra
        self.assertEqual(str(algebra), "1 + {2, 1} + 1 + (1)* + {1 + e, 1 + a + e}")


    def test_simpleTest(self):
        simple = alg.generate([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)], 0, [3], EmptyOptions)
        self.assertEquals("3 + a + e", str(simple))

    # Generated from a failing exmapl
    def test_cross_loop(self):
        nodes = [0, 1, 2, 3, 4, 5]
        edges = [(0, 1), (0, 2), (2, 3), (3, 4), (4, 2), (1, 5), (1, 3), (2, 5)]
        # print alg.generate(nodes,edges, 0, [5])

    def test_branch(self):
        simple_branches = alg.generate([0, 1, 2], [(0, 1), (0, 2)], 0, [1, 2], EmptyOptions)
        self.assertEquals("{1 + a + e, 1 + a + e}", str(simple_branches))

        delayed_branches = alg.generate([0, 1, 2, 3], [(0, 1), (1, 2), (1, 3)], 0, [2], EmptyOptions)
        self.assertEquals("1 + {1 + a + e, 1 + e}", str(delayed_branches))

    def test_complex_branch(self):
        branches = alg.generate([0, 1, 2, 3, 4, 5], [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)], 0, [2, 3, 5], EmptyOptions)
        self.assertEquals(str(branches), "{1 + e, 1 + a + e, 1 + a + {1 + e, 1 + a + e}}")

    def test_generate_linear_algebras(self):
        simple = str(alg.linear_algebra_for([0, 1, 2], [2]).normalize())
        self.assertEquals("2 + a", simple)

    def test_simple_loop(self):
        loop = alg.generate([0, 1, 2, 3], [(0, 1), (1, 2), (2, 1), (1, 3)], 0, [3], EmptyOptions)
        self.assertEquals("1 + (2)* + 1 + a + e", str(loop))

    def test_simple_loop_2(self):
        loop = alg.generate([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 1), (2, 3), (3, 2), (1, 4)], 0, [4], EmptyOptions)
        self.assertEquals("1 + (1 + (2)* + 1)* + 1 + a + e", str(loop))

    def test_mono_loop(self):
        loop = alg.generate([0], [(0, 0)], 0, [0], EmptyOptions)
        # I am not actually sure this is correct --- it might
        # need to be a + (1 + a)*?  --- Not too sure though.
        # Also not sure it matters.
        self.assertEqual("(1 + a)*", str(loop))

    # This is an edge case we don't really handle well, but
    # it isn't incorrect -- the algebra just makes the automaton
    # seem more general than it is.  That problem will get
    # fixed upon unification though.
    def test_diverging_triple(self):
        triple = alg.generate([0, 1, 2, 3, 4], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)], 0, [4], EmptyOptions)
        print str(triple)

    def test_converging_triple(self):
        triple = alg.generate([0, 1, 2, 3, 4], [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 4), (2, 4)], 0, [4], EmptyOptions)
        print triple

    def test_multiple_removed_edges(self):
        res = alg.generate([0, 1, 2, 3, 4], [(0, 1), (1, 1), (1, 2), (2, 1), (2, 3), (3, 4)], 0, [4], EmptyOptions)
        print(res)

    def test_multiple_loops(self):
        res = alg.generate([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 1), (1, 3), (3, 1), (1, 4)], 0, [4], EmptyOptions)

    def test_double_branch(self):
        res = alg.generate([0, 1, 2, 3, 4, 5], [(0, 1), (0, 2), (2, 3), (1, 3), (3, 4), (3, 5)], 0, [], EmptyOptions)
        self.assertEqual(str(res), "{2, 2} + {1 + e, 1 + e}")

    def test_double_branch_2(self):
        res = alg.generate([0, 1, 2, 3, 4, 5, 6], [(0, 1), (0, 2), (2, 3), (1, 3), (3, 4), (4, 5), (4, 6)], 0, [], EmptyOptions)
        self.assertEqual(str(res), "{2, 2} + 1 + {1 + e, 1 + e}")

class UnificationTest(unittest.TestCase):
    def test_simple_unifier(self):
        res = alg.leq_unify(Const(1, [(1, 2)]), Const(1, [(2, 3)]), EmptyOptions)[0]
        self.assertEqual(res.from_edges, [(1, 2)])
        self.assertEqual(res.to_edges, [(2, 3)])

    def test_unifier_branches(self):
        t1 = Branch([Const(2, [(1, 2), (2, 3)]), Const(1, [(4, 5)])]).normalize()
        t2 = Const(2, [(-1, -2), (-2, -3)]).normalize()

        res = alg.leq_unify(t2, t1, EmptyOptions)[0]
        self.assertEqual(res.to_edges, [(1, 2), (2, 3)])
        self.assertEqual(res.from_edges, [(-1, -2), (-2, -3)])
        self.assertEqual(res.disabled_edges, [(4, 5)])

    def test_structural_modification(self):
        t1 = Sum([Const(1, [1]), Product(Const(1, [1])), Const(1, [1])])
        t2 = Sum([Const(1, [2]), Product(Const(1, [2])), Const(1, [2]), Accept(), End()])

        self.assertEqual(alg.leq(t1, t2, EmptyOptions), False)

    def test_unifier_both_branches(self):
        t1 = Branch([Sum([Const(1, [(0, 1)]), Accept()]), Sum([Const(2, [(1, 2), (2, 3)]), End()])]).normalize()
        t2 = Branch([Sum([Const(1, [(0, 5)]), End()]), Sum([Const(1, [(0, 1)]), Accept()]), Sum([Const(2, [(2, 3), (3, 4)]), End()])]).normalize()
        res = alg.leq_unify(t1, t2, EmptyOptions)
        self.assertNotEqual(res, [])

        # print(res.from_edges)
        # print(res.to_edges)

    def test_unifier_sum_selection(self):
        t1 = Sum([Const(1, [(0, 1)]), Branch([Const(1, [(1, 2)]), End()])])
        t2 = Sum([Const(2, [(0, 1), (1, 2)])]).normalize()
        res = alg.leq_unify(t2, t1, EmptyOptions)
        print res
        self.assertNotEqual(res, [])

    def test_unifier_single_to_branch(self):
        t1 = End()
        t2 = Branch([End(), End()])
        res = alg.leq_unify(t2, t1, EmptyOptions)

        self.assertNotEqual(res, [])

    def test_unifier_deep_sum_selection(self):
        t1 = Sum([Const(3, [(1, 2), (2, 3), (3, 4)]), Accept()]).normalize()
        t2 = Sum([Const(1, [(0, 1)]), Branch([End(), Accept(), Const(1, [(1, 2)])]), Const(1, [(2, 3)]), Accept()]).normalize()
        res = alg.leq_unify(t1, t2, EmptyOptions)
        self.assertNotEqual(res, [])
        # TODO --- Need to consider a case where we have a branch that unifies with more than one term.

    def test_unifier_plus_plus(self):
        t1 = Sum([Const(30, [(0, 1)] * 30)]).normalize()
        t2 = Sum([Const(30, [(0, 1)] * 30)]).normalize()

        res = alg.leq_unify(t1, t2, EmptyOptions)
        self.assertNotEqual(res, [])

    def test_unifier_end(self):
        t1 = End()
        t2 = Sum([Const(1, [(0, 1)]), Const(1, [(1, 2)])]).normalize()

        res = alg.leq_unify(t1, t2, EmptyOptions)
        self.assertNotEqual(res, [])

    def test_unifier_sum_const(self):
        t1 = Const(1, [1])
        t2 = Sum([Const(1, [1]), Const(1, [1])])
        res = alg.leq_unify(t1, t2, EmptyOptions)
        res = [x for x in res if x is not None]
        self.assertNotEqual(res, [])

    def test_unifier_sum_sum(self):
        t1 = Sum([Const(1, [1]), Const(1, [2])])
        t2 = Sum([Const(1, [1]), Product(Const(1, [2])), Const(1, [3])])

        res = alg.leq_unify(t1, t2, EmptyOptions)
        res = [x for x in res if x is not None]
        self.assertNotEqual(res, [])

    def test_unifier_sum_product_const(self):
        t1 = Const(1, [1])
        t2 = Sum([Const(1, [1]), Product(Const(1, [2])), Const(1, [1]), Accept(), End()])

        res = alg.leq_unify(t1, t2, EmptyOptions)
        self.assertNotEqual(res, [])

    def test_unifier_const_to_branch_in_sum(self):
        t1 = Const(1, [1])
        t2 = Sum([Product(Const(1, [1])), Branch([Const(1, [1]), Const(1, [1])])])

        res = alg.leq_unify(t1, t2, EmptyOptions)
        self.assertNotEqual(res, [])

    def test_unifier_sum(self):
        t1 = Sum([ Product(Const(1, [1])), Const(1, [1])])
        t2 = Sum([ Const(1, [2])])

        res = alg.leq_unify(t1, t2, EmptyOptions)
        self.assertNotEqual(res, [])
        res = alg.leq_unify(t2, t1, EmptyOptions)
        self.assertNotEqual(res, [])

class TestDeconstruction(unittest.TestCase):
    def test_graph_for_const(self):
        graph, end_nodes = alg.graph_for(Const(1, [(0, 1)]), {(0, 1): 'a'})
        self.assertEqual(graph.edges, [(0, 1)])
        self.assertEqual(graph.start_state, 0)
        self.assertEqual(end_nodes, [1])

    def test_graph_for_sum(self):
        graph, end_nodes = alg.graph_for(Sum([Const(1, [(0, 1)]), Accept(), End()]), {(0, 1): 'a'})
        self.assertEqual(graph.accepting_states, [1])

    def test_graph_for_loop(self):
        graph, end_nodes = alg.graph_for(Product(Const(1, [(2, 2)])), {(2, 2): 'a'})
        self.assertEqual(graph.edges, [(0, 0)])
        self.assertEqual(end_nodes, [])
        self.assertEqual(graph.nodes, [0])

class StructuralTransformations(unittest.TestCase):
    def test_apply_structural_transformations(self):
        graph, _ = alg.graph_for(Sum([Const(1, [(0, 1)]), Const(1, [(1, 2)])]), {(0, 1): 'a', (1, 2): 'b'} )
        slookup = {
                (0, 0): 'c'
                }
        additions = [
                Modifications([], [Modification(Product(Const(1, [(0, 0)])), [(1, 2)])], slookup)
                ]
        result = alg.apply_structural_transformations_internal(graph, additions, EmptyOptions)
        self.assertTrue((1, 1) in result.edges)
        self.assertEqual(result.symbol_lookup[(1, 1)], 'c')

if __name__ == "__main__":
    unittest.main()
