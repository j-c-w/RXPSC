import unittest
import automata.FST.sjss as sjss

class SJSSTest(unittest.TestCase):
    def test_end_states(self):
        ends = sjss.compute_end_states([0, 1, 2, 3], [(0, 1), (0, 2), (2, 0), (1, 3)])

        self.assertEqual(ends, [3])

    def test_output_lookup(self):
        tab = sjss.generate_output_lookup([0, 1, 2, 3], [(0, 2), (0, 1)])
        
        self.assertEqual(tab[2], [])
        self.assertEqual(sorted(tab[0]), [1, 2])

    def test_loops(self):
        self.assertTrue([1, 2, 0, 1] in sjss.compute_loops_with_duplicates([0, 1, 2], [(0, 1), (1, 2), (2, 0)]))

    def test_sssp(self):
        dists, lut = (sjss.compute_sssp([0, 1, 2, 3], [(0, 1), (1, 2), (1, 3), (0, 3)], 0))
        self.assertTrue(lut[3] == [0, 3])
        self.assertTrue(dists[2] == 2)


    def test_compute_loop_groups(self):
        groups = sjss.compute_loop_groups([0, 1, 2, 3], [(0, 1), (1, 2), (2, 1)], 0)
        self.assertTrue(len(groups[1]) == 2)
        self.assertTrue(sum([len(groups[i]) for i in groups]) == 2)

    def test_compute_loops(self):
        groups = sjss.compute_loops([0, 1, 2, 3], [(0, 1), (1, 2), (2, 1)], 0)
        self.assertTrue(len(groups[1]) == 1)
        self.assertTrue(groups[1][0] == [1, 2, 1])

    def test_branches(self):
        # No change expected.
        self.assertEqual(sjss.compute_branches([0, 1, 2], [(0, 1), (1, 2)], 0)[0], [0, 1, 2])

        self.assertTrue([0, 1] in sjss.compute_branches([0, 1, 2], [(0, 1), (0, 2)], 0))

        self.assertTrue([1, 2] in sjss.compute_branches([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)], 0))

        self.assertEqual(sjss.compute_branches([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 1)], 0), [[0, 1], [1, 2, 3, 1]])
        self.assertTrue([3, 1] in sjss.compute_branches([0, 1, 2, 3, 4], [(0, 1), (2, 4), (1, 2), (2, 3), (3, 1), (3, 4)], 0) )

        self.assertEqual(len(sjss.compute_branches([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 2)], 0)), 2)

    def test_compute_loop_subregion(self):
        nodes = [0, 1, 2, 3, 4]
        edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4)]

        loops = sjss.compute_loops(nodes, edges, 0)
        branches = sjss.compute_branches(nodes, edges, 0)

        new_nodes, new_edges, removed_edges, branches_analysis, loops_analysis = sjss.compute_loop_subregion(nodes, edges, 1, loops, branches)

        self.assertEqual(sorted(new_nodes), [1, 2, 3])
        self.assertEqual(len(new_edges), 2)
        self.assertEqual(len(loops_analysis[1]), 0)


    def test_empty_loop(self):
        nodes = [0]
        edges = [(0, 0)]

        loops = sjss.compute_loops(nodes, edges, 0)
        branches = sjss.compute_branches(nodes, edges, 0)
        self.assertEqual(loops[0], [[ 0, 0 ]])
        # Not entirely sure this is the right thing.
        # Anyway.  Not a problem if this one has to change.
        self.assertEqual(branches, [[0, 0]])

        new_nodes, new_edges, removed_edges, branches_analysis, loops_analysis = sjss.compute_loop_subregion(nodes, edges, 0, loops, branches)
        # Not too sure these need to be like this either.
        self.assertEqual(new_edges, [])
        self.assertEqual(new_nodes, [0])

    def test_compute_loops_2(self):
        self.assertEqual(len(sjss.compute_loops_with_duplicates([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 1), (2, 3), (3, 2), (1, 4)])), 4)

    def test_branches_simple(self):
        self.assertEqual(sjss.compute_branches([1, 2, 3], [(1, 2), (2, 3), (3, 2)], 1), [[1, 2], [2, 3, 2]])

    def test_compute_loop_subregion_2(self):
        self.assertEqual(sjss.compute_loop_subregion([0], [(0, 0)], 0, {0: [[0, 0]]}, [0, 0])[2], [(0, 0)])


if __name__ == "__main__":
    unittest.main()
