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
        self.assertTrue([1, 2, 0, 1] in sjss.compute_loops([0, 1, 2], [(0, 1), (1, 2), (2, 0)]))

    def test_branches(self):
        # No change expected.
        self.assertEqual(sjss.compute_branches([0, 1, 2], [(0, 1), (1, 2)], 0)[0], [0, 1, 2])

        self.assertTrue([0, 1] in sjss.compute_branches([0, 1, 2], [(0, 1), (0, 2)], 0))

        self.assertTrue([0, 1, 3] in sjss.compute_branches([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)], 0))

        self.assertEqual(sjss.compute_branches([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 1)], 0), [[0, 1, 2, 3, 1]])
        self.assertTrue([3, 1] in sjss.compute_branches([0, 1, 2, 3, 4], [(0, 1), (2, 4), (1, 2), (2, 3), (3, 1), (3, 4)], 0) )


if __name__ == "__main__":
    unittest.main()