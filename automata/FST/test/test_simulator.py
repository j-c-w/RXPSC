import automata.FST.simulator.simulate_automata as sim
import automata.FST.simple_graph as simple_graph
import unittest

class TestSim(unittest.TestCase):
    def test_simple(self):
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 1), (1, 2)]
        symbol_lookup = {
                (0, 1): ['a'],
                (1, 1): ['b'],
                (1, 2): ['c']
                }
        accepting_states = [2]
        start_state = 0
        simple_automata = simple_graph.SimpleGraph(nodes, edges, symbol_lookup, accepting_states, start_state)

        self.assertTrue(sim.accepts(simple_automata, 'abbbbc'))
        self.assertTrue(sim.accepts(simple_automata, 'bac'))
        self.assertFalse(sim.accepts(simple_automata, 'caa'))
        self.assertTrue(sim.accepts(simple_automata, 'ac'))


if __name__ == "__main__":
    unittest.main()
