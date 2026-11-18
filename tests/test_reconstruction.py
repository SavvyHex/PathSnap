"""Check reconstructed paths against Member 3's original distance oracle."""
import unittest

import numpy as np
import pandas as pd

from api.reconstruction import reconstruct_route
from src.hmm.road_graph import RoadGraph
from src.hmm.transition_paths import TransitionPaths


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        nodes = [(10.0, 20.0), (10.01, 20.0), (10.02, 20.0),
                 (10.02, 20.01), (11.0, 21.0), (11.01, 21.0)]
        edges = [(0, 1), (1, 0), (1, 0), (2, 3), (1, 2), (4, 5)]
        self.arcs = pd.DataFrame([
            (arc, source, target, *nodes[source], *nodes[target])
            for arc, (source, target) in enumerate(edges)
        ], columns=["arc_id", "from_node", "to_node", "from_lon", "from_lat", "to_lon", "to_lat"])
        self.graph = RoadGraph(self.arcs)
        self.paths = TransitionPaths(self.graph)

    def transition(self, source, start, target, end):
        expected = self.graph.transition_distances([source], [start], [target], [end])[0, 0]
        pieces, intermediate, distance = self.paths.transition(source, start, target, end, expected)
        self.assertAlmostEqual(distance, expected)
        return [(part.arc_id, part.start, part.end) for part in pieces], intermediate

    def test_same_arc_forward_trims_both_ends(self):
        self.assertEqual(self.transition(0, 0.25, 0, 0.75), ([(0, 0.25, 0.75)], 0))
        self.assertAlmostEqual(self.paths.coordinate(0, 0.25)[0], 10.0025)
        self.assertEqual(self.paths.coordinate(0, 0.25)[1], 20.0)

    def test_same_arc_backward_takes_legal_reverse_edge_with_stable_parallel_tie(self):
        self.assertEqual(self.transition(0, 0.75, 0, 0.25),
                         ([(0, 0.75, 1), (1, 0, 1), (0, 0, 0.25)], 1))
        self.assertEqual(self.paths.coordinate(1, 0), (10.01, 20.0))
        self.assertEqual(self.paths.coordinate(1, 1), (10.0, 20.0))

    def test_parallel_edge_uses_minimum_graph_weight(self):
        # Create unequal weights for the same node pair to exercise edge selection.
        self.arcs.loc[1, "to_lon"] = 9.99
        graph = RoadGraph(self.arcs)
        expected = graph.transition_distances([0], [0.75], [0], [0.25])[0, 0]
        pieces, _, distance = TransitionPaths(graph).transition(0, 0.75, 0, 0.25, expected)
        self.assertEqual([piece.arc_id for piece in pieces], [0, 2, 0])
        self.assertAlmostEqual(distance, expected)

    def test_distinct_arcs_geojson_splits_at_unreachable_transition(self):
        arc_ids = [[0], [3], [5], [5]]
        fractions = [[0.5], [0.5], [0.0], [1.0]]
        matrices = {
            (i - 1, i): self.graph.transition_distances(arc_ids[i - 1], fractions[i - 1],
                                                        arc_ids[i], fractions[i])
            for i in range(1, 4)
        }
        self.assertTrue(np.isinf(matrices[1, 2][0, 0]))
        result = reconstruct_route(self.graph, [0, 0, 0, 0], arc_ids, fractions, matrices)
        self.assertEqual((result.connected_transitions, result.unreachable_transitions), (2, 1))
        self.assertEqual(result.connecting_arcs, 1)
        self.assertAlmostEqual(result.distance_m, matrices[0, 1][0, 0] + matrices[2, 3][0, 0])
        first, second = result.geojson["features"]
        self.assertEqual(first["properties"]["arc_ids"], [0, 4, 3])
        coordinates = np.array(first["geometry"]["coordinates"])
        np.testing.assert_allclose(coordinates, [[10.005, 20], [10.01, 20], [10.02, 20], [10.02, 20.005]])
        self.assertEqual(second["geometry"]["coordinates"], [(11.0, 21.0), (11.01, 21.0)])
