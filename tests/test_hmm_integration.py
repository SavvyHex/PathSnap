"""Exercise Member 3's decoder using a directed graph and real model inputs."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from api.inference import predict_route


class FixedClassifier:
    def predict_proba(self, features):
        assert list(features) == ["frac_along", "dist_to_segment_m"]
        probabilities = features["dist_to_segment_m"].to_numpy()
        return np.column_stack((1 - probabilities, probabilities))


class HMMIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.network = pd.DataFrame({
            "arc_id": [0, 1, 2], "from_node": [0, 1, 3], "to_node": [1, 2, 4],
            "from_lon": [77.0, 77.001, 78.0], "to_lon": [77.001, 77.002, 78.001],
            "from_lat": [12.0] * 3, "to_lat": [12.0] * 3,
        })

    def predict(self, candidates, network):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "00000007.csv"
            path.write_text(
                "obs_id,candidate_id,latitude,longitude,frac_along,dist_to_segment_m,label,partition\n"
                "0,0,12.0,77.001,1,0.9,private,private\n"
                "1,,12.0,77.001,,,,\n" + candidates,
                encoding="utf-8",
            )
            return predict_route(path, FixedClassifier(), ["frac_along", "dist_to_segment_m"], 10, network)

    def test_directed_distance_changes_the_selected_arc_and_bridges_candidate_gap(self):
        candidates = ("2,2,12.0,77.001,0,0.9,private,private\n"
                      "2,1,12.0,77.001,0,0.4,private,private\n")
        route, gps, imputed, mode, disconnected, _, _ = self.predict(candidates, self.network)
        self.assertEqual([point["segment"] for point in route], ["0", "0", "1"])
        self.assertTrue(route[1]["imputed"])
        self.assertEqual([point["obs_id"] for point in gps], [0, 1, 2])
        self.assertEqual((imputed, mode, disconnected), (1, "directed_road_network", 0))
        independent, _, _, mode, disconnected, _, _ = self.predict(candidates, None)
        self.assertEqual(independent[-1]["segment"], "2")
        self.assertEqual((mode, disconnected), ("classifier_only", None))

    def test_selected_disconnected_transition_is_explicit(self):
        _, _, _, mode, disconnected, _, _ = self.predict(
            "2,2,12.0,77.001,0,0.9,private,private\n", self.network,
        )
        self.assertEqual((mode, disconnected), ("directed_road_network", 1))
