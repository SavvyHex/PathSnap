"""Phase 4 coverage for directed geometry and private aggregate scoring."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from fastapi.testclient import TestClient

from api.geometry import load_road_arcs, lookup_geometry
from api.inference import MODEL_COLUMNS
from api.main import Settings, create_app
from api.scoring import score_arc_overlap, score_route


def point(obs_id, segment, imputed=False):
    return {"obs_id": obs_id, "segment": segment, "imputed": imputed, "lat": 1, "lon": 2}


def write_network(root):
    folder = root / "00000007"
    folder.mkdir(parents=True)
    (folder / "00000007.nodes").write_text("77.1 43.1\n77.2 43.2\n77.3 43.3\n", encoding="utf-8")
    (folder / "00000007.arcs").write_text("0 1\n99 1\n1 0\n1 2\n", encoding="utf-8")
    return folder


class GeometryScoringTests(unittest.TestCase):
    def test_raw_arcs_keep_direction_positions_and_repeated_predictions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_network(root)
            route = [point(0, "0"), point(1, "2"), point(2, "0"), point(3, "1")]
            result = lookup_geometry(root / "parquet", "00000007", route, root)
            self.assertEqual(result["status"], "partial")
            features = result["geojson"]["features"]
            self.assertEqual([feature["properties"]["obs_id"] for feature in features], [0, 1, 2])
            self.assertEqual([feature["geometry"]["coordinates"] for feature in features], [
                [[77.1, 43.1], [77.2, 43.2]],
                [[77.2, 43.2], [77.1, 43.1]],
                [[77.1, 43.1], [77.2, 43.2]],
            ])
            self.assertEqual(lookup_geometry(root, "00000007", route[-1:], root)["reason"],
                             "missing_candidate_arcs")
            self.assertEqual(lookup_geometry(root, "7", route, root)["reason"], "missing_network")

    def test_duplicate_catalog_ids_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                "arc_id": [0, 0], "from_lon": [77, 78], "from_lat": [43, 44],
                "to_lon": [78, 77], "to_lat": [44, 43],
            }).to_parquet(root / "00000007.arcs.parquet")
            with self.assertRaisesRegex(ValueError, "Invalid road arc IDs"):
                load_road_arcs(root, "00000007")

    def test_jaccard_counts_distinct_directed_arcs_and_ignores_imputation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "00000007.route"
            route = [point(0, "0"), point(1, "0"), point(2, "2"), point(3, "3", imputed=True)]
            arcs = {0: (), 2: (), 3: ()}
            self.assertIsNone(score_arc_overlap(path, route, arcs)["score"])
            path.write_text("0\n3\n3\n", encoding="utf-8")
            self.assertAlmostEqual(score_arc_overlap(path, route, arcs)["score"], 1 / 3)
            self.assertEqual(score_arc_overlap(path, [point(0, "2")], arcs)["score"], 0)
            self.assertIsNone(score_arc_overlap(path, route, None)["score"])
            path.write_text("0\n999\n", encoding="utf-8")
            self.assertEqual(score_arc_overlap(path, route, arcs)["reason"], "invalid_route_reference")

    def test_observation_accuracy_uses_only_matching_observation_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "00000007.csv"
            route = [point(0, "001"), point(1, "001", imputed=True), point(2, "002")]
            self.assertEqual(score_route(path, route), (None, 0))
            path.write_text("obs_id,matched_segment\n0,001\n1,1\n3,002\n", encoding="utf-8")
            self.assertEqual(score_route(path, route), (0.5, 2))
            path.write_text("obs_id,matched_segment\n0,001\n0,002\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid reference rows"):
                score_route(path, route)

    def test_api_preserves_route_and_never_exposes_private_reference_arcs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            network = root / "network"
            write_network(network)
            reference = root / "reference" / "00000007"
            reference.mkdir(parents=True)
            trips = root / "prepared"
            trips.mkdir()
            pd.DataFrame(
                [[0, 0, 1, 2, 0, 1, 0.5, 0, 0, 0], [1, 2, 1.1, 2.1, 0, 1, 0.5, 0, 0, 0]],
                columns=["obs_id", "candidate_id", "latitude", "longitude", *MODEL_COLUMNS],
            ).to_csv(trips / "00000007.csv", index=False)
            settings = Settings(evaluator_dir=trips, reference_dir=root / "observation_refs",
                                network_source_dir=network, route_reference_dir=reference.parent)
            with TestClient(create_app(settings)) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/ready").status_code, 200)
                self.assertEqual(client.get("/trips").json()["trips"], ["00000007"])
                before = client.post("/predict/00000007").json()
                self.assertIsNone(before["route_overlap"]["score"])
                (reference / "00000007.route").write_text("0\n3\n", encoding="utf-8")
                response = client.post("/predict/00000007")
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result["route"], before["route"])
                self.assertEqual(result["geometry"], before["geometry"])
                self.assertEqual(result["geometry"]["covered_observations"], 2)
                self.assertEqual(result["geometry"]["decoded_observations"], 2)
                self.assertEqual(result["geometry"]["status"], "available")
                self.assertEqual(result["route_kind"], "candidate_sequence")
                self.assertEqual(result["transition_model"], "directed_road_network")
                self.assertEqual(result["disconnected_transitions"], 0)
                self.assertEqual(result["route_completeness"], "connected")
                self.assertEqual(result["connected_transitions"], 1)
                self.assertEqual(result["reconstructed_connecting_arcs"], 0)
                self.assertGreater(result["route_distance_m"], 0)
                self.assertEqual(result["connected_geometry"], before["connected_geometry"])
                self.assertAlmostEqual(result["route_overlap"]["score"], 1 / 3)
                self.assertIsNone(result["top1_accuracy"])
                self.assertEqual(result["scored_observations"], 0)
                self.assertEqual(set(result["route_overlap"]), {"status", "reason", "method", "score"})
