"""Focused Phase 2 model, geometry, and reference contracts."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

from api.geometry import lookup_geometry
from api.inference import MODEL_COLUMNS, candidate_probabilities, load_model
from api.scoring import score_route
from src.config import ARTIFACTS


class ModelTests(unittest.TestCase):
    def test_tuned_artifact_matches_training_schema_and_predicts(self):
        model, columns = load_model(ARTIFACTS)
        self.assertEqual(tuple(columns), MODEL_COLUMNS)
        self.assertEqual(tuple(model.feature_names_in_), MODEL_COLUMNS)
        features = pd.DataFrame([[0, 2, 0.5, 12, 0, 0]], columns=columns)
        probabilities = candidate_probabilities(model, features)
        self.assertEqual(probabilities.shape, (1,))
        self.assertTrue(0 <= probabilities[0] <= 1)

    def test_rejects_reordered_or_incomplete_manifest(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for columns in (MODEL_COLUMNS[::-1], MODEL_COLUMNS[:-1]):
                with self.subTest(columns=columns):
                    (path / "feature_columns.json").write_text(json.dumps(columns), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "feature order"):
                        load_model(path)

    def test_requires_both_artifacts_and_probability_support(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_model(path)
            (path / "feature_columns.json").write_text(json.dumps(MODEL_COLUMNS), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_model(path)
            joblib.dump(object(), path / "model_tuned.joblib")
            with self.assertRaisesRegex(ValueError, "predict_proba"):
                load_model(path)


class GeometryTests(unittest.TestCase):
    def test_trip_scoped_arcs_use_longitude_latitude_and_report_missing_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            route = [
                {"obs_id": 0, "segment": "17", "lat": 10, "lon": 30, "imputed": False},
                {"obs_id": 1, "segment": "18", "lat": 11, "lon": 31, "imputed": True},
            ]
            network = pd.DataFrame({
                "arc_id": [17], "from_lon": [77.1], "from_lat": [12.1],
                "to_lon": [77.2], "to_lat": [12.2],
            })
            network.to_parquet(path / "00000007.arcs.parquet")
            result = lookup_geometry(path, "00000007", route)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["reason"], "missing_candidate_arcs")
            self.assertEqual(len(result["geojson"]["features"]), 1)
            feature = result["geojson"]["features"][0]
            self.assertEqual(feature["geometry"]["coordinates"], [[77.1, 12.1], [77.2, 12.2]])
            self.assertEqual(feature["properties"]["segment"], "17")
            self.assertEqual(lookup_geometry(path, "00000007", route[:1])["status"], "available")
            self.assertIsNone(lookup_geometry(path, "00000007", route[1:])["geojson"])
            self.assertEqual(lookup_geometry(path, "7", route), {
                "status": "unavailable", "reason": "missing_network", "geojson": None,
            })


class ScoringTests(unittest.TestCase):
    def test_missing_or_disjoint_reference_leaves_accuracy_null(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "00000007.csv"
            route = [{"obs_id": 0, "segment": "001"}]
            self.assertEqual(score_route(path, route), (None, 0))
            path.write_text("obs_id,matched_segment\n1,001\n", encoding="utf-8")
            self.assertEqual(score_route(path, route), (None, 0))
            path.write_text("obs_id,matched_segment\n0,001\n", encoding="utf-8")
            self.assertEqual(score_route(path, route), (1.0, 1))
