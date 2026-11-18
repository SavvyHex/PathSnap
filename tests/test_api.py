import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from fastapi.testclient import TestClient

from api.main import Settings, create_app


class FixtureClassifier:
    """Test-only classifier that verifies the exact prediction input."""

    classes_ = [0, 1]
    feature_names_in_ = ["dist_to_segment_m"]

    def predict_proba(self, features):
        assert list(features) == self.feature_names_in_
        positive = np.where(features["dist_to_segment_m"] < 5, 0.9, 0.1)
        return np.column_stack((1 - positive, positive))


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.trips = self.root / "evaluator"
        self.artifacts = self.root / "artifacts"
        self.references = self.root / "reference"
        for directory in (self.trips, self.artifacts, self.references):
            directory.mkdir()
        self.trip = self.trips / "00000007.csv"
        self.trip.write_text(
            "obs_id,candidate_id,latitude,longitude,dist_to_segment_m,label,matched_segment\n"
            "0,001,12.1,77.2,2,0,private\n"
            "0,002,12.1,77.2,10,1,private\n"
            "1,,12.2,77.3,,,\n"
            "2,003,12.3,77.4,5,1,private\n",
            encoding="utf-8",
        )

    def install_fixture_model(self, columns=("dist_to_segment_m",)):
        (self.artifacts / "feature_columns.json").write_text(json.dumps(columns), encoding="utf-8")
        joblib.dump(FixtureClassifier(), self.artifacts / "model_tuned.joblib")

    def client(self, mode="evaluation"):
        return TestClient(create_app(Settings(
            mode=mode, evaluator_dir=self.trips, artifacts_dir=self.artifacts,
            reference_dir=self.references, cors_origins=("http://lab-ui:8080",),
        )))

    def test_demo_without_artifacts_and_cors(self):
        with self.client("development") as client:
            self.assertEqual(client.get("/ready").status_code, 200)
            response = client.get("/trips", headers={"Origin": "http://lab-ui:8080"})
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://lab-ui:8080")
            self.assertTrue(response.headers["X-Request-ID"])
            self.assertEqual(response.json()["trips"], ["development-demo"])
            data = client.post("/predict/development-demo").json()
            self.assertEqual(data["mode"], "development")
            for key in ("top1_accuracy", "n_imputed", "elapsed_s"):
                self.assertIsNone(data[key])
            self.assertTrue(all(point["segment"] is None for point in data["route"]))
            self.assertEqual(client.post("/predict/00000007").status_code, 404)

    def test_missing_model_keeps_liveness_and_fails_readiness(self):
        with self.assertLogs("uvicorn.error.pathsnap", level="ERROR"):
            with self.client() as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/ready").status_code, 503)
                self.assertEqual(client.post("/predict/00000007").status_code, 503)
                self.assertEqual(client.post("/predict/absent").status_code, 404)

    def test_rejects_reference_feature_manifest(self):
        self.install_fixture_model(("label",))
        with self.assertLogs("uvicorn.error.pathsnap", level="ERROR"):
            with self.client() as client:
                self.assertFalse(client.get("/health").json()["model_loaded"])
                self.assertEqual(client.get("/ready").status_code, 503)

    def test_aggregate_scoring_does_not_influence_decoding(self):
        self.install_fixture_model()
        reference = self.references / self.trip.name
        with self.client() as client:
            self.assertEqual(client.get("/ready").status_code, 200)
            unscored = client.post("/predict/00000007").json()
            self.assertIsNone(unscored["top1_accuracy"])
            reference.write_text("obs_id,matched_segment\n0,001\n2,secret-reference\n", encoding="utf-8")
            response = client.post("/predict/00000007")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["route"], unscored["route"])
            self.assertEqual([point["segment"] for point in data["route"]], ["001", "001", "003"])
            self.assertEqual(data["n_imputed"], 1)
            self.assertTrue(data["route"][1]["imputed"])
            self.assertEqual(data["scored_observations"], 2)
            self.assertEqual(data["top1_accuracy"], 0.5)
            self.assertNotIn("secret-reference", response.text)
            self.assertNotIn("matched_segment", response.text)

    def test_invalid_reference_preserves_prediction(self):
        self.install_fixture_model()
        (self.references / self.trip.name).write_text("wrong,header\n0,001\n", encoding="utf-8")
        with self.client() as client, self.assertLogs("uvicorn.error.pathsnap", level="ERROR"):
            response = client.post("/predict/00000007")
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(response.json()["top1_accuracy"])

    def test_invalid_evaluator_data_returns_generic_error(self):
        self.install_fixture_model()
        self.trip.write_text("invalid,header\nprivate,0\n", encoding="utf-8")
        with self.client() as client, self.assertLogs("uvicorn.error.pathsnap", level="ERROR"):
            response = client.post("/predict/00000007", headers={"Origin": "http://lab-ui:8080"})
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://lab-ui:8080")
            self.assertEqual(response.json(), {"detail": "The server could not complete the request."})
