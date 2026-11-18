"""Tests for offline evaluator preparation and prepared-input discovery."""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from api.evaluator_adapter import (
    METADATA, discover_source_trips, prepare_trip, read_prepared_trip, write_prepared_trip,
)
from api.inference import read_feature_manifest
from api.main import Settings, create_app
from src.config import ARTIFACTS, ROOT, TEST_DIR


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.columns = read_feature_manifest(ARTIFACTS)

    def test_discovery_preserves_exact_ids_and_separates_student_test(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("00000007", "reserved-A", "bad.id"):
                (root / name).mkdir()
            (root / "not-a-directory").touch()
            self.assertEqual(list(discover_source_trips(root)), ["00000007", "reserved-A"])
            self.assertEqual(list(discover_source_trips(root, student_test=True)), ["00000007"])
            self.assertEqual(discover_source_trips(root / "missing"), {})

    def test_missing_and_empty_input_are_reported(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "00000007"
            folder.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, r"00000007\.track"):
                prepare_trip(folder, self.columns)
            (folder / "00000007.track").touch()
            with self.assertRaisesRegex(ValueError, r"Empty input: 00000007\.track"):
                prepare_trip(folder, self.columns)

    def test_no_candidates_is_reported(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "sample"
            folder.mkdir()
            for extension, data in {
                "track": "1 1 0\n", "nodes": "10 10\n11 11\n", "arcs": "0 1\n",
            }.items():
                (folder / f"sample.{extension}").write_text(data, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No road candidates found"):
                prepare_trip(folder, self.columns)

    def test_readiness_rejects_empty_malformed_and_candidate_free_csvs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            header = ",".join([*METADATA, *self.columns]) + "\n"
            path = root / "empty.csv"
            path.write_text(header, encoding="utf-8")
            (root / "malformed.csv").write_text("wrong,header\n", encoding="utf-8")
            (root / "gps-only.csv").write_text(
                header + "0,,43.3,77.1" + "," * len(self.columns) + "\n", encoding="utf-8",
            )
            with TestClient(create_app(Settings(evaluator_dir=root))) as client:
                self.assertEqual(client.get("/trips").json()["trips"], [])
                self.assertEqual(client.get("/ready").status_code, 503)
                path.write_text(header + "0,24,43.3,77.1,0,5,0.5,0,0,0\n", encoding="utf-8")
                self.assertEqual(client.get("/trips").json()["trips"], ["empty"])
                self.assertEqual(client.get("/ready").status_code, 200)


class GenuineStudentTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        supplied = os.environ.get("PATHSNAP_TEST_STUDENT_DIR")
        roots = (
            [Path(supplied)] if supplied else
            [TEST_DIR, ROOT.parent / "Student folder" / "Student folder" / "student_test"]
        )
        for root in roots:
            folder = root / "00000007"
            if all((folder / f"00000007.{extension}").is_file()
                   for extension in ("track", "nodes", "arcs")):
                cls.folder = folder
                break
        else:
            raise unittest.SkipTest("Genuine student-test trip 00000007 is unavailable")
        cls.columns = read_feature_manifest(ARTIFACTS)
        with patch("api.evaluator_adapter.pd.read_csv", wraps=pd.read_csv) as reads:
            cls.frame = prepare_trip(cls.folder, cls.columns)
        cls.input_files = [Path(call.args[0]).suffix for call in reads.call_args_list]

    def test_real_candidates_keep_ids_and_exclude_reference_files(self):
        self.assertEqual(self.input_files, [".track", ".nodes", ".arcs"])
        self.assertEqual(list(self.frame), ["trip_id", *METADATA, *self.columns])
        self.assertEqual(self.frame["trip_id"].unique().tolist(), ["00000007"])
        track = pd.read_csv(self.folder / "00000007.track", sep=r"\s+", header=None)
        arcs = pd.read_csv(self.folder / "00000007.arcs", sep=r"\s+", header=None)
        nodes = pd.read_csv(self.folder / "00000007.nodes", sep=r"\s+", header=None)
        observations = self.frame.drop_duplicates("obs_id")
        self.assertEqual(observations["obs_id"].tolist(), list(range(len(track))))
        self.assertEqual(observations[["longitude", "latitude"]].values.tolist(),
                         track[[0, 1]].values.tolist())
        candidates = self.frame.dropna(subset=["candidate_id"])
        self.assertGreater(len(candidates), 0)
        selected_arcs = arcs.iloc[candidates["candidate_id"].astype(int)]
        self.assertTrue(selected_arcs.isin(nodes.index).all().all())

    def test_idempotent_publication_and_api_prediction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "00000007.csv"
            self.assertTrue(write_prepared_trip(self.frame, path))
            self.assertFalse(write_prepared_trip(self.frame, path))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_prepared_trip(self.frame.iloc[:1], path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(read_prepared_trip(path, self.columns)), [*METADATA, *self.columns])

            settings = Settings(evaluator_dir=root, reference_dir=root / "reference",
                                network_dir=root / "network")
            with TestClient(create_app(settings)) as client:
                self.assertEqual(client.get("/trips").json()["trips"], ["00000007"])
                self.assertEqual(client.get("/ready").status_code, 200)
                with patch("api.evaluator_adapter.generate_candidates",
                           side_effect=AssertionError("Prediction must use prepared rows")):
                    response = client.post("/predict/00000007")
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result["n_observations"], self.frame["obs_id"].nunique())
                self.assertEqual(result["transition_model"], "classifier_only")
                self.assertIsNone(result["disconnected_transitions"])
                self.assertEqual(result["route_kind"], "candidate_sequence")
                self.assertIsNone(result["top1_accuracy"])
                direct = [point for point in result["route"] if not point["imputed"]]
                self.assertTrue(direct)
                real_pairs = {
                    (obs_id, str(segment))
                    for obs_id, segment in self.frame[["obs_id", "candidate_id"]].dropna().itertuples(index=False)
                }
                self.assertTrue(all((point["obs_id"], point["segment"]) in real_pairs for point in direct))
