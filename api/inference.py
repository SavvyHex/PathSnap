"""Model validation, candidate emissions, and Viterbi integration."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from api.evaluator_adapter import read_prepared_trip
from api.geometry import TOPOLOGY_COLUMNS, parse_arc_id
from src.features.build_features import haversine_m
from src.hmm.road_graph import RoadGraph
from src.hmm.viterbi import viterbi


MODEL_COLUMNS = (
    "obs_rank", "dist_to_segment_m", "frac_along", "speed_mps",
    "accel_mps2", "turning_angle_deg",
)
ROAD_NETWORK = "directed_road_network"
CLASSIFIER_ONLY = "classifier_only"


def read_feature_manifest(directory: Path, *, allow_custom_classifier: bool = False) -> list[str]:
    """Read the ordered input schema shared by preparation and inference."""
    columns = json.loads((directory / "feature_columns.json").read_text(encoding="utf-8"))
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(col, str) or col not in MODEL_COLUMNS for col in columns)
            or len(columns) != len(set(columns))):
        raise ValueError("Invalid feature manifest")
    ordered_subset = [column for column in MODEL_COLUMNS if column in columns]
    if columns != ordered_subset or (not allow_custom_classifier and tuple(columns) != MODEL_COLUMNS):
        raise ValueError("Feature manifest differs from the required model feature order")
    return columns


def load_model(directory: Path, *, allow_custom_classifier: bool = False):
    """Load the tuned artifact and validate its schema and probability interface."""
    columns = read_feature_manifest(directory, allow_custom_classifier=allow_custom_classifier)
    model = joblib.load(directory / "model_tuned.joblib")
    if isinstance(model, BaseEstimator) and tuple(columns) != MODEL_COLUMNS:
        raise ValueError("The scikit-learn model requires the full six-feature manifest")
    if not callable(getattr(model, "predict_proba", None)) or list(getattr(model, "classes_", ())) != [0, 1]:
        raise ValueError("Expected a binary classifier with predict_proba and classes [0, 1]")
    if list(getattr(model, "feature_names_in_", ())) != columns:
        raise ValueError("Model feature names and order differ from manifest")
    if getattr(model, "n_features_in_", len(columns)) != len(columns):
        raise ValueError("Model feature count differs from manifest")
    return model, columns


def candidate_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    """Return one positive-class probability per candidate row."""
    if features.empty:
        return np.empty(0)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if (probabilities.shape != (len(features), 2) or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Invalid classifier probabilities")
    return probabilities[:, 1]


def predict_route(path, model, columns, beta, network=None):
    frame = read_prepared_trip(path, columns)
    candidates = frame.loc[frame["candidate_id"].notna()]
    features = candidates[columns].apply(pd.to_numeric)
    probabilities = candidate_probabilities(model, features)
    frame["score"] = pd.Series(probabilities, index=candidates.index)

    observations = []
    for obs_id, group in frame.groupby("obs_id", sort=True):
        lat, lon = group[["latitude", "longitude"]].iloc[0]
        group = group.dropna(subset=["candidate_id"])
        observations.append((int(obs_id), float(lat), float(lon), group))
    emissions = [group["score"].to_numpy() for _, _, _, group in observations]
    observed = [i for i, scores in enumerate(emissions) if scores.size]
    mode = CLASSIFIER_ONLY
    disconnected = None
    limitation = "Directed road topology is unavailable."
    imputed_count = len(observations) - len(observed)

    if network is not None and not network.empty and set(TOPOLOGY_COLUMNS).issubset(network.columns):
        topology = network[TOPOLOGY_COLUMNS]
        if (not all(pd.api.types.is_integer_dtype(topology[col]) for col in topology)
                or topology.isna().any().any() or (topology < 0).any().any()):
            raise ValueError("Invalid directed road topology")
        if "frac_along" in frame and len(observed) >= 2:
            sequence, disconnected = decode_road_sequence(observations, emissions, network, beta)
            mode = ROAD_NETWORK
            limitation = "Member 3's decoder returns distances and candidate states. Connecting arc sequences are unavailable."
        else:
            limitation = ("Fewer than two observations have candidates." if len(observed) < 2
                          else "Candidate projection fractions are unavailable.")
    if mode == CLASSIFIER_ONLY:
        sequence = [int(np.argmax(scores)) if scores.size else -1 for scores in emissions]
    route = []
    previous = None
    for (obs_id, lat, lon, group), index in zip(observations, sequence, strict=True):
        imputed = group.empty
        segment = previous if imputed else (str(group["candidate_id"].iloc[index]) if index >= 0 else None)
        if segment is not None:
            route.append(dict(obs_id=obs_id, segment=segment, lat=lat, lon=lon, imputed=imputed))
        previous = segment
    gps = [dict(obs_id=obs_id, lat=lat, lon=lon) for obs_id, lat, lon, _ in observations]
    return route, gps, imputed_count, mode, disconnected, limitation


def decode_road_sequence(observations, emissions, network, beta):
    """Supply Member 3's directed transition matrices in candidate row order."""
    graph = RoadGraph(network)
    arc_ids, fractions = [], []
    for _, _, _, group in observations:
        ids = [parse_arc_id(value) for value in group["candidate_id"]]
        arc_ids.append(np.array([-1 if value is None else value for value in ids], dtype=np.int64))
        fractions.append(pd.to_numeric(group["frac_along"]).to_numpy(dtype=float))
    gps_distances = [
        haversine_m(lat1, lon1, lat2, lon2)
        for (_, lat1, lon1, _), (_, lat2, lon2, _) in zip(observations, observations[1:])
    ]
    matrices = {}

    def candidate_distances(start, end):
        distances = graph.transition_distances(arc_ids[start], fractions[start], arc_ids[end], fractions[end])
        matrices[start, end] = distances
        return distances

    normalized = []
    for scores in emissions:
        positive = np.clip(scores, 1e-12, None)
        normalized.append(positive / positive.sum() if positive.size else positive)
    decoded = viterbi(normalized, gps_distances, candidate_distances, beta=beta)
    disconnected = sum(
        not np.isfinite(distances[decoded.sequence[start], decoded.sequence[end]])
        for (start, end), distances in matrices.items()
    )
    return decoded.sequence, int(disconnected)


