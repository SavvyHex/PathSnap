"""Aggregate scoring against private server-side references."""
from pathlib import Path

import pandas as pd

from api.geometry import parse_arc_id


OVERLAP_METHOD = "directed_arc_set_jaccard"


def score_route(reference_path, route):
    """Return exact observation-level match fraction and comparison count."""
    if reference_path.is_symlink() or reference_path.parent.is_symlink():
        return None, 0
    try:
        reference = pd.read_csv(reference_path, usecols=["obs_id", "matched_segment"],
                                dtype={"matched_segment": "string"},
                                keep_default_na=False, na_values=[""])
    except FileNotFoundError:
        return None, 0
    if reference.empty:
        return None, 0
    obs_ids = pd.to_numeric(reference["obs_id"])
    if (not pd.api.types.is_integer_dtype(obs_ids) or obs_ids.isna().any()
            or (obs_ids < 0).any() or obs_ids.duplicated().any()
            or reference["matched_segment"].fillna("").str.strip().eq("").any()):
        raise ValueError("Invalid reference rows")
    reference["obs_id"] = obs_ids
    segments = reference.set_index("obs_id")["matched_segment"].to_dict()
    comparisons = [
        point["segment"] == segments[point["obs_id"]]
        for point in route if point["obs_id"] in segments
    ]
    return (sum(comparisons) / len(comparisons) if comparisons else None), len(comparisons)


def overlap_unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason, "method": OVERLAP_METHOD, "score": None}


def score_arc_overlap(reference_path: Path, route: list[dict], road_arcs: dict | None) -> dict:
    """Compare distinct predicted and reference directed arc IDs as sets."""
    if reference_path.is_symlink() or reference_path.parent.is_symlink():
        return overlap_unavailable("missing_route_reference")
    try:
        reference = {parse_arc_id(line.strip()) for line in
                     reference_path.read_text(encoding="utf-8").splitlines()}
    except FileNotFoundError:
        return overlap_unavailable("missing_route_reference")
    if not reference or None in reference:
        return overlap_unavailable("invalid_route_reference")
    if road_arcs is None:
        return overlap_unavailable("missing_network")
    if not reference.issubset(road_arcs):
        return overlap_unavailable("invalid_route_reference")
    predicted = {parse_arc_id(point["segment"]) for point in route if not point["imputed"]}
    if not predicted:
        return overlap_unavailable("no_decoded_candidates")
    if not predicted.issubset(road_arcs):
        return overlap_unavailable("unresolved_predicted_arcs")
    return {
        "status": "available", "reason": None, "method": OVERLAP_METHOD,
        "score": len(predicted & reference) / len(predicted | reference),
    }


