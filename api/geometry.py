"""Resolve directed road arcs from prepared Parquet or original network files."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from api.evaluator_adapter import TRIP_ID, read_raw_table


ARC_ID = re.compile(r"[0-9]{1,19}")
ARC_COLUMNS = ["arc_id", "from_lon", "from_lat", "to_lon", "to_lat"]
TOPOLOGY_COLUMNS = ["from_node", "to_node"]


def parse_arc_id(value) -> int | None:
    text = str(value)
    arc_id = int(text) if ARC_ID.fullmatch(text) else None
    return arc_id if arc_id is None or arc_id <= 2**63 - 1 else None


def geometry_unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason, "geojson": None}


def _raw_arc_table(folder: Path) -> pd.DataFrame:
    nodes = read_raw_table(folder, "nodes")
    arcs = read_raw_table(folder, "arcs")
    # Keep the original row index even when an arc references a missing node.
    valid = arcs["from_node"].isin(nodes.index) & arcs["to_node"].isin(nodes.index)
    arcs = arcs.loc[valid]
    starts = nodes.loc[arcs["from_node"]]
    ends = nodes.loc[arcs["to_node"]]
    return pd.DataFrame({
        "arc_id": arcs.index,
        "from_node": arcs["from_node"].to_numpy(),
        "to_node": arcs["to_node"].to_numpy(),
        "from_lon": starts["longitude"].to_numpy(),
        "from_lat": starts["latitude"].to_numpy(),
        "to_lon": ends["longitude"].to_numpy(),
        "to_lat": ends["latitude"].to_numpy(),
    })


def load_network(network_dir: Path, trip_id: str, source_dir: Path | None = None) -> pd.DataFrame | None:
    """Read and validate the geometry shared by decoding and rendering."""
    if not TRIP_ID.fullmatch(trip_id):
        raise ValueError("Invalid network trip ID")
    try:
        if source_dir is None:
            arcs = pd.read_parquet(network_dir / f"{trip_id}.arcs.parquet")
        else:
            arcs = _raw_arc_table(source_dir / trip_id)
    except FileNotFoundError:
        return None

    ids = arcs["arc_id"]
    if (not pd.api.types.is_integer_dtype(ids) or ids.isna().any()
            or (ids < 0).any() or ids.duplicated().any()):
        raise ValueError("Invalid road arc IDs")
    coordinates = arcs[ARC_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or (np.abs(coordinates) > [180, 90, 180, 90]).any():
        raise ValueError("Invalid road arc coordinates")
    arcs[ARC_COLUMNS[1:]] = coordinates
    return arcs


def index_road_arcs(network: pd.DataFrame | None) -> dict | None:
    if network is None:
        return None
    return dict(zip(network["arc_id"], network[ARC_COLUMNS[1:]].to_numpy(), strict=True))


def load_road_arcs(network_dir: Path, trip_id: str, source_dir: Path | None = None) -> dict | None:
    """Load a trip-local index keyed by zero-based directed arc ID."""
    return index_road_arcs(load_network(network_dir, trip_id, source_dir))


def resolve_geometry(by_id: dict | None, route: list[dict]) -> dict:
    """Emit individual directed LineStrings in observation order with coverage."""
    if not route:
        return geometry_unavailable("no_decoded_candidates")
    if by_id is None:
        return geometry_unavailable("missing_network")
    features = []
    for point in route:
        arc_id = parse_arc_id(point["segment"])
        if arc_id not in by_id:
            continue
        from_lon, from_lat, to_lon, to_lat = by_id[arc_id]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[from_lon, from_lat], [to_lon, to_lat]],
            },
            "properties": {
                "obs_id": point["obs_id"], "segment": point["segment"],
                "imputed": point["imputed"],
            },
        })
    if not features:
        return geometry_unavailable("missing_candidate_arcs")
    partial = len(features) < len(route)
    return {
        "status": "partial" if partial else "available",
        "reason": "missing_candidate_arcs" if partial else None,
        "geojson": {"type": "FeatureCollection", "features": features},
    }


def lookup_geometry(
    network_dir: Path, trip_id: str, route: list[dict], source_dir: Path | None = None,
) -> dict:
    """Look up a trip and return GeoJSON for its selected road arcs."""
    arcs = load_road_arcs(network_dir, trip_id, source_dir) if route else None
    return resolve_geometry(arcs, route)
