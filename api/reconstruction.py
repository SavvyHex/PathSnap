"""Build ordered GeoJSON sections from Viterbi-selected road positions."""
from dataclasses import dataclass

import numpy as np

from src.hmm.transition_paths import TransitionPaths


@dataclass(frozen=True)
class ReconstructedRoute:
    geojson: dict | None
    connected_transitions: int
    unreachable_transitions: int
    geometry_failures: int
    connecting_arcs: int
    distance_m: float | None
    spans_trip: bool


def reconstruct_route(graph, sequence, arc_ids, fractions, matrices):
    paths = TransitionPaths(graph)
    features, coordinates, section_arcs = [], [], []
    connected = unreachable = failures = connectors = 0
    total_distance = 0.0

    def finish_section():
        if len(coordinates) >= 2:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates.copy()},
                "properties": {"arc_ids": section_arcs.copy()},
            })
        coordinates.clear()
        section_arcs.clear()

    for (start, end), matrix in matrices.items():
        source, target = sequence[start], sequence[end]
        distance = matrix[source, target]
        if not np.isfinite(distance):
            unreachable += 1
            finish_section()
            continue
        connected += 1
        try:
            pieces, count, length = paths.transition(
                arc_ids[start][source], fractions[start][source],
                arc_ids[end][target], fractions[end][target], distance,
            )
            points = []
            for piece in pieces:
                first = paths.coordinate(piece.arc_id, piece.start)
                last = paths.coordinate(piece.arc_id, piece.end)
                if points and points[-1] != first:
                    raise ValueError("Arc endpoints disagree at a shared node")
                if not points:
                    points.append(first)
                if last != first:
                    points.append(last)
            if coordinates and coordinates[-1] != points[0]:
                finish_section()
                failures += 1
        except (ValueError, KeyError):
            failures += 1
            finish_section()
            continue
        connectors += count
        total_distance += length
        coordinates.extend(points[1:] if coordinates else points)
        section_arcs.extend(int(piece.arc_id) for piece in pieces if piece.start != piece.end)

    finish_section()
    return ReconstructedRoute(
        {"type": "FeatureCollection", "features": features} if features else None,
        connected, unreachable, failures, connectors,
        total_distance if connected and (features or (total_distance == 0 and not failures)) else None,
        bool(sequence and sequence[0] >= 0 and sequence[-1] >= 0),
    )
