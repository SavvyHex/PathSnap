"""Directed road-network distances for HMM candidate transitions."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.features.build_features import haversine_m


class RoadGraph:
    """Road arcs plus lazily cached shortest paths from arc end nodes.

    Candidate transition distance includes the remaining distance on the
    source arc, the directed shortest path through the network, and the
    traversed distance on the destination arc. Missing/disconnected paths
    remain infinite for the decoder to penalize explicitly.
    """

    def __init__(self, arcs: pd.DataFrame):
        required = {"arc_id", "from_node", "to_node", "from_lat", "from_lon", "to_lat", "to_lon"}
        missing = required.difference(arcs.columns)
        if missing:
            raise ValueError(f"road network is missing columns: {sorted(missing)}; rerun preprocessing")
        self.arcs = arcs.drop_duplicates("arc_id").set_index("arc_id").copy()
        self.arcs["from_node"] = self.arcs["from_node"].astype(int)
        self.arcs["to_node"] = self.arcs["to_node"].astype(int)
        self.arcs["length_m"] = haversine_m(
            self.arcs["from_lat"].to_numpy(), self.arcs["from_lon"].to_numpy(),
            self.arcs["to_lat"].to_numpy(), self.arcs["to_lon"].to_numpy(),
        )
        self.node_ids = np.unique(np.r_[self.arcs.from_node.to_numpy(), self.arcs.to_node.to_numpy()])
        self.node_index = {int(node): i for i, node in enumerate(self.node_ids)}
        edges = (self.arcs.reset_index().groupby(["from_node", "to_node"], as_index=False)
                 .length_m.min())
        rows = edges.from_node.map(self.node_index).to_numpy()
        cols = edges.to_node.map(self.node_index).to_numpy()
        self.graph = csr_matrix((edges.length_m.to_numpy(), (rows, cols)),
                                shape=(len(self.node_ids), len(self.node_ids)))

    @lru_cache(maxsize=512)
    def _from_node(self, node_id: int) -> np.ndarray:
        """Shortest directed distances from a node to all graph nodes."""
        return dijkstra(self.graph, directed=True, indices=self.node_index[int(node_id)])

    def transition_distances(self, from_arc_ids, from_fracs, to_arc_ids, to_fracs) -> np.ndarray:
        """Return an (n_from, n_to) matrix of road distances in metres."""
        source_ids = np.asarray(from_arc_ids, dtype=int)
        target_ids = np.asarray(to_arc_ids, dtype=int)
        source_fracs = np.clip(np.asarray(from_fracs, dtype=float), 0.0, 1.0)
        target_fracs = np.clip(np.asarray(to_fracs, dtype=float), 0.0, 1.0)
        out = np.full((len(source_ids), len(target_ids)), np.inf, dtype=float)
        for i, source_id in enumerate(source_ids):
            if source_id not in self.arcs.index:
                continue
            source = self.arcs.loc[source_id]
            dist_from_end = self._from_node(int(source.to_node))
            remaining = (1.0 - source_fracs[i]) * float(source.length_m)
            for j, target_id in enumerate(target_ids):
                if target_id not in self.arcs.index:
                    continue
                target = self.arcs.loc[target_id]
                middle = dist_from_end[self.node_index[int(target.from_node)]]
                if np.isfinite(middle):
                    out[i, j] = remaining + middle + target_fracs[j] * float(target.length_m)
                # Same directed arc can be traversed directly without a loop.
                if source_id == target_id and target_fracs[j] >= source_fracs[i]:
                    out[i, j] = min(out[i, j], (target_fracs[j] - source_fracs[i]) * float(source.length_m))
        return out
