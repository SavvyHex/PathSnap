"""Recover directed paths using the same weighted graph as the Viterbi decoder."""
from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class ArcSlice:
    arc_id: int
    start: float
    end: float


class TransitionPaths:
    def __init__(self, graph):
        self.graph = graph
        self.arcs = {int(row.Index): row for row in graph.arcs.itertuples()}
        self.edges = {}
        for arc_id, arc in sorted(self.arcs.items(), key=lambda item: (item[1].length_m, item[0])):
            self.edges.setdefault((arc.from_node, arc.to_node), arc_id)
        self.predecessors = lru_cache(maxsize=128)(self._predecessors)

    def _predecessors(self, node_index):
        _, predecessors = dijkstra(self.graph.graph, directed=True, indices=node_index,
                                   return_predecessors=True)
        return predecessors

    def transition(self, source_id, source_frac, target_id, target_frac, expected_distance):
        """Return directed arc slices whose length agrees with the decoder matrix."""
        source_frac, target_frac = (float(np.clip(value, 0, 1)) for value in (source_frac, target_frac))
        source, target = self.arcs[source_id], self.arcs[target_id]
        if source_id == target_id and source_frac <= target_frac:
            pieces = [ArcSlice(source_id, source_frac, target_frac)]
            connectors = 0
        else:
            start = self.graph.node_index[source.to_node]
            end = self.graph.node_index[target.from_node]
            middle = []
            if start != end:
                predecessors = self.predecessors(start)
                while end != start:
                    previous = int(predecessors[end])
                    if previous < 0 or len(middle) >= len(predecessors):
                        raise ValueError("Missing predecessor path for a finite transition")
                    pair = (self.graph.node_ids[previous], self.graph.node_ids[end])
                    middle.append(ArcSlice(self.edges[pair], 0.0, 1.0))
                    end = previous
            connectors = len(middle)
            pieces = [ArcSlice(source_id, source_frac, 1.0), *reversed(middle),
                      ArcSlice(target_id, 0.0, target_frac)]
        distance = sum((piece.end - piece.start) * self.arcs[piece.arc_id].length_m for piece in pieces)
        if not isclose(distance, expected_distance, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Reconstructed road length differs from the decoder distance")
        return pieces, connectors, float(distance)

    def coordinate(self, arc_id, fraction):
        arc = self.arcs[arc_id]
        start = (float(arc.from_lon), float(arc.from_lat))
        end = (float(arc.to_lon), float(arc.to_lat))
        if fraction == 0:
            return start
        if fraction == 1:
            return end
        return tuple(a + fraction * (b - a) for a, b in zip(start, end))
