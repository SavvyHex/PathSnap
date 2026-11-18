"""Raw-data parsing, cleaning, candidate generation and labelling.

File formats (verified against the real 00000001 sample files):
  <trip>.track  TAB-separated: lon, lat, seconds   (one GPS obs per line, CRLF)
  <trip>.nodes  TAB-separated: lon, lat            (node id = 0-based line number)
  <trip>.arcs   TAB-separated: from_node, to_node  (arc id = 0-based line number)
  <trip>.route  one arc id per line (reference route, route-level ground truth)

Coordinates are lon/lat degrees. Node/arc ids are positional - the id
is the line number in its file.

Outputs per trip into data/clean/<partition>/<trip>.parquet:
  one row per (observation, candidate) with candidate attributes and a
  binary label. The per-observation reference arc is approximated as
  the nearest candidate arc that belongs to the trip's reference
  route: .route is route-level, not a per-point match, so this proxy
  is stated in the report, not treated as verified ground truth.

Also writes results/data_inspection.csv (counts and missing values per
trip) and data/clean/network/<trip>.arcs.parquet (arc geometry for the
HMM transition stage and Folium maps).
"""
import numpy as np
import pandas as pd

from src.config import (
    DATA_CLEAN, RESULTS,
    TRAIN_DIR, TEST_DIR, TRAIN_TRIPS, TEST_TRIPS,
    N_CANDIDATES, CANDIDATE_MAX_DIST_M,
)

EARTH_R = 6_371_000.0


def parse_trip(folder):
    tid = folder.name
    track = pd.read_csv(folder / f"{tid}.track", sep=r"\s+", header=None,
                        names=["longitude", "latitude", "seconds"])
    track["obs_id"] = range(len(track))
    nodes = pd.read_csv(folder / f"{tid}.nodes", sep=r"\s+", header=None,
                        names=["longitude", "latitude"])
    nodes["node_id"] = range(len(nodes))
    arcs = pd.read_csv(folder / f"{tid}.arcs", sep=r"\s+", header=None,
                       names=["from_node", "to_node"])
    arcs["arc_id"] = range(len(arcs))
    route_file = folder / f"{tid}.route"
    route_ids = []
    if route_file.exists():
        route_ids = (pd.read_csv(route_file, header=None)[0]
                     .dropna().astype(int).tolist())
    return tid, track, nodes, arcs, route_ids


def point_to_segment_m(plat, plon, alat, alon, blat, blon):
    """Approximate point-to-segment distance on a local equirectangular
    projection (fine at these scales). Returns (distance, proj_lat,
    proj_lon, fraction along segment)."""
    lat0 = np.radians(plat)
    kx = 111_320.0 * np.cos(lat0)
    ky = 110_540.0
    px, py = plon * kx, plat * ky
    ax, ay = alon * kx, alat * ky
    bx, by = blon * kx, blat * ky
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    t = np.where(denom > 0,
                 np.clip(((px - ax) * abx + (py - ay) * aby) / np.where(denom > 0, denom, 1), 0, 1),
                 0.0)
    cx, cy = ax + t * abx, ay + t * aby
    dist = np.hypot(px - cx, py - cy)
    return dist, cy / ky, cx / kx, t


def generate_candidates(track, nodes, arcs):
    node_xy = nodes.set_index("node_id")[["latitude", "longitude"]]
    valid = arcs["from_node"].isin(node_xy.index) & arcs["to_node"].isin(node_xy.index)
    a = arcs[valid]
    geo = pd.DataFrame({
        "arc_id": a["arc_id"].values,
        "from_lat": node_xy.loc[a["from_node"], "latitude"].values,
        "from_lon": node_xy.loc[a["from_node"], "longitude"].values,
        "to_lat": node_xy.loc[a["to_node"], "latitude"].values,
        "to_lon": node_xy.loc[a["to_node"], "longitude"].values,
    })
    cand_rows = []
    lats = track["latitude"].values
    lons = track["longitude"].values
    oids = track["obs_id"].values
    for i in range(len(track)):
        dist, plat, plon, frac = point_to_segment_m(
            lats[i], lons[i],
            geo["from_lat"].values, geo["from_lon"].values,
            geo["to_lat"].values, geo["to_lon"].values)
        order = np.argsort(dist)[:N_CANDIDATES]
        for rank, j in enumerate(order):
            if dist[j] <= CANDIDATE_MAX_DIST_M:
                cand_rows.append({
                    "trip_id": track["trip_id"].iloc[0] if "trip_id" in track else None,
                    "obs_id": int(oids[i]), "obs_rank": rank,
                    "latitude": lats[i], "longitude": lons[i],
                    "seconds": track["seconds"].iloc[i],
                    "candidate_id": int(geo["arc_id"].iloc[j]),
                    "dist_to_segment_m": float(dist[j]),
                    "proj_lat": float(plat[j]), "proj_lon": float(plon[j]),
                    "frac_along": float(frac[j]),
                })
    return pd.DataFrame(cand_rows), geo


def main() -> None:
    report_rows = []
    for partition, base, trips in (("train", TRAIN_DIR, TRAIN_TRIPS),
                                   ("student_test", TEST_DIR, TEST_TRIPS)):
        if not base.exists():
            print(f"NOTE: {base} not found, skipping {partition}")
            continue
        out_dir = DATA_CLEAN / partition
        net_dir = DATA_CLEAN / "network"
        out_dir.mkdir(parents=True, exist_ok=True)
        net_dir.mkdir(parents=True, exist_ok=True)
        present = {p.name for p in base.iterdir() if p.is_dir()}
        for tid in trips:
            folder = base / tid
            if tid not in present:
                report_rows.append({"trip": tid, "partition": partition,
                                    "status": "MISSING FOLDER"})
                continue
            tid, track, nodes, arcs, route_ids = parse_trip(folder)
            report_rows.append({
                "trip": tid, "partition": partition, "status": "ok",
                "track_rows": len(track), "nodes": len(nodes),
                "arcs": len(arcs), "route_arcs": len(route_ids),
                "missing_lat": int(track["latitude"].isna().sum()),
                "track_span_s": float(track["seconds"].max() - track["seconds"].min()),
            })
            track["trip_id"] = tid
            cand, geo = generate_candidates(track, nodes, arcs)
            route_set = set(route_ids)
            cand["label"] = 0
            if route_set:
                # label the nearest in-route candidate per observation as 1
                in_route = cand["candidate_id"].isin(route_set)
                idx = cand[in_route].groupby("obs_id")["dist_to_segment_m"].idxmin()
                cand.loc[idx, "label"] = 1
            cand.to_parquet(out_dir / f"{tid}.parquet", index=False)
            geo.to_parquet(net_dir / f"{tid}.arcs.parquet", index=False)
    report = pd.DataFrame(report_rows)
    report.to_csv(RESULTS / "data_inspection.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
