"""Raw-data parsing, cleaning, candidate generation and labelling.

Dataset layout (confirmed): per trip folder under data/raw/train|student_test,
four files named <trip>.track / .nodes / .arcs / .route.

Format assumptions (sniffed and corrected on first real run):
  .track  one GPS observation per line: timestamp/order, lat, lon
          (extra columns preserved; delimiter and header auto-detected)
  .nodes  node_id, lat, lon
  .arcs   arc_id, from_node, to_node [, length ...]
  .route  reference arc ids, one per line or comma-separated

Outputs per trip into data/clean/<partition>/<trip>.parquet:
  one row per (observation, candidate) with candidate distance/bearing
  attributes and a binary label. The per-observation reference arc is
  approximated as the nearest candidate arc that belongs to the trip's
  reference route - the .route file gives the route as a whole, not a
  per-point match, so this proxy is stated in the report.

Also writes results/data_inspection.csv with row counts and missing
values per file, and data/clean/network/<trip>.arcs.parquet with arc
geometry for the HMM and Folium stages.
"""
import re

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW, DATA_CLEAN, RESULTS,
    TRAIN_DIR, TEST_DIR, TRAIN_TRIPS, TEST_TRIPS,
    N_CANDIDATES, CANDIDATE_MAX_DIST_M,
)

EARTH_R = 6_371_000.0


def sniff_read(path):
    """Read a delimited text file with unknown header/delimiter."""
    raw = path.read_text().strip().splitlines()
    if not raw:
        return pd.DataFrame()
    delim = "," if raw[0].count(",") >= raw[0].count(" ") else None
    first = raw[0].lower()
    has_header = bool(re.search(r"[a-z]", first))
    df = pd.read_csv(path, sep=delim or r"\s+", engine="python",
                     header=0 if has_header else None)
    if not has_header:
        df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_to_segment_m(plat, plon, alat, alon, blat, blon):
    """Approximate point-to-segment distance on a local equirectangular
    projection (fine at hackathon scales). Returns (distance, proj_lat,
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


def parse_trip(folder):
    tid = folder.name
    track = sniff_read(folder / f"{tid}.track")
    nodes = sniff_read(folder / f"{tid}.nodes")
    arcs = sniff_read(folder / f"{tid}.arcs")
    route_file = folder / f"{tid}.route"
    route_ids = []
    if route_file.exists():
        route_ids = re.findall(r"\d+", route_file.read_text())
    return tid, track, nodes, arcs, route_ids


def standardise(tid, track, nodes, arcs):
    """Rename guessed columns to canonical names. VERIFY on real data."""
    def ren(df, mapping):
        cols = {c: mapping[i] for i, c in enumerate(df.columns[: len(mapping)])}
        return df.rename(columns=cols)

    # guesses: track = [id?, time?, lat, lon, ...]; take last two numeric
    # as lat/lon if names are absent - adjusted after first inspection.
    if "lat" not in track.columns.str.lower().tolist():
        track = ren(track, ["obs_id", "timestamp", "latitude", "longitude"])
    if "lat" not in nodes.columns.str.lower().tolist():
        nodes = ren(nodes, ["node_id", "latitude", "longitude"])
    if "from_node" not in arcs.columns.str.lower().tolist():
        arcs = ren(arcs, ["arc_id", "from_node", "to_node"])
    track["trip_id"] = tid
    return track, nodes, arcs


def generate_candidates(track, nodes, arcs):
    node_xy = nodes.set_index("node_id")[["latitude", "longitude"]]
    arc_rows = []
    for _, a in arcs.iterrows():
        try:
            alat, alon = node_xy.loc[a["from_node"]]
            blat, blon = node_xy.loc[a["to_node"]]
        except KeyError:
            continue
        arc_rows.append((a["arc_id"], alat, alon, blat, blon))
    cand_rows = []
    for i, o in track.iterrows():
        dists = []
        for arc_id, alat, alon, blat, blon in arc_rows:
            d, plat, plon, frac = point_to_segment_m(
                o["latitude"], o["longitude"], alat, alon, blat, blon)
            dists.append((d, arc_id, plat, plon, frac))
        dists.sort(key=lambda r: r[0])
        for rank, (d, arc_id, plat, plon, frac) in enumerate(dists[:N_CANDIDATES]):
            if d <= CANDIDATE_MAX_DIST_M:
                cand_rows.append({
                    "trip_id": o["trip_id"], "obs_id": i, "obs_rank": rank,
                    "latitude": o["latitude"], "longitude": o["longitude"],
                    "candidate_id": arc_id, "dist_to_segment_m": d,
                    "proj_lat": plat, "proj_lon": plon, "frac_along": frac,
                })
    return pd.DataFrame(cand_rows), arc_rows


def main() -> None:
    report_rows = []
    for partition, base, trips in (("train", TRAIN_DIR, TRAIN_TRIPS),
                                   ("student_test", TEST_DIR, TEST_TRIPS)):
        out_dir = DATA_CLEAN / partition
        net_dir = DATA_CLEAN / "network"
        out_dir.mkdir(parents=True, exist_ok=True)
        net_dir.mkdir(parents=True, exist_ok=True)
        for tid in trips:
            folder = base / tid
            if not folder.exists():
                report_rows.append({"trip": tid, "partition": partition,
                                    "status": "MISSING FOLDER"})
                continue
            tid, track, nodes, arcs, route_ids = parse_trip(folder)
            report_rows.append({
                "trip": tid, "partition": partition, "status": "ok",
                "track_rows": len(track), "nodes": len(nodes),
                "arcs": len(arcs), "route_arcs": len(route_ids),
                "missing_lat": int(track.iloc[:, -2].isna().sum()) if len(track) else "n/a",
            })
            track, nodes, arcs = standardise(tid, track, nodes, arcs)
            cand, arc_rows = generate_candidates(track, nodes, arcs)
            route_set = set(route_ids)
            cand["label"] = 0
            if route_set:
                # label the nearest in-route candidate per observation as 1
                in_route = cand["candidate_id"].astype(str).isin(route_set)
                idx = cand[in_route].groupby("obs_id")["dist_to_segment_m"].idxmin()
                cand.loc[idx, "label"] = 1
            cand.to_parquet(out_dir / f"{tid}.parquet", index=False)
            pd.DataFrame(arc_rows, columns=["arc_id", "from_lat", "from_lon",
                                            "to_lat", "to_lon"]).to_parquet(
                net_dir / f"{tid}.arcs.parquet", index=False)
    report = pd.DataFrame(report_rows)
    report.to_csv(RESULTS / "data_inspection.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
