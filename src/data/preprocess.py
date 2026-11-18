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
    # Index padded arc bounding boxes; exact projection still determines rank.
    from collections import defaultdict
    grid = defaultdict(list)
    size = .003
    max_lat = np.abs(track["latitude"]).max()
    pad_y = CANDIDATE_MAX_DIST_M / 110540.
    pad_x = CANDIDATE_MAX_DIST_M / (111320. * np.cos(np.radians(max_lat)))
    coords = geo[["from_lat", "from_lon", "to_lat", "to_lon"]].to_numpy()
    for j, (y, x, v, u) in enumerate(coords):
        for gx in range(int(np.floor((min(x,u)-pad_x)/size)), int(np.floor((max(x,u)+pad_x)/size))+1):
            for gy in range(int(np.floor((min(y,v)-pad_y)/size)), int(np.floor((max(y,v)+pad_y)/size))+1):
                grid[gx, gy].append(j)
    tracks = track.copy()
    tracks["gx"] = np.floor(tracks.longitude/size).astype(int)
    tracks["gy"] = np.floor(tracks.latitude/size).astype(int)
    frames = []
    for key, observations in tracks.groupby(["gx","gy"]):
        ids = np.asarray(grid.get(key, []), dtype=int)
        if not len(ids):
            continue
        for start in range(0,len(observations),128):
            obs = observations.iloc[start:start+128]
            dist, lat, lon, frac = point_to_segment_m(obs.latitude.to_numpy()[:,None],
                obs.longitude.to_numpy()[:,None], *coords[ids].T)
            order = np.argsort(dist,axis=1,kind="stable")[:,:N_CANDIDATES]
            row, rank = np.where(np.take_along_axis(dist,order,axis=1)<=CANDIDATE_MAX_DIST_M)
            col = order[row,rank]
            frame = obs.iloc[row][["trip_id","obs_id","latitude","longitude","seconds"]].copy()
            frame["obs_rank"] = rank
            frame["candidate_id"] = geo.arc_id.to_numpy()[ids[col]]
            frame["dist_to_segment_m"] = dist[row,col]
            frame["proj_lat"] = lat[row,col]
            frame["proj_lon"] = lon[row,col]
            frame["frac_along"] = frac[row,col]
            frames.append(frame)
    return pd.concat(frames,ignore_index=True).sort_values(["obs_id","obs_rank"]).reset_index(drop=True), geo


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
        expected = set(trips)
        for extra in sorted(present - expected):
            report_rows.append({"trip": extra, "partition": partition,
                                "status": "UNEXPECTED FOLDER (not in expected id list)"})
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
