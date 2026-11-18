
"""Raw-data parsing, cleaning, candidate generation and labelling.

Input layout:
  train/<trip>/<trip>.{track,nodes,arcs,route}
  student_test/<trip>/<trip>.{track,nodes,arcs,route}

File formats:
  .track: longitude, latitude, seconds
  .nodes: longitude, latitude (node ID = zero-based line number)
  .arcs: from_node, to_node (arc ID = zero-based line number)
  .route: one reference arc ID per line

Outputs:
  data/clean/train/<trip>.parquet
  data/clean/student_test/<trip>.parquet
  data/clean/network/<trip>.arcs.parquet
  results/data_inspection.csv

The .route file is route-level ground truth, not pointwise ground
truth. Labels are therefore proxies: for each observation, the
nearest candidate belonging to the reference route is labelled 1.

Network files preserve directed endpoints and geodesic arc lengths
in metres for the HMM transition stage.
"""

import numpy as np
import pandas as pd

from src.config import (
    DATA_CLEAN,
    RESULTS,
    TRAIN_DIR,
    TEST_DIR,
    TRAIN_TRIPS,
    TEST_TRIPS,
    N_CANDIDATES,
    CANDIDATE_MAX_DIST_M,
)

EARTH_R = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in metres."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    )

    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def parse_trip(folder):
    """Read the four raw files belonging to one trip."""

    tid = folder.name

    track = pd.read_csv(
        folder / f"{tid}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "seconds"],
    )
    track["obs_id"] = range(len(track))

    nodes = pd.read_csv(
        folder / f"{tid}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"],
    )
    nodes["node_id"] = range(len(nodes))

    arcs = pd.read_csv(
        folder / f"{tid}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
    )
    arcs["arc_id"] = range(len(arcs))

    route_file = folder / f"{tid}.route"

    route_ids = []
    if route_file.exists():
        route_ids = (
            pd.read_csv(route_file, header=None)[0]
            .dropna()
            .astype(int)
            .tolist()
        )

    return tid, track, nodes, arcs, route_ids


def point_to_segment_m(
    plat, plon,
    alat, alon,
    blat, blon,
):
    """Calculate point-to-segment distance in metres.

    Uses a local equirectangular projection.

    Returns:
        distance: distance from point to segment
        proj_lat: projected latitude
        proj_lon: projected longitude
        fraction: position along segment (0 to 1)
    """

    lat0 = np.radians(plat)

    kx = 111_320.0 * np.cos(lat0)
    ky = 110_540.0

    px = plon * kx
    py = plat * ky

    ax = alon * kx
    ay = alat * ky

    bx = blon * kx
    by = blat * ky

    abx = bx - ax
    aby = by - ay

    denom = abx * abx + aby * aby

    t = np.where(
        denom > 0,
        np.clip(
            ((px - ax) * abx + (py - ay) * aby)
            / np.where(denom > 0, denom, 1),
            0,
            1,
        ),
        0.0,
    )

    cx = ax + t * abx
    cy = ay + t * aby

    dist = np.hypot(px - cx, py - cy)

    return dist, cy / ky, cx / kx, t


def generate_candidates(track, nodes, arcs):
    """Generate nearest road-segment candidates for each GPS observation.

    Also creates the directed road-network geometry needed by the HMM.
    """

    node_xy = nodes.set_index("node_id")[
        ["latitude", "longitude"]
    ]

    # Remove arcs with invalid endpoint node IDs.
    valid = (
        arcs["from_node"].isin(node_xy.index)
        & arcs["to_node"].isin(node_xy.index)
    )

    a = arcs[valid].copy()

    # Preserve directed network information.
    geo = pd.DataFrame({
        "arc_id": a["arc_id"].values,
        "from_node": a["from_node"].values,
        "to_node": a["to_node"].values,
<<<<<<< Updated upstream
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
=======

        "from_lat": node_xy.loc[
            a["from_node"], "latitude"
        ].values,

        "from_lon": node_xy.loc[
            a["from_node"], "longitude"
        ].values,

        "to_lat": node_xy.loc[
            a["to_node"], "latitude"
        ].values,

        "to_lon": node_xy.loc[
            a["to_node"], "longitude"
        ].values,
    })

    # Calculate geodesic length of each directed arc.
    geo["length_m"] = haversine_m(
        geo["from_lat"].values,
        geo["from_lon"].values,
        geo["to_lat"].values,
        geo["to_lon"].values,
    )

    cand_rows = []

    lats = track["latitude"].values
    lons = track["longitude"].values
    oids = track["obs_id"].values

    # Generate candidates independently for every GPS observation.
    for i in range(len(track)):

        dist, plat, plon, frac = point_to_segment_m(
            lats[i],
            lons[i],
            geo["from_lat"].values,
            geo["from_lon"].values,
            geo["to_lat"].values,
            geo["to_lon"].values,
        )

        order = np.argsort(dist)[:N_CANDIDATES]

        for rank, j in enumerate(order):

            if dist[j] <= CANDIDATE_MAX_DIST_M:

                cand_rows.append({
                    "trip_id": (
                        track["trip_id"].iloc[0]
                        if "trip_id" in track
                        else None
                    ),

                    "obs_id": int(oids[i]),
                    "obs_rank": rank,

                    "latitude": lats[i],
                    "longitude": lons[i],
                    "seconds": track["seconds"].iloc[i],

                    "candidate_id": int(
                        geo["arc_id"].iloc[j]
                    ),

                    "dist_to_segment_m": float(dist[j]),

                    "proj_lat": float(plat[j]),
                    "proj_lon": float(plon[j]),

                    "frac_along": float(frac[j]),
                })

    candidates = pd.DataFrame(cand_rows)

    return candidates, geo
>>>>>>> Stashed changes


def main():
    """Preprocess training and student-test trips."""

    report_rows = []

    partitions = (
        ("train", TRAIN_DIR, TRAIN_TRIPS),
        ("student_test", TEST_DIR, TEST_TRIPS),
    )

    for partition, base, trips in partitions:

        if not base.exists():
            print(
                f"NOTE: {base} not found, skipping {partition}"
            )
            continue

        out_dir = DATA_CLEAN / partition
        net_dir = DATA_CLEAN / "network"

        out_dir.mkdir(parents=True, exist_ok=True)
        net_dir.mkdir(parents=True, exist_ok=True)
<<<<<<< Updated upstream
        present = sorted(p.name for p in base.iterdir() if p.is_dir())
        if partition == "train":
            # the brief fixes the train set at 80 trips; the exact ids are
            # discovered from disk (the 1-99 numbering has gaps)
            trips = present
            if len(present) != 80:
                report_rows.append({"trip": "-", "partition": partition,
                                    "status": f"NOTE: brief expects 80 train trips, found {len(present)}"})
        else:
            # test trip ids are explicitly named by the team; absence is loud
            for tid in trips:
                if tid not in present:
                    report_rows.append({"trip": tid, "partition": partition,
                                        "status": "MISSING FOLDER"})
            trips = [t for t in trips if t in present]
        for tid in trips:
            folder = base / tid
            print(f"[preprocess] {partition}/{tid} ...", flush=True)
            tid, track, nodes, arcs, route_ids = parse_trip(folder)
=======

        present = {
            p.name for p in base.iterdir() if p.is_dir()
        }

        expected = set(trips)

        # Record unexpected folders without processing them.
        for extra in sorted(present - expected):

>>>>>>> Stashed changes
            report_rows.append({
                "trip": extra,
                "partition": partition,
                "status": "UNEXPECTED FOLDER",
            })

        # Process only the configured trip IDs.
        for trip_id in trips:

            folder = base / trip_id

            if trip_id not in present:

                report_rows.append({
                    "trip": trip_id,
                    "partition": partition,
                    "status": "MISSING FOLDER",
                })

                continue

            print(
                f"[preprocess] {partition}/{trip_id} ...",
                flush=True,
            )

            tid, track, nodes, arcs, route_ids = parse_trip(
                folder
            )

            report_rows.append({
                "trip": tid,
                "partition": partition,
                "status": "ok",
                "track_rows": len(track),
                "nodes": len(nodes),
                "arcs": len(arcs),
                "route_arcs": len(route_ids),
                "missing_lat": int(
                    track["latitude"].isna().sum()
                ),
                "track_span_s": float(
                    track["seconds"].max()
                    - track["seconds"].min()
                ),
            })

            track["trip_id"] = tid

            candidates, geo = generate_candidates(
                track,
                nodes,
                arcs,
            )

            # Route-level reference labels.
            route_set = set(route_ids)

            candidates["label"] = 0

            if route_set and not candidates.empty:

                in_route = candidates[
                    "candidate_id"
                ].isin(route_set)

                idx = (
                    candidates[in_route]
                    .groupby("obs_id")[
                        "dist_to_segment_m"
                    ]
                    .idxmin()
                )

                candidates.loc[idx, "label"] = 1

            # Save candidate features.
            candidates.to_parquet(
                out_dir / f"{tid}.parquet",
                index=False,
            )

            # Save directed road-network geometry.
            geo.to_parquet(
                net_dir / f"{tid}.arcs.parquet",
                index=False,
            )

            print(
                f"[preprocess] {partition}/{tid} done: "
                f"{len(track)} observations, "
                f"{len(candidates)} candidates, "
                f"{len(geo)} arcs",
                flush=True,
            )

    # Write inspection report.
    report = pd.DataFrame(report_rows)

    RESULTS.mkdir(parents=True, exist_ok=True)

    report.to_csv(
        RESULTS / "data_inspection.csv",
        index=False,
    )

    print("\n===== PREPROCESSING REPORT =====")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

