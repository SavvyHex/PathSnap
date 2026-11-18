"""Derived candidate features: within-observation geometry and heading alignment.

All columns are computed from fields already present in results/features.parquet
(candidate distances, projection points, GPS coordinates, motion features), so no
raw data or candidate regeneration is needed. Used by build_features (full rebuild)
and by `python -m src.features.derived` (cheap in-place augmentation of an existing
features.parquet).

Heading-alignment note: the heading features compare the GPS movement bearing from
observation t to t+1 with the bearing implied by the same candidate arc's projection
points at t and t+1. This uses one-step lookahead geometry, which is consistent with
the batch Viterbi decoder (it already conditions on the whole trip); the features
would not be available in a streaming/online setting. They carry no label
information: only geometry. Observations without a matching next-observation
candidate (trip ends, gaps) get neutral fills and has_heading=0.
"""
import numpy as np
import pandas as pd

from src.config import COL_TRIP, RESULTS

BASE_COLS = ["obs_rank", "dist_to_segment_m", "frac_along",
             "speed_mps", "accel_mps2", "turning_angle_deg"]
DERIVED_COLS = ["n_cand", "dist_minus_min", "dist_ratio_min", "dist_rank", "is_closest",
                "log1p_dist", "dist_sq", "cand_frac", "abs_turn", "speed_sq",
                "abs_accel", "frac_edge",
                "heading_err", "heading_err_bi", "has_heading",
                "cos_heading", "cos_heading_bi"]


def _bearing(lat1, lon1, lat2, lon2):
    """Forward bearing in degrees [0, 360); equirectangular approximation,
    accurate for the short hops between consecutive observations."""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    y = np.radians(lon2 - lon1) * np.cos((lat1 + lat2) / 2)
    return (np.degrees(np.arctan2(y, lat2 - lat1)) + 360.0) % 360.0


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby([COL_TRIP, "obs_id"])["dist_to_segment_m"]
    df["n_cand"] = g.transform("size").astype(np.float32)
    dmin = g.transform("min")
    df["dist_minus_min"] = (df["dist_to_segment_m"] - dmin).astype(np.float32)
    df["dist_ratio_min"] = (df["dist_to_segment_m"] / (dmin + 1e-6)).astype(np.float32)
    df["dist_rank"] = g.rank(method="average").astype(np.float32)
    df["is_closest"] = (df["dist_to_segment_m"] == dmin).astype(np.float32)
    df["log1p_dist"] = np.log1p(df["dist_to_segment_m"]).astype(np.float32)
    df["dist_sq"] = (df["dist_to_segment_m"] ** 2).astype(np.float32)
    rmax = df.groupby(COL_TRIP)["obs_rank"].transform("max")
    df["cand_frac"] = (df["obs_rank"] / rmax.clip(lower=1)).astype(np.float32)
    df["abs_turn"] = df["turning_angle_deg"].abs().astype(np.float32)
    df["speed_sq"] = (df["speed_mps"] ** 2).astype(np.float32)
    df["abs_accel"] = df["accel_mps2"].abs().astype(np.float32)
    df["frac_edge"] = (df["frac_along"] - 0.5).abs().astype(np.float32)

    key = [COL_TRIP, "obs_id", "candidate_id"]
    nxt = df[key + ["proj_lat", "proj_lon", "latitude", "longitude"]].copy()
    nxt["obs_id"] = nxt["obs_id"] - 1  # align the NEXT observation onto the current one
    nxt = nxt.rename(columns={"proj_lat": "proj_lat_n", "proj_lon": "proj_lon_n",
                              "latitude": "lat_n", "longitude": "lon_n"})
    m = df.merge(nxt, on=key, how="left")
    have = m["proj_lat_n"].notna() & m["lat_n"].notna()
    gps_b = _bearing(m["latitude"], m["longitude"], m["lat_n"], m["lon_n"])
    arc_b = _bearing(m["proj_lat"], m["proj_lon"], m["proj_lat_n"], m["proj_lon_n"])
    err = np.abs(gps_b - arc_b)
    err = np.minimum(err, 360.0 - err)
    m["heading_err"] = np.where(have, err, np.nan)
    m["heading_err_bi"] = np.where(have, np.minimum(err, 180.0 - err), np.nan)
    m["has_heading"] = have.astype(np.float32)
    m["cos_heading"] = np.cos(np.radians(m["heading_err"].fillna(90.0))).astype(np.float32)
    m["cos_heading_bi"] = np.cos(np.radians(m["heading_err_bi"].fillna(45.0))).astype(np.float32)
    m["heading_err"] = m["heading_err"].fillna(90.0).astype(np.float32)
    m["heading_err_bi"] = m["heading_err_bi"].fillna(45.0).astype(np.float32)
    return m


def main() -> None:
    """Add derived columns to an existing results/features.parquet in place
    (idempotent: columns are recomputed, not duplicated)."""
    path = RESULTS / "features.parquet"
    df = pd.read_parquet(path)
    out = add_derived_features(df.drop(columns=[c for c in DERIVED_COLS if c in df.columns]))
    out.to_parquet(path, index=False)
    print(f"features.parquet augmented: {len(out)} rows, "
          f"{len(out.columns)} columns (+{len(DERIVED_COLS)} derived)")


if __name__ == "__main__":
    main()
