"""Feature engineering for (GPS observation, candidate segment) pairs.

The classifier scores each candidate of each observation; the label is
1 for the candidate nearest to the observation among those on the
reference route, else 0 (see src/data/preprocess.py for the labelling
rule and its caveat). Per-observation Top-1 prediction = highest
scoring candidate.

Candidate attributes from preprocessing (distance to segment,
projection fraction) are passed through; motion features (speed,
acceleration, turning angle) are derived from the GPS trace using the
real .track seconds column.
"""
import numpy as np
import pandas as pd

from src.config import DATA_CLEAN, RESULTS

EARTH_R = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def motion_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Speed / acceleration / turning angle per observation.

    Uses the .track seconds column when present (real sampling rate);
    falls back to observation order (uniform-time proxy, documented).
    """
    out = pd.DataFrame(index=obs.index)
    if "seconds" in obs.columns:
        dt = obs["seconds"].diff().replace(0, np.nan).fillna(1.0)
    else:
        dt = pd.Series(1.0, index=obs.index)
    dist = haversine_m(obs["latitude"].shift(), obs["longitude"].shift(),
                       obs["latitude"], obs["longitude"]).fillna(0.0)
    out["speed_mps"] = (dist / dt).clip(0, 80)
    out["accel_mps2"] = (out["speed_mps"].diff() / dt).clip(-20, 20)
    dlat = obs["latitude"].diff().fillna(0.0)
    dlon = obs["longitude"].diff().fillna(0.0)
    bearing = np.degrees(np.arctan2(dlon, dlat))
    out["turning_angle_deg"] = bearing.diff().abs().clip(0, 180).fillna(0.0)
    return out.fillna(0.0)


def main() -> None:
    """Concatenate per-trip candidate files, add motion features and
    write results/features.parquet with a partition column."""
    frames = []
    for partition in ("train", "student_test"):
        for path in sorted((DATA_CLEAN / partition).glob("*.parquet")):
            df = pd.read_parquet(path)
            df["partition"] = partition
            # motion features from the observation sequence of this trip
            obs = (df.drop_duplicates("obs_id")
                     .sort_values("obs_id")[["obs_id", "latitude", "longitude", "seconds"]]
                     .reset_index(drop=True))
            mf = motion_features(obs)
            mf["obs_id"] = obs["obs_id"].values
            df = df.merge(mf, on="obs_id", how="left")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(RESULTS / "features.parquet", index=False)
    print(f"features.parquet: {len(out)} rows, "
          f"{out.groupby(['trip_id','obs_id']).ngroups} observations, "
          f"{out['trip_id'].nunique()} trips")


if __name__ == "__main__":
    main()
