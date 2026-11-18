"""Feature engineering for (GPS observation, candidate segment) pairs.

The classifier scores each candidate of each observation; the label is
1 for the candidate whose segment id equals the reference matched
segment, else 0. Per-observation Top-1 prediction = highest scoring
candidate.

Provided candidate attributes (distance to segment, bearing difference
...) are passed through; motion features (speed, acceleration, turning
angle) are derived from the GPS trace itself. Adjust to the real
columns once the schema is confirmed.
"""
import numpy as np
import pandas as pd

from src.config import (
    DATA_CLEAN, RESULTS,
    COL_TRIP, COL_TIME, COL_LAT, COL_LON, COL_CAND, COL_LABEL,
)

EARTH_R = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def motion_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Speed / acceleration / turning angle per observation.

    .track files carry an observation order; if no usable timestamp is
    present the caller passes a RangeIndex so dt=1 everywhere and speed
    is expressed in metres-per-sample (documented in the report).
    """
    out = pd.DataFrame(index=obs.index)
    ts = pd.to_numeric(pd.to_datetime(obs["timestamp"], errors="coerce"), errors="coerce")
    dt = pd.Series(ts).diff().fillna(1.0).replace(0, np.nan).fillna(1.0)
    dist = haversine_m(obs["latitude"].shift(), obs["longitude"].shift(),
                       obs["latitude"], obs["longitude"]).fillna(0.0)
    out["speed_mps"] = (dist / dt).clip(0, 80)
    out["accel_mps2"] = (out["speed_mps"].diff() / dt).clip(-20, 20)
    dlat = obs["latitude"].diff().fillna(0.0)
    dlon = obs["longitude"].diff().fillna(0.0)
    bearing = np.degrees(np.arctan2(dlon, dlat))
    out["turning_angle_deg"] = bearing.diff().abs().clip(0, 180).fillna(0.0)
    return out.fillna(0.0)


def build_trip(obs: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Join candidates with motion features and the binary label."""
    df = cand.copy()
    for col in ("speed_mps", "accel_mps2", "turning_angle_deg"):
        df[col] = obs[col].reindex(df.index).values if col in obs else 0.0
    if COL_LABEL in obs.columns:
        df["label"] = (df[COL_CAND] == obs[COL_LABEL].reindex(df.index).values).astype(int)
    return df


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
                     .sort_values("obs_id")[["obs_id", "latitude", "longitude"]]
                     .reset_index(drop=True))
            obs["timestamp"] = pd.RangeIndex(len(obs))  # uniform sampling proxy
            mf = motion_features(obs)
            mf["obs_id"] = obs["obs_id"].values
            df = df.merge(mf, on="obs_id", how="left")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(RESULTS / "features.parquet", index=False)
    print(f"features.parquet: {len(out)} rows, "
          f"{out['obs_id'].nunique()} observations, "
          f"{out['trip_id'].nunique()} trips")


if __name__ == "__main__":
    main()
