"""Exploratory data analysis: trajectory plots, network visualisation,
missing-value summary and per-trip statistics.

Reads data/clean (produced by src.data.preprocess) and writes PNGs to
results/plots/ plus a summary CSV. Run after preprocessing, before
training - every figure is generated from the actual parsed data.

Outputs:
  results/plots/trajectory_<trip>.png   raw GPS trace of a trip
  results/plots/network_<trip>.png      road network + GPS overlay
  results/plots/obs_per_trip.png        observations per trip, by partition
  results/plots/speed_hist.png          speed distribution (features)
  results/eda_summary.csv               per-trip stats
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd

from src.config import DATA_CLEAN, RESULTS

PLOTS = RESULTS / "plots"
MAX_TRAJECTORY_PLOTS = 6   # a representative set, not all trips


def load_trip_obs(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    return (df.drop_duplicates("obs_id")
              .sort_values("obs_id"))


def plot_trajectory(obs, trip, partition):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(obs["longitude"], obs["latitude"], "-", lw=0.8,
            color="tab:blue", alpha=0.8, label="GPS trace")
    ax.scatter(obs["longitude"].iloc[0], obs["latitude"].iloc[0],
               color="green", zorder=3, label="start")
    ax.scatter(obs["longitude"].iloc[-1], obs["latitude"].iloc[-1],
               color="red", zorder=3, label="end")
    ax.set_title(f"GPS trajectory - trip {trip} ({partition})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / f"trajectory_{trip}.png", dpi=120)
    plt.close(fig)


def plot_network(obs, geo, trip, partition):
    # clip the network to the trajectory bounding box (the full network
    # has ~166k arcs - plotting all of them is slow and unreadable)
    lat_min, lat_max = obs["latitude"].min() - 0.01, obs["latitude"].max() + 0.01
    lon_min, lon_max = obs["longitude"].min() - 0.01, obs["longitude"].max() + 0.01
    in_box = geo[(geo["from_lat"].between(lat_min, lat_max)) &
                 (geo["from_lon"].between(lon_min, lon_max))]
    # stride-sample when the nearby subset is still huge (this trip
    # spans the whole network): keeps the figure readable and fast
    MAX_PLOT_ARCS = 20000
    if len(in_box) > MAX_PLOT_ARCS:
        step = len(in_box) // MAX_PLOT_ARCS + 1
        in_box = in_box.iloc[::step]
    segments = [[[a.from_lon, a.from_lat], [a.to_lon, a.to_lat]]
                for a in in_box.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(LineCollection(segments, colors="grey", linewidths=0.4))
    ax.autoscale()
    ax.plot(obs["longitude"], obs["latitude"], "-", lw=1.2,
            color="tab:blue", label="GPS trace")
    ax.set_title(f"Road network + trajectory - trip {trip} ({partition}) "
                 f"({len(in_box)} nearby arcs of {len(geo)})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / f"network_{trip}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = []
    plotted = 0
    for partition in ("train", "student_test"):
        for path in sorted((DATA_CLEAN / partition).glob("*.parquet")):
            obs = load_trip_obs(path)
            print(f"[eda] {partition}/{path.stem}: {len(obs)} obs", flush=True)
            span = obs["seconds"].max() - obs["seconds"].min() if "seconds" in obs else None
            rows.append({"trip": path.stem, "partition": partition,
                         "observations": len(obs), "span_s": span,
                         "missing_lat": int(obs["latitude"].isna().sum()),
                         "missing_lon": int(obs["longitude"].isna().sum())})
            if plotted < MAX_TRAJECTORY_PLOTS:
                plot_trajectory(obs, path.stem, partition)
                geo_path = DATA_CLEAN / "network" / f"{path.stem}.arcs.parquet"
                if geo_path.exists():
                    plot_network(obs, pd.read_parquet(geo_path),
                                 path.stem, partition)
                plotted += 1

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "eda_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    for partition, grp in summary.groupby("partition"):
        ax.bar(grp["trip"], grp["observations"], label=partition, alpha=0.8)
    ax.set_title("Observations per trip")
    ax.set_ylabel("GPS observations")
    ax.tick_params(axis="x", rotation=90, labelsize=5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "obs_per_trip.png", dpi=120)
    plt.close(fig)

    feat_path = RESULTS / "features.parquet"
    if feat_path.exists():
        feats = pd.read_parquet(feat_path)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(feats.loc[feats["obs_rank"] == 0, "speed_mps"], bins=50,
                color="tab:blue", alpha=0.8)
        ax.set_title("Speed distribution (per observation)")
        ax.set_xlabel("speed (m/s)"); ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(PLOTS / "speed_hist.png", dpi=120)
        plt.close(fig)

    print(summary.describe(include="all").to_string())
    print(f"\nplots written to {PLOTS}")


if __name__ == "__main__":
    main()
