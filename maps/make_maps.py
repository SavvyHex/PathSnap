"""Folium map generation plus model-result plots.

Folium maps (required by the brief): original GPS observations, road
network geometry, reference route and - after the HMM stage - predicted
route, as colour-coded layers with a legend and layer control. One map
per selected trip, saved as HTML to maps/.

Result plots (for the report): model comparison bars (CV Macro F1,
student-test Top-1 / Macro F1 / log loss) and the tuned-vs-untuned
table chart, all from the CSVs the training scripts actually wrote.
Run after the corresponding pipeline stage; every figure comes from
real output files, never hand-entered numbers.
"""
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import MeasureControl

from src.config import DATA_CLEAN, MAPS_DIR, RESULTS


def load_trip(trip, partition="student_test"):
    obs = pd.read_parquet(DATA_CLEAN / partition / f"{trip}.parquet")
    obs = obs.drop_duplicates("obs_id").sort_values("obs_id")
    geo = pd.read_parquet(DATA_CLEAN / "network" / f"{trip}.arcs.parquet")
    return obs, geo


def reference_route_points(trip, geo, partition="student_test"):
    """Reconstruct the reference route polyline from .route arc ids."""
    route_file = None
    for base in ("student_test", "train"):
        candidate = Path(__file__).resolve().parents[1] / base / trip / f"{trip}.route"
        if candidate.exists():
            route_file = candidate
            break
    if route_file is None:
        return []
    arc_ids = pd.read_csv(route_file, header=None)[0].astype(int).tolist()
    by_id = geo.set_index("arc_id")
    pts = []
    for aid in arc_ids:
        if aid in by_id.index:
            a = by_id.loc[aid]
            pts.append([a["from_lat"], a["from_lon"]])
            pts.append([a["to_lat"], a["to_lon"]])
    return pts


def trip_map(trip, partition="student_test", predicted_points=None):
    obs, geo = load_trip(trip, partition)
    m = folium.Map(location=[obs["latitude"].mean(), obs["longitude"].mean()],
                   zoom_start=13, tiles="OpenStreetMap")
    m.add_child(MeasureControl())

    # clip network to the GPS bounding box: the full network has ~166k
    # arcs, which would make the HTML unusable
    lat_min, lat_max = obs["latitude"].min() - 0.01, obs["latitude"].max() + 0.01
    lon_min, lon_max = obs["longitude"].min() - 0.01, obs["longitude"].max() + 0.01
    near = geo[(geo["from_lat"].between(lat_min, lat_max)) &
               (geo["from_lon"].between(lon_min, lon_max))]
    # cap rendered arcs: the full network is ~166k arcs and even the
    # nearby subset can be tens of thousands, which makes the HTML
    # unusable (87MB observed). A deterministic stride keeps coverage.
    MAX_NET_ARCS = 6000
    if len(near) > MAX_NET_ARCS:
        step = len(near) // MAX_NET_ARCS + 1
        near = near.iloc[::step]
    net = folium.FeatureGroup(name=f"Road network (near trace, {len(near)} arcs)")
    for a in near.itertuples():
        folium.PolyLine([[a.from_lat, a.from_lon], [a.to_lat, a.to_lon]],
                        color="grey", weight=1, opacity=0.4).add_to(net)
    net.add_to(m)

    gps = folium.FeatureGroup(name="GPS observations")
    folium.PolyLine(list(zip(obs["latitude"], obs["longitude"])),
                    color="blue", weight=2, opacity=0.7).add_to(gps)
    for _, r in obs.iloc[::5].iterrows():  # every 5th point keeps the HTML small
        folium.CircleMarker([r["latitude"], r["longitude"]], radius=2,
                            color="blue", fill=True, fill_opacity=0.7).add_to(gps)
    gps.add_to(m)

    ref = reference_route_points(trip, geo, partition)
    if ref:
        rfg = folium.FeatureGroup(name="Reference route")
        folium.PolyLine(ref, color="green", weight=4, opacity=0.9).add_to(rfg)
        rfg.add_to(m)

    if predicted_points:
        pfg = folium.FeatureGroup(name="Predicted route")
        folium.PolyLine(predicted_points, color="red", weight=4,
                        opacity=0.9).add_to(pfg)
        pfg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out = MAPS_DIR / f"map_{partition}_{trip}.html"
    m.save(str(out))
    return out


def plot_model_comparison():
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    plots_dir = RESULTS / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comp["model"], comp["cv_macro_f1_mean"],
           yerr=comp["cv_macro_f1_std"], capsize=4, color="tab:blue", alpha=0.85)
    ax.set_title("Cross-validation Macro F1 by model (group-aware folds)")
    ax.set_ylabel("Macro F1"); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(plots_dir / "cv_macro_f1.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(axes,
                              ["top1_accuracy", "macro_f1", "log_loss"],
                              ["Student-test Top-1 accuracy",
                               "Student-test Macro F1", "Student-test log loss"]):
        ax.bar(comp["model"], comp[col], color="tab:orange", alpha=0.85)
        ax.set_title(title); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(plots_dir / "student_test_metrics.png", dpi=120)
    plt.close(fig)

    tuned = RESULTS / "tuned_vs_untuned.csv"
    if tuned.exists():
        tv = pd.read_csv(tuned)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(tv))
        ax.bar([i - 0.2 for i in x], tv["top1_accuracy"], width=0.4,
               label="Top-1 accuracy")
        ax.bar([i + 0.2 for i in x], tv["macro_f1"], width=0.4, label="Macro F1")
        ax.set_xticks(list(x)); ax.set_xticklabels(tv["variant"])
        ax.set_title("Tuned vs untuned on student-test")
        ax.legend()
        fig.tight_layout(); fig.savefig(plots_dir / "tuned_vs_untuned.png", dpi=120)
        plt.close(fig)
    print("model comparison plots written")


def main() -> None:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for partition in ("student_test", "train"):
        d = DATA_CLEAN / partition
        if d.exists():
            for path in sorted(d.glob("*.parquet"))[:3]:
                print(f"[maps] rendering {partition}/{path.stem}", flush=True)
                made.append(str(trip_map(path.stem, partition)))
                print(f"[maps] done {partition}/{path.stem}", flush=True)
    print("maps written:", *made, sep="\n  ")
    if (RESULTS / "model_comparison.csv").exists():
        plot_model_comparison()


if __name__ == "__main__":
    main()
