<<<<<<< Updated upstream
"""Compare classifier-only and road-network HMM route reconstruction."""
import json

=======

"""Compare classifier-only predictions with road-network HMM decoding."""

import argparse
import json
from pathlib import Path

>>>>>>> Stashed changes
import joblib
import numpy as np
import pandas as pd

<<<<<<< Updated upstream
from src.config import ARTIFACTS, DATA_CLEAN, RESULTS, TEST_DIR, HMM_BETA, COL_TRIP
from src.data.preprocess import parse_trip
=======
from src.config import ARTIFACTS, RESULTS, COL_TRIP, HMM_BETA
>>>>>>> Stashed changes
from src.features.build_features import haversine_m
from src.hmm.road_graph import RoadGraph
from src.hmm.viterbi import viterbi


<<<<<<< Updated upstream
def _route_continuity(states, positions, arc_ids, fracs, gps_dists, graph, beta):
    """Fraction of decoded consecutive observations with feasible road paths."""
    pairs = [(t, int(state)) for t, state in enumerate(states) if state >= 0]
    if len(pairs) < 2:
        return float("nan"), 0
    good = 0
    for (t0, i0), (t1, i1) in zip(pairs, pairs[1:]):
        road_dist = graph.transition_distances(
            [arc_ids[t0][i0]], [fracs[t0][i0]], [arc_ids[t1][i1]], [fracs[t1][i1]],
        )[0, 0]
        gps_gap = float(np.sum(gps_dists[t0:t1]))
        # Allow GPS noise while penalizing implausible detours and disconnected paths.
        if np.isfinite(road_dist) and road_dist <= gps_gap + max(2 * beta, 0.5 * gps_gap):
            good += 1
    return good / (len(pairs) - 1), len(pairs) - 1


def compare_trip(candidates, track, model, X_cols, graph, beta=HMM_BETA):
    """Score classifier and HMM on one trip, preserving observations without candidates."""
    candidates = candidates.sort_values(["obs_id", "obs_rank"]).reset_index(drop=True)
    track = track.sort_values("obs_id").reset_index(drop=True)
    n_obs = len(track)
    candidates["score"] = model.predict_proba(candidates[X_cols])[:, 1]

    groups = {int(oid): group.reset_index(drop=True)
              for oid, group in candidates.groupby("obs_id", sort=True)}
    probs, arc_ids, fracs, classifier = [], [], [], []
    for t in range(n_obs):
        group = groups.get(t)
        if group is None or group.empty:
            probs.append(np.array([], dtype=float)); arc_ids.append(np.array([], dtype=int))
            fracs.append(np.array([], dtype=float)); classifier.append(-1)
            continue
        score = np.clip(group.score.to_numpy(dtype=float), 1e-12, None)
        probs.append(score / score.sum())
        arc_ids.append(group.candidate_id.to_numpy(dtype=int))
        fracs.append(group.frac_along.to_numpy(dtype=float))
        classifier.append(int(np.argmax(score)))

    lat, lon = track.latitude.to_numpy(), track.longitude.to_numpy()
    gps_dists = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]).tolist()

    def candidate_distances(t_from, t_to):
        return graph.transition_distances(arc_ids[t_from], fracs[t_from],
                                         arc_ids[t_to], fracs[t_to])

    decoded = viterbi(probs, gps_dists, candidate_distances, beta=beta)
    eval_obs = [t for t, group in groups.items() if (group.label == 1).any()]
    clf_top1 = float(np.mean([groups[t].iloc[classifier[t]].label == 1 for t in eval_obs])) if eval_obs else float("nan")
    hmm_top1 = float(np.mean([groups[t].iloc[decoded.sequence[t]].label == 1
                              for t in eval_obs if decoded.sequence[t] >= 0])) if eval_obs else float("nan")
    clf_cont, clf_transitions = _route_continuity(classifier, range(n_obs), arc_ids, fracs, gps_dists, graph, beta)
    hmm_cont, hmm_transitions = _route_continuity(decoded.sequence, range(n_obs), arc_ids, fracs, gps_dists, graph, beta)
    return {
        "trip": str(candidates[COL_TRIP].iloc[0]), "clf_top1": clf_top1, "hmm_top1": hmm_top1,
        "clf_continuity": clf_cont, "hmm_continuity": hmm_cont,
        "n_obs": n_obs, "n_candidate_obs": len(groups), "n_imputed": len(decoded.imputed),
        "clf_transitions": clf_transitions, "hmm_transitions": hmm_transitions,
    }


def main() -> None:
    df = pd.read_parquet(RESULTS / "features.parquet")
    X_cols = json.loads((ARTIFACTS / "feature_columns.json").read_text())
    model = joblib.load(ARTIFACTS / "model_tuned.joblib")
    test = df[df["partition"] == "student_test"]
    rows = []
    for trip_id, candidates in test.groupby(COL_TRIP):
        trip_id = str(trip_id)
        folder = TEST_DIR / trip_id
        _, track, _, _, _ = parse_trip(folder)
        track["obs_id"] = np.arange(len(track))
        arcs_path = DATA_CLEAN / "network" / f"{trip_id}.arcs.parquet"
        graph = RoadGraph(pd.read_parquet(arcs_path))
        rows.append(compare_trip(candidates, track, model, X_cols, graph))
    out = pd.DataFrame(rows)
    if not out.empty:
        def weighted_mean(values, weights):
            values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
            valid = np.isfinite(values) & (weights > 0)
            return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")

        out.loc[len(out)] = {
            "trip": "overall",
            "clf_top1": weighted_mean(out.clf_top1, out.n_candidate_obs),
            "hmm_top1": weighted_mean(out.hmm_top1, out.n_candidate_obs),
            "clf_continuity": weighted_mean(out.clf_continuity, out.clf_transitions),
            "hmm_continuity": weighted_mean(out.hmm_continuity, out.hmm_transitions),
            "n_obs": out.n_obs.sum(), "n_candidate_obs": out.n_candidate_obs.sum(),
            "n_imputed": out.n_imputed.sum(),
            "clf_transitions": out.clf_transitions.sum(), "hmm_transitions": out.hmm_transitions.sum(),
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "hmm_vs_classifier.csv", index=False)
    print(out.to_string(index=False))
=======
FEATURES_PATH = RESULTS / "features.parquet"
MODEL_PATH = ARTIFACTS / "model_tuned.joblib"
FEATURE_COLUMNS_PATH = ARTIFACTS / "feature_columns.json"

NETWORK_DIR = Path("data/clean/network")


def compare_trip(df_obs, model, X_cols, beta=HMM_BETA):
    """Evaluate one trip using classifier-only and HMM predictions."""

    df_obs = df_obs.copy()

    df_obs["obs_id"] = df_obs["obs_id"].astype(int)

    # Preserve candidate ordering within each GPS observation.
    df_obs = df_obs.sort_values(
        ["obs_id", "obs_rank"]
    ).reset_index(drop=True)

    # Get probability of class 1 from the trained classifier.
    positive_class_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(
        df_obs[X_cols]
    )[:, positive_class_index]

    df_obs["score"] = probabilities

    # Group candidates by GPS observation.
    groups = [
        (int(obs_id), group.reset_index(drop=True))
        for obs_id, group in df_obs.groupby("obs_id", sort=True)
    ]

    obs_ids = [item[0] for item in groups]
    candidate_groups = [item[1] for item in groups]

    n_obs = len(groups)

    # ---------------------------------------------------------
    # 1. Classifier-only prediction
    # ---------------------------------------------------------

    classifier_correct = 0
    evaluation_observations = 0

    for group in candidate_groups:

        # Evaluate only observations with at least one positive
        # proxy label.
        if not (group["label"] == 1).any():
            continue

        evaluation_observations += 1

        best_index = group["score"].idxmax()

        predicted_label = int(
            group.loc[best_index, "label"]
        )

        classifier_correct += int(predicted_label == 1)

    classifier_accuracy = (
        classifier_correct / evaluation_observations
        if evaluation_observations
        else float("nan")
    )

    # ---------------------------------------------------------
    # 2. Load this trip's directed road network
    # ---------------------------------------------------------

    trip_id = str(df_obs[COL_TRIP].iloc[0]).zfill(8)

    network_path = (
        NETWORK_DIR / f"{trip_id}.arcs.parquet"
    )

    if not network_path.exists():
        raise FileNotFoundError(
            f"Road network not found: {network_path}"
        )

    network_df = pd.read_parquet(network_path)

    road_graph = RoadGraph(network_df)

    # ---------------------------------------------------------
    # 3. Prepare HMM emissions and candidate positions
    # ---------------------------------------------------------

    candidate_probs = []
    candidate_positions = []

    latitudes = []
    longitudes = []

    for group in candidate_groups:

        candidate_probs.append(
            group["score"].to_numpy(dtype=float)
        )

        positions = list(
            zip(
                group["candidate_id"].astype(int),
                group["frac_along"].astype(float),
            )
        )

        candidate_positions.append(positions)

        latitudes.append(
            float(group["latitude"].iloc[0])
        )

        longitudes.append(
            float(group["longitude"].iloc[0])
        )

    # GPS displacement between consecutive observations.
    gps_dists = [
        haversine_m(
            latitudes[i],
            longitudes[i],
            latitudes[i + 1],
            longitudes[i + 1],
        )
        for i in range(n_obs - 1)
    ]

    # ---------------------------------------------------------
    # 4. Actual road-network transition distances
    # ---------------------------------------------------------

    def candidate_distance_function(t_from, t_to):

        return road_graph.distance_matrix(
            candidate_positions[t_from],
            candidate_positions[t_to],
        )

    # ---------------------------------------------------------
    # 5. Run Viterbi
    # ---------------------------------------------------------

    result = viterbi(
        candidate_probs=candidate_probs,
        gps_dists=gps_dists,
        cand_dist_fn=candidate_distance_function,
        beta=beta,
    )

    # ---------------------------------------------------------
    # 6. Evaluate HMM predictions
    # ---------------------------------------------------------

    hmm_correct = 0
    hmm_evaluated = 0
    decoded_observations = 0

    selected_candidates = []

    for t, group in enumerate(candidate_groups):

        selected_index = result.sequence[t]

        if selected_index < 0:
            selected_candidates.append(None)
            continue

        selected = group.iloc[selected_index]

        selected_candidates.append(
            (
                int(selected["candidate_id"]),
                float(selected["frac_along"]),
            )
        )

        decoded_observations += 1

        if (group["label"] == 1).any():

            hmm_evaluated += 1

            hmm_correct += int(
                int(selected["label"]) == 1
            )

    hmm_accuracy = (
        hmm_correct / hmm_evaluated
        if hmm_evaluated
        else float("nan")
    )

    # ---------------------------------------------------------
    # 7. Directed road-network continuity
    # ---------------------------------------------------------

    connected_transitions = 0
    evaluated_transitions = 0

    for i in range(len(selected_candidates) - 1):

        current = selected_candidates[i]
        following = selected_candidates[i + 1]

        if current is None or following is None:
            continue

        evaluated_transitions += 1

        distance = road_graph.candidate_distance(
            current[0],
            current[1],
            following[0],
            following[1],
        )

        if np.isfinite(distance):
            connected_transitions += 1

    continuity_rate = (
        connected_transitions / evaluated_transitions
        if evaluated_transitions
        else float("nan")
    )

    return {
        "trip": trip_id,
        "n_obs": n_obs,
        "n_eval_obs": evaluation_observations,
        "clf_correct": classifier_correct,
        "hmm_correct": hmm_correct,
        "clf_proxy_accuracy": classifier_accuracy,
        "hmm_proxy_accuracy": hmm_accuracy,
        "decoded_observations": decoded_observations,
        "coverage": decoded_observations / n_obs if n_obs else 0.0,
        "n_transitions": evaluated_transitions,
        "connected_transitions": connected_transitions,
        "continuity_rate": continuity_rate,
        "n_imputed": len(result.imputed),
        "log_probability": result.logprob,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trip",
        type=str,
        default=None,
        help="Evaluate one trip, e.g. 00000052",
    )

    args = parser.parse_args()

    # Load existing features without rebuilding them.
    df = pd.read_parquet(FEATURES_PATH)

    # Load Member 2's exact feature-column definition.
    with open(FEATURE_COLUMNS_PATH) as f:
        X_cols = json.load(f)

    missing = set(X_cols) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing model features: {sorted(missing)}"
        )

    # Load Member 2's trained classifier.
    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The trained model does not support predict_proba."
        )

    # Use student_test only.
    test = df[
        df["partition"] == "student_test"
    ].copy()

    if args.trip:
        trip_id = args.trip.zfill(8)

        test = test[
            test[COL_TRIP].astype(str) == trip_id
        ]

        if test.empty:
            raise ValueError(
                f"Trip {trip_id} was not found in student_test."
            )

    rows = []

    for trip_id, group in test.groupby(COL_TRIP, sort=True):

        print(f"\nEvaluating trip {trip_id}...")

        row = compare_trip(
            group,
            model,
            X_cols,
        )

        rows.append(row)

        print(
            f"Classifier proxy accuracy: "
            f"{row['clf_proxy_accuracy']:.4f}"
        )

        print(
            f"HMM proxy accuracy: "
            f"{row['hmm_proxy_accuracy']:.4f}"
        )

        print(
            f"Directed continuity: "
            f"{row['continuity_rate']:.4f}"
        )

    output = pd.DataFrame(rows)

    # Add a weighted overall row.
    if len(output) > 1:

        total_eval = output["n_eval_obs"].sum()
        total_obs = output["n_obs"].sum()
        total_transitions = output["n_transitions"].sum()

        overall = {
            "trip": "overall",
            "n_obs": total_obs,
            "n_eval_obs": total_eval,
            "clf_correct": output["clf_correct"].sum(),
            "hmm_correct": output["hmm_correct"].sum(),
            "clf_proxy_accuracy": (
                output["clf_correct"].sum() / total_eval
                if total_eval else float("nan")
            ),
            "hmm_proxy_accuracy": (
                output["hmm_correct"].sum() / total_eval
                if total_eval else float("nan")
            ),
            "decoded_observations": output[
                "decoded_observations"
            ].sum(),
            "coverage": (
                output["decoded_observations"].sum() / total_obs
                if total_obs else 0.0
            ),
            "n_transitions": total_transitions,
            "connected_transitions": output[
                "connected_transitions"
            ].sum(),
            "continuity_rate": (
                output["connected_transitions"].sum()
                / total_transitions
                if total_transitions else float("nan")
            ),
            "n_imputed": output["n_imputed"].sum(),
            "log_probability": float("nan"),
        }

        output = pd.concat(
            [output, pd.DataFrame([overall])],
            ignore_index=True,
        )

    if args.trip:
        output_path = RESULTS / (
            f"hmm_vs_classifier_{args.trip.zfill(8)}.csv"
        )
    else:
        output_path = RESULTS / "hmm_vs_classifier.csv"

    output.to_csv(output_path, index=False)

    print("\n===== COMPARISON RESULTS =====")
    print(output.to_string(index=False))

    print(f"\nSaved results to: {output_path}")
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()

