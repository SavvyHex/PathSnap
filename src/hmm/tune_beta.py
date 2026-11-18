"""Tune HMM beta using out-of-fold predictions on training trips only."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.config import (
    ARTIFACTS,
    RESULTS,
    COL_TRIP,
    RANDOM_SEED,
    N_CV_FOLDS,
    HMM_BETA,
)
from src.features.build_features import haversine_m
from src.hmm.road_graph import RoadGraph
from src.hmm.viterbi import viterbi


FEATURES_PATH = RESULTS / "features.parquet"
MODEL_PATH = ARTIFACTS / "model_tuned.joblib"
FEATURE_COLUMNS_PATH = ARTIFACTS / "feature_columns.json"
NETWORK_DIR = Path("data/clean/network")

DEFAULT_BETAS = [10, 20, 30, 50, 75, 100]


def evaluate_trip(df_trip, model, feature_columns, betas, fold):
    """Evaluate one held-out training trip for multiple beta values."""

    df_trip = df_trip.copy()

    df_trip["obs_id"] = df_trip["obs_id"].astype(int)
    df_trip = df_trip.sort_values(
        ["obs_id", "obs_rank"]
    ).reset_index(drop=True)

    positive_index = list(model.classes_).index(1)

    df_trip["score"] = model.predict_proba(
        df_trip[feature_columns]
    )[:, positive_index]

    groups = [
        (int(obs_id), group.reset_index(drop=True))
        for obs_id, group in df_trip.groupby("obs_id", sort=True)
    ]

    trip_id = str(df_trip[COL_TRIP].iloc[0]).zfill(8)

    candidate_groups = [group for _, group in groups]
    n_obs = len(candidate_groups)

    # Classifier-only baseline.
    clf_correct = 0
    n_eval_obs = 0

    for group in candidate_groups:
        if not (group["label"] == 1).any():
            continue

        n_eval_obs += 1
        best_index = int(group["score"].to_numpy().argmax())
        clf_correct += int(group.iloc[best_index]["label"] == 1)

    clf_accuracy = (
        clf_correct / n_eval_obs if n_eval_obs else float("nan")
    )

    # Build this trip's directed road graph.
    network_path = NETWORK_DIR / f"{trip_id}.arcs.parquet"

    if not network_path.exists():
        raise FileNotFoundError(
            f"Missing network file for trip {trip_id}: {network_path}"
        )

    road_graph = RoadGraph(pd.read_parquet(network_path))

    candidate_probs = []
    candidate_positions = []
    latitudes = []
    longitudes = []

    for group in candidate_groups:
        candidate_probs.append(
            group["score"].to_numpy(dtype=float)
        )

        candidate_positions.append(
            list(zip(
                group["candidate_id"].astype(int),
                group["frac_along"].astype(float),
            ))
        )

        latitudes.append(float(group["latitude"].iloc[0]))
        longitudes.append(float(group["longitude"].iloc[0]))

    gps_dists = [
        haversine_m(
            latitudes[i],
            longitudes[i],
            latitudes[i + 1],
            longitudes[i + 1],
        )
        for i in range(n_obs - 1)
    ]

    # Cache road-distance matrices so all beta values reuse them.
    distance_matrix_cache = {}

    def candidate_distance_function(t_from, t_to):
        key = (t_from, t_to)

        if key not in distance_matrix_cache:
            from_ids, from_fracs = zip(*candidate_positions[t_from])
            to_ids, to_fracs = zip(*candidate_positions[t_to])
            distance_matrix_cache[key] = road_graph.transition_distances(
                from_ids,
                from_fracs,
                to_ids,
                to_fracs,
            )

        return distance_matrix_cache[key]

    results = []

    for beta in betas:
        decoded = viterbi(
            candidate_probs=candidate_probs,
            gps_dists=gps_dists,
            cand_dist_fn=candidate_distance_function,
            beta=beta,
        )

        hmm_correct = 0
        hmm_evaluated = 0
        decoded_count = 0

        for t, group in enumerate(candidate_groups):
            selected_index = decoded.sequence[t]

            if selected_index < 0:
                continue

            decoded_count += 1

            if (group["label"] == 1).any():
                hmm_evaluated += 1
                hmm_correct += int(
                    group.iloc[selected_index]["label"] == 1
                )

        hmm_accuracy = (
            hmm_correct / hmm_evaluated
            if hmm_evaluated
            else float("nan")
        )

        # Measure whether consecutive selected candidates are connected
        # by a finite directed road-network path.
        connected = 0
        transitions = 0

        for t in range(n_obs - 1):
            first = decoded.sequence[t]
            second = decoded.sequence[t + 1]

            if first < 0 or second < 0:
                continue

            transitions += 1

            matrix = candidate_distance_function(t, t + 1)

            if np.isfinite(matrix[first, second]):
                connected += 1

        continuity = (
            connected / transitions if transitions else float("nan")
        )

        results.append({
            "fold": fold,
            "trip": trip_id,
            "beta": beta,
            "n_obs": n_obs,
            "n_eval_obs": n_eval_obs,
            "clf_correct": clf_correct,
            "hmm_correct": hmm_correct,
            "clf_proxy_accuracy": clf_accuracy,
            "hmm_proxy_accuracy": hmm_accuracy,
            "decoded_observations": decoded_count,
            "coverage": decoded_count / n_obs if n_obs else 0.0,
            "transitions": transitions,
            "connected_transitions": connected,
            "continuity_rate": continuity,
            "n_imputed": len(decoded.imputed),
        })

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-trips",
        type=int,
        default=None,
        help="Optional pilot limit: process only the first N validation trips.",
    )

    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=DEFAULT_BETAS,
        help="Beta values to evaluate in metres.",
    )

    args = parser.parse_args()

    print("===== LOADING TRAINING FEATURES =====")

    df = pd.read_parquet(FEATURES_PATH)

    train_all = df[df["partition"] == "train"].copy()
    train_all[COL_TRIP] = train_all[COL_TRIP].astype(str).str.zfill(8)
    train_all["obs_id"] = train_all["obs_id"].astype(int)

    # Match Member 2's training-row selection exactly.
    train_sample = train_all[
        train_all["obs_id"] % 10 == 0
    ].reset_index(drop=True)

    print("All training candidate rows:", len(train_all))
    print("Rows used for classifier fitting:", len(train_sample))
    print("Training trips:", train_all[COL_TRIP].nunique())

    feature_columns = json.loads(
        FEATURE_COLUMNS_PATH.read_text()
    )

    final_model = joblib.load(MODEL_PATH)

    print("\n===== MODEL =====")
    print("Model:", type(final_model).__name__)
    print("Parameters:", final_model.get_params())
    print("Features:", feature_columns)
    print("Beta values:", args.betas)

    # GroupKFold ensures that no trip appears in both the fitting
    # and validation groups for the same fold.
    cv = GroupKFold(n_splits=N_CV_FOLDS)

    all_results = []
    processed_trips = 0

    for fold, (train_indices, validation_indices) in enumerate(
        cv.split(
            train_sample,
            train_sample["label"],
            groups=train_sample[COL_TRIP],
        ),
        start=1,
    ):
        fold_train = train_sample.iloc[train_indices]
        validation_ids = sorted(
            train_sample.iloc[validation_indices][COL_TRIP].unique()
        )

        print(f"\n===== FOLD {fold}/{N_CV_FOLDS} =====")
        print("Classifier training trips:", fold_train[COL_TRIP].nunique())
        print("Validation trips:", len(validation_ids))

        # Clone preserves the tuned hyperparameters but discards
        # the fitted state. This fold model has not seen validation trips.
        fold_model = clone(final_model)

        fold_model.fit(
            fold_train[feature_columns],
            fold_train["label"],
        )

        for trip_id in validation_ids:
            if (
                args.max_trips is not None
                and processed_trips >= args.max_trips
            ):
                break

            trip_df = train_all[
                train_all[COL_TRIP] == trip_id
            ]

            print(
                f"Evaluating trip {trip_id} "
                f"({processed_trips + 1}"
                f"{'/' + str(args.max_trips) if args.max_trips else ''})...",
                flush=True,
            )

            trip_results = evaluate_trip(
                df_trip=trip_df,
                model=fold_model,
                feature_columns=feature_columns,
                betas=args.betas,
                fold=fold,
            )

            all_results.extend(trip_results)
            processed_trips += 1

        if (
            args.max_trips is not None
            and processed_trips >= args.max_trips
        ):
            break

    if not all_results:
        raise RuntimeError("No validation trips were evaluated.")

    result_df = pd.DataFrame(all_results)

    if args.max_trips is None:
        output_path = RESULTS / "hmm_beta_tuning.csv"
    else:
        output_path = RESULTS / "hmm_beta_tuning_pilot.csv"

    result_df.to_csv(output_path, index=False)

    # Aggregate counts across trips, rather than taking an unweighted
    # average of trip-level accuracy percentages.
    summary = (
        result_df.groupby("beta")
        .agg(
            trips=("trip", "nunique"),
            observations=("n_obs", "sum"),
            evaluation_observations=("n_eval_obs", "sum"),
            clf_correct=("clf_correct", "sum"),
            hmm_correct=("hmm_correct", "sum"),
            decoded=("decoded_observations", "sum"),
            transitions=("transitions", "sum"),
            connected=("connected_transitions", "sum"),
        )
        .reset_index()
    )

    summary["clf_proxy_accuracy"] = (
        summary["clf_correct"] / summary["evaluation_observations"]
    )

    summary["hmm_proxy_accuracy"] = (
        summary["hmm_correct"] / summary["evaluation_observations"]
    )

    summary["coverage"] = (
        summary["decoded"] / summary["observations"]
    )

    summary["continuity_rate"] = (
        summary["connected"] / summary["transitions"]
    )

    print("\n===== BETA TUNING SUMMARY =====")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
