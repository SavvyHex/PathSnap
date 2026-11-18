"""Compare classifier-only and road-network HMM route reconstruction."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, DATA_CLEAN, RESULTS, TEST_DIR, HMM_BETA, COL_TRIP
from src.data.preprocess import parse_trip
from src.features.build_features import haversine_m
from src.hmm.road_graph import RoadGraph
from src.hmm.viterbi import viterbi


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


MODEL_CHOICES = ("tuned", "gnb", "logreg", "rf", "et", "dtree")

# The committed results/hmm_vs_classifier.csv was recorded with the tuned model
# at beta 30; only that exact combination keeps the legacy filename so the
# recorded baseline is never silently overwritten.
LEGACY_MODEL = "tuned"
LEGACY_BETA = 30.0


def _output_name(model_name: str, beta: float) -> str:
    if model_name == LEGACY_MODEL and beta == LEGACY_BETA:
        return "hmm_vs_classifier.csv"
    return f"hmm_vs_classifier_{model_name}_beta{beta:g}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=MODEL_CHOICES, default="tuned",
        help="emission classifier artifact: artifacts/model_<name>.joblib (default: tuned)",
    )
    parser.add_argument(
        "--beta", type=float, default=HMM_BETA,
        help=f"HMM transition scale in metres (default: config HMM_BETA={HMM_BETA:g})",
    )
    args = parser.parse_args()
    if not np.isfinite(args.beta) or args.beta <= 0:
        raise SystemExit("--beta must be finite and positive")

    df = pd.read_parquet(RESULTS / "features.parquet")
    X_cols = json.loads((ARTIFACTS / "feature_columns.json").read_text())
    model = joblib.load(ARTIFACTS / f"model_{args.model}.joblib")
    test = df[df["partition"] == "student_test"]
    rows = []
    for trip_id, candidates in test.groupby(COL_TRIP):
        trip_id = str(trip_id)
        folder = TEST_DIR / trip_id
        _, track, _, _, _ = parse_trip(folder)
        track["obs_id"] = np.arange(len(track))
        arcs_path = DATA_CLEAN / "network" / f"{trip_id}.arcs.parquet"
        graph = RoadGraph(pd.read_parquet(arcs_path))
        rows.append(compare_trip(candidates, track, model, X_cols, graph, beta=args.beta))
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
    out_path = RESULTS / _output_name(args.model, args.beta)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"[compare] model={args.model} beta={args.beta:g} -> {out_path}")


if __name__ == "__main__":
    main()

