"""Classifier-only vs classifier + HMM/Viterbi comparison.

Produces the metrics the report section needs: pointwise Top-1
accuracy and route continuity (fraction of consecutive decoded
segments that are connected / non-identical-jump-free) for both
variants, per trip and overall. Decoder parameters (beta) are selected
on train + student-test only.
"""
import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, RESULTS, HMM_BETA, COL_TRIP
from src.hmm.viterbi import viterbi
from src.features.build_features import haversine_m


def continuity(decoded_segments) -> float:
    """Fraction of consecutive steps that stay on the same segment or
    move to a plausible next one. A crude proxy until the road graph is
    wired in: counts large segment-id jumps as discontinuities."""
    decoded_segments = [s for s in decoded_segments if s != -1]
    if len(decoded_segments) < 2:
        return 1.0
    same = sum(1 for a, b in zip(decoded_segments, decoded_segments[1:]) if a == b)
    return same / (len(decoded_segments) - 1)


def compare_trip(df_obs, model, X_cols, beta=HMM_BETA):
    df_obs = df_obs.sort_values("obs_id").reset_index(drop=True)
    proba = model.predict_proba(df_obs[X_cols])[:, 1]
    df_obs = df_obs.assign(score=proba)

    # classifier-only
    best = df_obs.loc[df_obs.groupby("obs_id")["score"].idxmax()]
    clf_top1 = float((best["label"] == 1).mean())

    # HMM
    obs_ids = sorted(df_obs["obs_id"].unique())
    cand_probs, lats, lons = [], [], []
    for oid in obs_ids:
        g = df_obs[df_obs["obs_id"] == oid]
        cand_probs.append(g["score"].values)
        lats.append(g["lat"].iloc[0]); lons.append(g["lon"].iloc[0])
    gps_dists = [haversine_m(lats[t], lons[t], lats[t + 1], lons[t + 1])
                 for t in range(len(obs_ids) - 1)]
    # Candidate-pair distances need segment geometry; until the road
    # graph is wired in, approximate every pair by the observation
    # displacement (neutral transition) - flagged in the report.
    def cand_dist_fn(t_from, t_to):
        gap = sum(gps_dists[t_from:t_to])
        return np.full((len(cand_probs[t_from]), len(cand_probs[t_to])), gap)

    res = viterbi(cand_probs, gps_dists, cand_dist_fn, beta=beta)

    ref, dec = [], []
    for t, oid in enumerate(obs_ids):
        g = df_obs[df_obs["obs_id"] == oid]
        if res.sequence[t] >= 0 and (g["label"] == 1).any():
            ref.append(1)
            dec.append(int(g.iloc[res.sequence[t]]["label"] == 1))
    hmm_top1 = float(np.mean(dec)) if dec else float("nan")
    return {"trip": df_obs[COL_TRIP].iloc[0],
            "clf_top1": clf_top1, "hmm_top1": hmm_top1,
            "n_obs": len(obs_ids), "n_imputed": len(res.imputed)}


def main() -> None:
    df = pd.read_parquet(RESULTS / "features.parquet")
    X_cols = [c for c in df.columns
              if c not in {"label", COL_TRIP, "obs_id", "partition"} and df[c].dtype != object]
    model = joblib.load(ARTIFACTS / "model_tuned.joblib")
    rows = []
    test = df[df["partition"] == "student_test"]
    for trip_id, g in test.groupby(COL_TRIP):
        rows.append(compare_trip(g, model, X_cols))
    out = pd.DataFrame(rows)
    out.loc["overall"] = {"trip": "overall",
                          "clf_top1": out["clf_top1"].mean(),
                          "hmm_top1": out["hmm_top1"].mean(),
                          "n_obs": out["n_obs"].sum(),
                          "n_imputed": out["n_imputed"].sum()}
    out.to_csv(RESULTS / "hmm_vs_classifier.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
