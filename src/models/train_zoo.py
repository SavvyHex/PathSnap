"""Model zoo: train >=5 algorithm families and compare them honestly.

Design rules baked in (per the hackathon brief):
- cross-validation is group-aware: GroupKFold on trip_id, so
  observations of one trip never straddle train/validation folds;
- the student-testing partition is used for development comparisons
  only, never for fitting;
- every run logs configuration, timings and metrics to CSV.

Each observation is scored per candidate (binary: candidate is the
reference segment or not); Top-1 prediction = argmax over candidate
scores of one observation. Macro F1 and log loss are computed on the
binary candidate labels; route continuity is handled by the HMM stage.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import ARTIFACTS, RESULTS, RANDOM_SEED, N_CV_FOLDS, COL_TRIP, TEST_DIR

FEATURES_PATH = RESULTS / "features.parquet"   # produced by src.features
EXCLUDE = {"label", COL_TRIP, "obs_id"}

MODELS = {
    # name -> (family, estimator, config string for the report)
    "logreg": ("linear/probabilistic",
               Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))]),
               "L2 logistic regression, C=1.0, lbfgs"),
    "gnb": ("linear/probabilistic",
            GaussianNB(),
            "Gaussian Naive Bayes, default priors"),
    "knn": ("instance-based",
            Pipeline([("sc", StandardScaler()),
                      ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance"))]),
            "distance-weighted KNN, k=15"),
    "dtree": ("tree-based",
              DecisionTreeClassifier(max_depth=12, random_state=RANDOM_SEED),
              "CART decision tree, max_depth=12"),
    "rf": ("ensemble",
           RandomForestClassifier(n_estimators=35, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=RANDOM_SEED),
           "random forest, 35 trees, depth 12, minimum leaf 4"),
    "et": ("ensemble",
           ExtraTreesClassifier(n_estimators=35, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=RANDOM_SEED),
           "extra trees, 35 trees, depth 12, minimum leaf 4"),
    "mlp": ("neural network",
            Pipeline([("sc", StandardScaler()),
                      ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                            random_state=RANDOM_SEED))]),
            "MLP 64-32, ReLU, Adam"),
}


def top1_accuracy(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Per-observation: highest-scoring candidate vs reference label."""
    tmp = df[[COL_TRIP, "obs_id", "label"]].copy()
    tmp["score"] = scores
    best = tmp.loc[tmp.groupby([COL_TRIP,"obs_id"])["score"].idxmax()]
    return float((best["label"] == 1).mean())


def candidate_miss_rate(df: pd.DataFrame) -> float:
    per_obs = df.groupby([COL_TRIP,"obs_id"])["label"].max()
    return float((per_obs == 0).mean())


def continuity(df, scores):
    ranked = df.assign(score=scores)
    winners = ranked.loc[ranked.groupby([COL_TRIP,"obs_id"]).score.idxmax()]
    connected = total = 0
    for tid, group in winners.groupby(COL_TRIP):
        arcs = np.loadtxt(TEST_DIR / tid / f"{tid}.arcs",dtype=int)
        group = group.sort_values("obs_id")
        ids = group.candidate_id.to_numpy()
        adjacent = np.diff(group.obs_id.to_numpy()) == 1
        matches = (ids[:-1] == ids[1:]) | (arcs[ids[:-1],1] == arcs[ids[1:],0])
        connected += (matches & adjacent).sum()
        total += adjacent.sum()
    return connected/max(total,1)


def evaluate(df, model, X_cols):
    t0 = time.perf_counter()
    proba = model.predict_proba(df[X_cols])[:, 1]
    infer_s = time.perf_counter() - t0
    pred = (proba >= 0.5).astype(int)
    y = df["label"].values
    inventory = pd.read_csv(RESULTS / "data_inspection.csv", dtype={"trip": str})
    total_obs = int(inventory.loc[inventory.trip.isin(df[COL_TRIP].unique()), "track_rows"].sum())
    present = df.groupby([COL_TRIP,"obs_id"]).ngroups
    if total_obs < present:
        raise ValueError("Observation inventory does not match the evaluation partition")
    return {
        "n_observations": total_obs, "empty_candidate_sets": total_obs - present,
        "top1_accuracy": top1_accuracy(df, proba) * present / total_obs,
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "log_loss": log_loss(y, np.c_[1 - proba, proba], labels=[0, 1]),
        "candidate_miss_rate": (total_obs-present + candidate_miss_rate(df)*present)/total_obs,
        "route_continuity": continuity(df,proba),
        "inference_s": round(infer_s, 4),
    }


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    X_cols = ["obs_rank","dist_to_segment_m","frac_along","speed_mps","accel_mps2","turning_angle_deg"]
    # Sample entire observations at a fixed interval for a reproducible CPU budget.
    train = df[(df.partition == "train") & (df.obs_id % 10 == 0)].reset_index(drop=True)
    test = df[df.partition == "student_test"].reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    rows = []
    for name in ("logreg", "gnb", "dtree", "rf", "et"):
        family, model, cfg = MODELS[name]
        cv_f1 = []
        for tr_idx, va_idx in gkf.split(train, train["label"], groups=train[COL_TRIP]):
            m = clone(model)
            m.fit(train.iloc[tr_idx][X_cols], train.iloc[tr_idx]["label"])
            proba = m.predict_proba(train.iloc[va_idx][X_cols])[:, 1]
            cv_f1.append(f1_score(train.iloc[va_idx]["label"],
                                  (proba >= 0.5).astype(int), average="macro"))
        t0 = time.perf_counter()
        model.fit(train[X_cols], train["label"])
        train_s = time.perf_counter() - t0
        metrics = evaluate(test, model, X_cols)
        rows.append({"model": name, "family": family, "config": cfg,
                     "cv_macro_f1_mean": np.mean(cv_f1), "cv_macro_f1_std": np.std(cv_f1),
                     "train_s": round(train_s, 3), "train_macro_f1": f1_score(train.label,model.predict(train[X_cols]),average="macro"),
                     "cv_folds": json.dumps(cv_f1), **metrics})
        joblib.dump(model, ARTIFACTS / f"model_{name}.joblib", compress=3)
        print(name, rows[-1])

    out = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    out.to_csv(RESULTS / "model_comparison.csv", index=False)
    (ARTIFACTS / "feature_columns.json").write_text(json.dumps(X_cols))
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
