"""Train the histogram gradient-boosting emission model on derived features.

Selection discipline: model/threshold choices use grouped CV over train trips
only (GroupKFold on trip_id); the student-test partition is evaluated once, for
characterisation, after the configuration is frozen.

Writes artifacts/model_hgb.joblib and artifacts/feature_columns_hgb.json, and
appends a row to results/model_comparison_hgb.csv.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from src.config import ARTIFACTS, COL_TRIP, N_CV_FOLDS, RANDOM_SEED, RESULTS
from src.features.derived import BASE_COLS, DERIVED_COLS, add_derived_features

X_COLS = BASE_COLS + DERIVED_COLS
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
              min_samples_leaf=40, random_state=RANDOM_SEED)


def _top1(df: pd.DataFrame, proba: np.ndarray) -> float:
    tmp = df[[COL_TRIP, "obs_id", "label"]].copy()
    tmp["p"] = proba
    winners = tmp.loc[tmp.groupby([COL_TRIP, "obs_id"])["p"].idxmax()]
    return float(winners["label"].mean())


def main() -> None:
    df = pd.read_parquet(RESULTS / "features.parquet")
    missing = [c for c in X_COLS if c not in df.columns]
    if missing:
        df = add_derived_features(df)  # in-memory; run src.features.derived to persist
    train = df[df.partition == "train"].reset_index(drop=True)
    test = df[df.partition == "student_test"].reset_index(drop=True)

    folds = min(N_CV_FOLDS, train[COL_TRIP].nunique())
    if folds < 2:
        raise SystemExit('need at least 2 train trips for grouped CV')
    gkf = GroupKFold(n_splits=folds)
    macro, macro_t, pos_t, ll, top1 = [], [], [], [], []
    for tr, va in gkf.split(train[X_COLS], train["label"], groups=train[COL_TRIP]):
        m = HistGradientBoostingClassifier(**PARAMS)
        m.fit(train.iloc[tr][X_COLS], train.iloc[tr]["label"])
        pv = m.predict_proba(train.iloc[va][X_COLS])[:, 1]
        yv = train.iloc[va]["label"].to_numpy()
        pt = m.predict_proba(train.iloc[tr][X_COLS])[:, 1]
        yt = train.iloc[tr]["label"].to_numpy()
        thr = max((f1_score(yt, pt >= c, average="macro"), c)
                  for c in np.linspace(0.05, 0.95, 38))[1]
        macro.append(f1_score(yv, pv >= 0.5, average="macro"))
        macro_t.append(f1_score(yv, pv >= thr, average="macro"))
        pos_t.append(f1_score(yv, pv >= thr, pos_label=1))
        ll.append(log_loss(yv, np.clip(pv, 1e-12, 1 - 1e-12)))
        top1.append(_top1(train.iloc[va], pv))
    cv = {"cv_macro_f1": float(np.mean(macro)),
          "cv_macro_f1_tuned_threshold": float(np.mean(macro_t)),
          "cv_positive_f1_tuned_threshold": float(np.mean(pos_t)),
          "cv_log_loss": float(np.mean(ll)), "cv_top1": float(np.mean(top1))}
    print("CV:", json.dumps(cv, indent=2))

    t0 = time.perf_counter()
    model = HistGradientBoostingClassifier(**PARAMS)
    model.fit(train[X_COLS], train["label"])
    train_s = time.perf_counter() - t0
    pt = model.predict_proba(test[X_COLS])[:, 1]
    yt = test["label"].to_numpy()
    test_metrics = {"test_macro_f1@0.5": float(f1_score(yt, pt >= 0.5, average="macro")),
                    "test_log_loss": float(log_loss(yt, np.clip(pt, 1e-12, 1 - 1e-12))),
                    "test_top1": _top1(test, pt)}
    print("student-test characterisation (single, post-selection):",
          json.dumps(test_metrics, indent=2))

    joblib.dump(model, ARTIFACTS / "model_hgb.joblib", compress=3)
    (ARTIFACTS / "feature_columns_hgb.json").write_text(json.dumps(X_COLS))
    row = pd.DataFrame([{"model": "hgb", "config": json.dumps(PARAMS),
                         "n_features": len(X_COLS), "train_s": round(train_s, 3),
                         **cv, **test_metrics}])
    out_path = RESULTS / "model_comparison_hgb.csv"
    row.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    print("wrote artifacts/model_hgb.joblib, artifacts/feature_columns_hgb.json,", out_path)


if __name__ == "__main__":
    main()
