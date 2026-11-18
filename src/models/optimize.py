"""Bayesian hyperparameter optimisation of the selected model.

Optuna TPE search with group-aware cross-validation (GroupKFold on
trip_id) over the training trips only; the best configuration is then
refit on all training trips and evaluated once on the student-testing
trips, alongside the untuned baseline, for the tuned-vs-untuned table.

Trials are logged to results/optuna_trials.csv for the report.
"""
import json
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from src.config import ARTIFACTS, RESULTS, RANDOM_SEED, N_CV_FOLDS, COL_TRIP
from src.models.train_zoo import MODELS, evaluate

N_TRIALS = 12


def objective(trial, train, X_cols, selected):
    if selected == "gnb":
        params = {"var_smoothing": trial.suggest_float("var_smoothing",1e-12,1e-3,log=True)}
    elif selected == "logreg":
        params = {"clf__C": trial.suggest_float("clf__C",.01,100,log=True)}
    else:
        params = {"max_depth":trial.suggest_int("max_depth",4,18),
                  "min_samples_leaf":trial.suggest_int("min_samples_leaf",2,20)}
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    scores = []
    for tr_idx, va_idx in gkf.split(train, train["label"], groups=train[COL_TRIP]):
        m = clone(MODELS[selected][1]).set_params(**params)
        m.fit(train.iloc[tr_idx][X_cols], train.iloc[tr_idx]["label"])
        proba = m.predict_proba(train.iloc[va_idx][X_cols])[:, 1]
        scores.append(f1_score(train.iloc[va_idx]["label"],
                               (proba >= 0.5).astype(int), average="macro"))
    return float(np.mean(scores))


def main() -> None:
    df = pd.read_parquet(RESULTS / "features.parquet")
    X_cols = json.loads((ARTIFACTS / "feature_columns.json").read_text())
    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    winner = comparison.loc[comparison.cv_macro_f1_mean.idxmax()]
    selected = winner["model"]
    train = df[(df.partition == "train") & (df.obs_id % 10 == 0)].reset_index(drop=True)
    test = df[df["partition"] == "student_test"].reset_index(drop=True)

    t0 = time.perf_counter()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=4))
    study.optimize(lambda t: objective(t, train, X_cols, selected), n_trials=N_TRIALS)
    opt_minutes = (time.perf_counter() - t0) / 60

    trials = study.trials_dataframe()
    trials.to_csv(RESULTS / "optuna_trials.csv", index=False)

    best = study.best_params
    tuned = clone(MODELS[selected][1]).set_params(**best)
    tuned.fit(train[X_cols], train["label"])
    joblib.dump(tuned, ARTIFACTS / "model_tuned.joblib", compress=3)
    (ARTIFACTS / "tuned_params.json").write_text(json.dumps({"model": selected, "parameters": best, "objective": "grouped_cv_macro_f1",
                         "n_trials": len(study.trials), "seed": RANDOM_SEED}, indent=2))

    untuned = joblib.load(ARTIFACTS / f"model_{selected}.joblib")
    comp = pd.DataFrame([
        {"variant": "untuned", **evaluate(test, untuned, X_cols)},
        {"variant": "tuned", **evaluate(test, tuned, X_cols)},
    ])
    comp["model"] = selected
    comp["cv_macro_f1"] = [winner.cv_macro_f1_mean, study.best_value]
    comp["optimisation_minutes"] = round(opt_minutes, 2)
    comp.to_csv(RESULTS / "tuned_vs_untuned.csv", index=False)
    print(comp.to_string(index=False))
    print("best params:", best)


if __name__ == "__main__":
    main()
