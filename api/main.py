"""PathSnap FastAPI backend.

Runs on laptop 1 during evaluation. Loads the tuned model and HMM
configuration once at startup. Ground-truth reference segments live in
api/reference/ (server-side only) and are never sent to the model or
the UI - the /predict endpoint uses them only to compute accuracy.

Endpoints:
  GET  /trips                 list the reserved evaluation trip ids
  POST /predict/{trip_id}     run the full pipeline on one trip and
                              return the decoded route + accuracy
  GET  /health                liveness check
"""
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import ARTIFACTS, DATA_CLEAN, REFERENCE, HMM_BETA
from src.features.build_features import haversine_m, motion_features
from src.hmm.viterbi import viterbi

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODEL = None
X_COLS = None
EVAL_DIR = DATA_CLEAN / "evaluator"   # reserved trips, features precomputed


@app.on_event("startup")
def load_artifacts():
    global MODEL, X_COLS
    MODEL = joblib.load(ARTIFACTS / "model_tuned.joblib")
    X_COLS = json.loads((ARTIFACTS / "feature_columns.json").read_text())
    log.info("model loaded: %d features", len(X_COLS))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/trips")
def trips():
    if not EVAL_DIR.exists():
        return {"trips": []}
    return {"trips": sorted(p.stem for p in EVAL_DIR.glob("*.csv"))}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    t0 = time.perf_counter()
    log.info("predict requested trip=%s", trip_id)
    path = EVAL_DIR / f"{trip_id}.csv"
    if not path.exists():
        log.warning("unknown trip requested: %s", trip_id)
        raise HTTPException(404, f"unknown trip {trip_id}")

    df = pd.read_csv(path)
    proba = MODEL.predict_proba(df[X_COLS])[:, 1]
    df = df.assign(score=proba)

    obs_ids = sorted(df["obs_id"].unique())
    cand_probs, lats, lons = [], [], []
    for oid in obs_ids:
        g = df[df["obs_id"] == oid]
        cand_probs.append(g["score"].values)
        lats.append(float(g["latitude"].iloc[0]))
        lons.append(float(g["longitude"].iloc[0]))
    gps_dists = [haversine_m(lats[t], lons[t], lats[t + 1], lons[t + 1])
                 for t in range(len(obs_ids) - 1)]

    def cand_dist_fn(a, b):
        gap = sum(gps_dists[a:b])
        return np.full((len(cand_probs[a]), len(cand_probs[b])), gap)

    res = viterbi(cand_probs, gps_dists, cand_dist_fn, beta=HMM_BETA)

    route, correct, total = [], 0, 0
    for t, oid in enumerate(obs_ids):
        g = df[df["obs_id"] == oid].iloc[res.sequence[t]] if res.sequence[t] >= 0 else None
        if g is not None:
            route.append({"obs_id": int(oid), "segment": str(g["candidate_id"]),
                          "lat": lats[t], "lon": lons[t]})

    accuracy = None
    ref_path = REFERENCE / f"{trip_id}.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        ref_map = dict(zip(ref["obs_id"], ref["matched_segment"]))
        for r in route:
            if r["obs_id"] in ref_map:
                total += 1
                correct += int(r["segment"] == str(ref_map[r["obs_id"]]))
        accuracy = correct / total if total else None

    elapsed = time.perf_counter() - t0
    log.info("predict done trip=%s obs=%d imputed=%d accuracy=%s elapsed=%.3fs",
             trip_id, len(obs_ids), len(res.imputed),
             f"{accuracy:.4f}" if accuracy is not None else "n/a", elapsed)
    return {"trip_id": trip_id, "n_observations": len(obs_ids),
            "n_imputed": len(res.imputed),
            "top1_accuracy": accuracy, "scored_observations": total,
            "elapsed_s": round(elapsed, 3), "route": route}
