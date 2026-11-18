# PathSnap

GPS map-matching and route reconstruction, built for the PES University Machine Learning Hackathon.

## What it does

Given noisy GPS trip observations, PathSnap scores candidate road segments with a trained classifier, then reconstructs a coherent route with an HMM/Viterbi decoder. A FastAPI backend serves predictions and scores them against server-side reference data; a small web UI displays the predicted route and evaluation results.

## Repository layout

```
train/, student_test/  trip folders at the repo root (LOCAL ONLY, gitignored)
data/clean/          processed observations and candidates
src/config.py        column names, paths, seeds - single place to adjust to the dataset schema
src/data/            data inspection and preprocessing
src/features/        feature engineering for (observation, candidate) pairs
src/models/          model zoo training + comparison, Bayesian hyperparameter optimisation
src/hmm/             HMM/Viterbi route reconstruction and comparison against classifier-only
api/                 FastAPI backend (model loading, prediction, server-side scoring)
ui/                  evaluator web interface
maps/                Folium map generation scripts and exported maps
results/             metrics CSVs, experiment logs, plots
artifacts/           saved model, preprocessing objects, HMM configuration
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution

```bash
# 1. Preprocess raw trips and inspect missing values
python -m src.data.preprocess

# 2. Build features for every (observation, candidate) pair
python -m src.features.build_features

# 3. Train and compare the model zoo (group-aware CV by trip)
python -m src.models.train_zoo

# 4. Bayesian hyperparameter optimisation of the selected model
python -m src.models.optimize

# 5. HMM/Viterbi reconstruction and classifier-only comparison
python -m src.hmm.compare

# 6. Generate Folium maps
python -m maps.make_maps
```

## Deployment

```bash
# API on laptop 1
uvicorn api.main:app --host 0.0.0.0 --port 8000

# UI on laptop 2 (any static server)
cd ui && python -m http.server 8080
```

Set `PATHSNAP_API` in `ui/app.js` to the API host. Ground-truth labels stay server-side in `api/reference/` and are never sent to the model or the UI.
