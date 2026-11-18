# PathSnap

GPS map-matching and route reconstruction, built for the PES University Machine Learning Hackathon.

## What it does

Given noisy GPS trip observations, PathSnap scores candidate road segments with a trained classifier and selects a candidate sequence with directed road-network Viterbi transitions. FastAPI serves the predictions, available road-arc geometry, and aggregate results from private scoring references.

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

## Local API and UI

Run both commands from the repository root in separate terminals after installing
the requirements. PowerShell examples follow. On Bash, use
`export PATHSNAP_MODE=development` or `export PATHSNAP_MODE=evaluation`.

```powershell
# API in development mode
$env:PATHSNAP_MODE = "development"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```powershell
# UI in the second terminal
python -m http.server 8080 --bind 127.0.0.1 --directory ui
```

Open `http://localhost:8080`, select `development-demo`, and choose **Preview
demo route**. Development mode uses a synthetic GPS trace and requires no
trained model or trip files. Model-derived metrics are unavailable in this mode.
The Leaflet library and OpenStreetMap basemap use external network resources.

### Evaluation on two laptops

On laptop 1, run these commands from PathSnap in PowerShell. Substitute the
dataset directory if the supplied student-test files are stored elsewhere.

```powershell
$env:PATHSNAP_MODE = 'evaluation'
$env:PATHSNAP_EVALUATOR_DIR = 'data/clean/evaluator_student_test'
$env:PATHSNAP_REFERENCE_DIR = 'api/reference/student_test'
$env:PATHSNAP_NETWORK_SOURCE_DIR = '..\Student folder\Student folder\student_test'
$env:PATHSNAP_ROUTE_REFERENCE_DIR = $env:PATHSNAP_NETWORK_SOURCE_DIR
$env:PATHSNAP_CORS_ORIGINS = 'http://127.0.0.1:8080,http://localhost:8080'
.\.venv\Scripts\python.exe -m api.evaluator_adapter student_test --source-dir $env:PATHSNAP_NETWORK_SOURCE_DIR
ipconfig
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On laptop 2, run the following from the directory that contains the updated
`ui` folder.

```powershell
python -m http.server 8080 --bind 127.0.0.1 --directory ui
```

Open `http://127.0.0.1:8080/`, enter `http://<laptop-1-LAN-IP>:8000`
in **API address**, and select **Connect**. Trip `00000052` is a short demo trip.
The address is stored in the page URL as the `api` query parameter. Both
laptops need network reachability to port 8000 on laptop 1. The API's CORS
allowlist must include the exact origin shown in the browser on laptop 2.
The default covers `http://localhost:8080` and `http://127.0.0.1:8080`.
If the UI is opened through its LAN address, add that origin explicitly.

| API environment variable | Default | Purpose |
| --- | --- | --- |
| `PATHSNAP_MODE` | `evaluation` | Select `evaluation` or `development`. |
| `PATHSNAP_EVALUATOR_DIR` | `data/clean/evaluator` | Server-side reserved trip CSVs. |
| `PATHSNAP_ARTIFACTS_DIR` | `artifacts` | Trusted classifier and feature manifest. |
| `PATHSNAP_REFERENCE_DIR` | `api/reference` | Private scoring CSVs. Keep outside the static UI directory. |
| `PATHSNAP_NETWORK_SOURCE_DIR` | Unset | Original trip folders containing `.nodes` and `.arcs`. When set, use this network source instead of the prepared Parquet directory. |
| `PATHSNAP_ROUTE_REFERENCE_DIR` | Unset | Private trip folders containing route-level `.route` files. Use the reference root for the same dataset as the candidate CSVs and network. |
| `PATHSNAP_CORS_ORIGINS` | `http://localhost:8080,http://127.0.0.1:8080` | Comma-separated browser origins. |
| `PATHSNAP_HMM_BETA` | `30.0` | Positive transition scale, in metres. |

`GET /health` reports process liveness and whether a model loaded.
`GET /ready` returns 503 in evaluation mode until a model has loaded and
at least one prepared CSV has valid model inputs. Trip discovery validates the
CSV and caches the result until the file changes. Restart the API after
replacing the model or feature manifest. `GET /trips` lists trip IDs and mode,
and `POST /predict/{trip_id}` returns the route and aggregate metrics.
Use the `X-Request-ID` header to correlate responses with API logs and the
`Server-Timing` header to read server request duration in milliseconds.

### Integration contracts

- **Preprocessing.** Export one UTF-8 CSV per reserved trip. File names use
  `<trip_id>.csv`, with IDs of 1–64 ASCII letters, digits, underscores or
  hyphens. Include `obs_id` (integer in chronological order), `candidate_id`
  (text), `latitude` and `longitude` (GPS coordinates in degrees), and every
  feature in the manifest. Use one row per candidate and consistent coordinates
  for all rows of an observation. Use one row with a blank candidate ID for
  observations without candidates. Candidate features can be blank on that row.
  The offline adapter uses Member 1's candidate and motion functions. It reads
  only the trip's `.track`, `.nodes`, and `.arcs` files. Blank candidate
  rows preserve the original observation IDs through gaps.
- **ML.** Provide `model_tuned.joblib` from a trusted source and
  `feature_columns.json` containing the ordered, unique numeric feature names
  used to fit that model. The estimator must implement `predict_proba(frame)`
  with one two-column probability row per candidate and `classes_ == [0, 1]`.
  Match the training dependency versions when loading the artifact. The tuned
  model requires the exact six-feature order defined in `api/inference.py`.
  The local training scripts write the model and manifest in separate steps,
  so publish a matching pair.
- **HMM.** The API groups observations by ascending ID and keeps candidate
  row order within each group. `viterbi(candidate_probs, gps_dists,
  cand_dist_fn, beta=...)` receives one probability array per observation and
  successive GPS distances in metres. A road-distance provider must return an
  `(n_from, n_to)` matrix in candidate row order for any pair of observations
  bridged by the decoder, using metres and infinity for disconnected pairs.
  The decoder returns candidate indices and imputed observation positions.
  The API uses Member 3's directed `RoadGraph.transition_distances` and
  reports `transition_model: "directed_road_network"`. If topology or
  projection fractions are unavailable, it reports `"classifier_only"`
  and explains why. `disconnected_transitions` counts selected transitions
  with no finite directed path. `connected_geometry` reconstructs directed
  shortest paths between the selected candidate projections. It contains one
  LineString per continuous section and leaves gaps where transitions are
  unreachable. `route_completeness` is `connected` when those sections cover
  the trip continuously, and `partial` when there are gaps or uncovered
  endpoints. `reconstructed_connecting_arcs` counts traversals of intermediate
  arcs, including repeat traversals. `reconstructed_distance_m` sums the
  verified sections in metres, while `route_distance_m` is populated for
  a connected trip. Distances use the decoder's graph edge lengths and
  projection fractions. The network supplies only arc endpoints, so each
  arc is drawn as a straight segment between its known endpoints or trimmed
  positions. The UI offers separate GPS, candidate-arc, and reconstructed
  path layers. The API also returns a Leaflet-compatible
  `geometry.geojson` FeatureCollection with one directed LineString per
  resolved observation, in observation order. Each line uses the from-node
  and to-node of the original zero-based arc ID, in longitude/latitude order.
  Set `PATHSNAP_NETWORK_SOURCE_DIR` for original network files, or use
  `data/clean/network/<trip_id>.arcs.parquet` with optional
  `PATHSNAP_NETWORK_DIR` override. `geometry.status` and its coverage counts
  report how many decoded observations have resolvable geometry.
- **Scoring.** Keep `<trip_id>.csv` with unique `obs_id,matched_segment` rows
  in the private reference directory. Segment strings must match evaluator IDs,
  including leading zeroes. Only the server-side scoring step reads this
  file, after decoding. Accuracy is the fraction of returned route points with
  a matching segment among points with a reference, including imputed points
  with references. Missing or invalid references leave accuracy unavailable.
  `top1_accuracy` is a unitless fraction from 0 to 1. The response contains
  aggregate accuracy and count, without per-point labels.
- **Route-level overlap.** The student-test `.route` files contain directed arc
  IDs and have no observation alignment. They leave `top1_accuracy` null.
  `route_overlap.score` is Jaccard similarity, the number of distinct directed
  arc IDs shared by the prediction and reference divided by the number of
  distinct IDs in their union. It is a unitless fraction from 0 to 1. Repeated
  arcs count once, imputed selections are excluded, and opposite directions
  are distinct. The score measures arc membership. It does not assess traversal
  order, road length, or connectivity. A missing or invalid reference or an
  unresolved arc leaves the score null. Only the scoring step opens `.route`
  files, after decoding. The API returns the aggregate score and availability.

### Prepare trips on Windows

Run these commands from the repository root. For reserved evaluator inputs,
place each trip under `data/raw/evaluator/<trip_id>/` with files named
`<trip_id>.track`, `<trip_id>.nodes`, and `<trip_id>.arcs`. Use
`--source-dir` if the reserved dataset is stored elsewhere.

```powershell
.\.venv\Scripts\python.exe -m api.evaluator_adapter evaluator
```

This writes reserved inputs to `data/clean/evaluator/`. Prepare the supplied
student-test dataset separately for local testing. Omit `--source-dir` if the
dataset is at `student_test/` in the repository root.

```powershell
.\.venv\Scripts\python.exe -m api.evaluator_adapter student_test --source-dir '..\Student folder\Student folder\student_test'
$env:PATHSNAP_EVALUATOR_DIR = 'data/clean/evaluator_student_test'
$env:PATHSNAP_REFERENCE_DIR = 'api/reference/student_test'
$env:PATHSNAP_NETWORK_SOURCE_DIR = '..\Student folder\Student folder\student_test'
$env:PATHSNAP_ROUTE_REFERENCE_DIR = $env:PATHSNAP_NETWORK_SOURCE_DIR
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
```

Preparation skips incomplete, empty, and candidate-free trips and reports
them. Identical CSVs are left in place. If an existing CSV differs, move it
aside before regenerating. To return to the default reserved evaluator
directories, clear all four environment overrides in your PowerShell session.

### Tests

```powershell
python -m pip install httpx
python -m unittest discover -s tests -v
node --test tests/test_ui.cjs
```
