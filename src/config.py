"""Central configuration.

Dataset layout (confirmed with the team, member 1):
  train/00000001/00000001.{arcs,nodes,route,track}          (repo root, trips 1-99 minus test ids)
  student_test/00000000/00000000.{arcs,nodes,route,track}   (repo root)

The dataset stays local only - train/, student_test/ and Archive.zip
are gitignored and never committed.

Per trip:
  .track  GPS observations (the trajectory)
  .nodes  road-network nodes
  .arcs   road-network arcs (directed segments between nodes)
  .route  reference matched route (ground truth, train/student-test only)

Column semantics below are best guesses for these file formats and are
verified/adjusted by src/data/preprocess.py's sniffing on first run.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
RESULTS = ROOT / "results"
MAPS_DIR = ROOT / "maps"
ARTIFACTS = ROOT / "artifacts"
REFERENCE = ROOT / "api" / "reference"   # server-side ground truth, never shipped to UI

for _d in (DATA_RAW, DATA_CLEAN, RESULTS, MAPS_DIR, ARTIFACTS, REFERENCE):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "student_test"

TEST_TRIPS = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091",
]
# trips 1-99 excluding the ten test ids
TRAIN_TRIPS = sorted(p.name for p in TRAIN_DIR.glob("[0-9]*") if p.is_dir() and p.name not in TEST_TRIPS)

# candidate generation
N_CANDIDATES = 5            # nearest arcs kept per GPS observation
CANDIDATE_MAX_DIST_M = 100  # drop candidates beyond this distance

RANDOM_SEED = 42
N_CV_FOLDS = 5

COL_TRIP = "trip_id"

# HMM defaults (tuned later on train + student-test only)
HMM_BETA = 30.0    # transition scale metres
HMM_EPS = 1e-6     # floor probability for disconnected transitions
