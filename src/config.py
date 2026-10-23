"""Shared settings. Set ANN_QUICK=1 (done by `run_all.sh --quick`) for the small run."""
import os
from pathlib import Path

QUICK = os.environ.get("ANN_QUICK") == "1"

ROOT = Path(__file__).resolve().parents[2]          # folder that holds code/, results/, data/
DATA_DIR = ROOT / "data"                            # datasets + cached ground truth (not submitted)
RES_DIR = ROOT / ("results_quick" if QUICK else "results")
FIG_DIR = RES_DIR / "figures"
RESULTS_CSV = RES_DIR / "results.csv"

K = 10                                              # we always measure recall@10
GT_K = 100                                          # ground truth keeps 100 neighbours (for ties)

N_TRAIN = 100_000 if QUICK else None                # None = use every vector
N_HALF = 1_000 if QUICK else 5_000                  # queries per half (tune / final)
N_TIME = 200 if QUICK else 500                      # queries used for the one-at-a-time timing
REPEATS = 3                                         # timing repeats; we report the median
STOP_RECALL = 0.995                                 # stop a sweep once tuning recall reaches this

DATASETS = {
    "sift": "sift-128-euclidean.hdf5",
    "glove": "glove-100-angular.hdf5",
}
BASE_URL = "http://ann-benchmarks.com/"
SEED = 0

for d in (DATA_DIR, RES_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
