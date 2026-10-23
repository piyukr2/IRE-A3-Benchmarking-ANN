# Benchmarking Approximate Nearest Neighbour Search

HNSW, IVF-Flat and IVF-PQ (FAISS) compared against exact search on SIFT-1M and GloVe-100.
Everything is Euclidean distance; GloVe vectors are normalised first.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The datasets (`sift-128-euclidean.hdf5`, `glove-100-angular.hdf5`) are downloaded automatically
from ann-benchmarks.com into `../data/` the first time you run. Machine specs and library
versions of each run are written to `results/environment.txt`.

## Running
| Command | What it does | Output folder |
|---|---|---|
| `bash run_all.sh` | full run: all 1M / 1.18M vectors, 5,000 + 5,000 queries | `../results/` |
| `bash run_all.sh --quick` | first 100,000 vectors, 1,000 + 1,000 queries, smaller settings | `../results_quick/` |

Each run: unit tests -> Part 1 -> Part 2 -> Part 3 -> Part 4 -> figures/tables.
Outputs: `results.csv` (one row per benchmark run), `figures/*.png`, `tables/*.md|csv`.

## Layout
| File | Purpose |
|---|---|
| `src/config.py` | all settings, quick-mode switch, paths |
| `src/data.py` | load datasets, split queries (first half tunes, last half reports), exact ground truth (cached) |
| `src/metrics.py` | our recall@10 (tie-aware), single-query and batch timing |
| `src/indexes.py` | builders for HNSW, IVF-Flat, IVF-PQ (+ optional re-ranking) |
| `src/bench.py` | sweep a search parameter -> recall and QPS |
| `src/part1_exact.py` ... `part4_duplicates.py` | the four experiments |
| `src/gen_data.py` | seeded generator for the Part 4 near-duplicate data |
| `src/make_report_assets.py` | results.csv -> figures and tables |
| `tests/test_metrics.py` | unit tests for recall@10 (`pytest tests`) |
| `hypothesis.md` | hypotheses for Parts 3 and 4, committed before running them |

## Method in short
- **Recall@10** is computed by our own function. Ties at the 10th neighbour are not mistakes:
  neighbours strictly closer than the 10th must be found, equally-far ones are interchangeable.
- **Tuning vs. reporting:** parameters (e.g. the efSearch/nprobe that first reaches recall 0.90 / 0.95)
  are chosen on queries 0-4,999; every reported number is measured on queries 5,000-9,999.
- **Speed:** one query per `search` call, FAISS limited to 1 thread, median of 3 passes over the first
  500 final queries (200 in quick mode). Recall itself is computed with batch search (same results, faster).
- Indexes are built with all threads (build time is therefore a multi-core number).
