#!/usr/bin/env bash
# Regenerates every number, figure and table in the report.
#   bash run_all.sh           full run (SIFT 1M, GloVe 1.18M)      -> ../results/
#   bash run_all.sh --quick   first 100,000 vectors, fewer queries -> ../results_quick/
set -euo pipefail
cd "$(dirname "$0")"

export ANN_QUICK=0
[ "${1:-}" = "--quick" ] && export ANN_QUICK=1
RES=../results; [ "$ANN_QUICK" = "1" ] && RES=../results_quick
rm -rf "$RES"                                   # start clean so results.csv holds only this run

python -m pytest -q tests                        # unit tests for recall@10
python src/env_info.py                           # machine specs + library versions
python src/part1_exact.py                        # Part 1: exact baseline
python src/part2_benchmark.py                    # Part 2: HNSW / IVF-Flat / IVF-PQ
python src/part3_compression.py                  # Part 3: PQ compression experiment
python src/part4_duplicates.py                   # Part 4: near-duplicate experiment
python src/make_report_assets.py                 # figures + tables from results.csv
echo "Done. Results in $RES"
