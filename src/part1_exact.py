"""Part 1: exact baseline, agreement with the stored neighbours, and single vs batch timing."""
import numpy as np

import config
import data
from bench import build_info
from indexes import build_flat
from metrics import median_batch_qps, median_qps, recall_at_k, single_thread
from results import save_rows

rows = []
for name in config.DATASETS:
    ds = data.load(name)
    index, build_s = build_flat(ds.train)
    info = build_info(ds, "exact", "flat", index, build_s)

    # 1. do our exact results agree with the neighbours stored in the data file?
    if ds.stored_final is not None:
        ours = index.search(ds.final_q, config.K)[1]
        same = np.mean([set(a) == set(b[: config.K]) for a, b in zip(ours, ds.stored_final)])
        tie_aware = recall_at_k(ds.stored_final, *ds.gt_final)
        print(f"{name}: queries with identical top-10 sets = {same:.4f}; "
              f"stored neighbours scored by our recall@10 against our exact search = {tie_aware:.4f}")
        rows.append({**info, "index": "check-vs-stored", "recall": tie_aware, "identical_sets": same})

    # 2. timing: one query at a time vs. all queries in one batch
    queries = ds.final_q[: config.N_TIME]
    search_batch = lambda qs: index.search(qs, config.K)
    with single_thread():
        qps_single = median_qps(search_batch, queries, config.REPEATS)
        qps_batch_1t = median_batch_qps(search_batch, queries, config.REPEATS)   # batch, still 1 thread
    qps_batch = median_batch_qps(search_batch, queries, config.REPEATS)          # batch, all threads
    print(f"{name}: exact qps  single-query 1 thread={qps_single:,.1f}  "
          f"batch 1 thread={qps_batch_1t:,.1f}  batch all threads={qps_batch:,.1f}")
    for label, qps in [("exact-single", qps_single), ("exact-batch-1thread", qps_batch_1t),
                       ("exact-batch", qps_batch)]:
        rows.append({**info, "index": label, "recall": 1.0, "qps": qps})

save_rows("part1", rows)
