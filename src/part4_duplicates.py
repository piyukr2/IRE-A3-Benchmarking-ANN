"""Part 4 (hard geometry): what do many near-duplicate vectors do to HNSW and IVF-Flat?
Only one thing changes: how many copies of each vector are in the index. 1 copy = control."""
import numpy as np

import config
import data
from bench import build_info, sweep
from data import Dataset
from gen_data import near_duplicates
from indexes import build_hnsw, build_ivfflat, set_ef_search, set_nprobe
from results import save_rows

N_BASE = 10_000 if config.QUICK else 50_000          # distinct real SIFT vectors
COPIES = [1, 2, 5, 10]                                # up to 500,000 vectors in the index
EF = [10, 20, 40, 80, 160, 320, 640, 1280]
NPROBE = [1, 2, 4, 8, 16, 32, 64, 128]
M, EF_CONSTRUCTION, NLIST = 16, (100 if config.QUICK else 200), (128 if config.QUICK else 256)

real = data.load("sift")                              # we reuse its (real) queries
base = real.train[:N_BASE]
queries = (real.tune_q, real.final_q)

rows = []
for copies in COPIES:
    train = near_duplicates(base, copies)
    ids, dists = data.exact_ground_truth(train, np.vstack(queries))
    n = len(queries[0])
    ds = Dataset(f"sift-dup{copies}", train, queries[0], queries[1],
                 (ids[:n], dists[:n]), (ids[n:], dists[n:]), None)
    print(f"=== {copies} copies per vector -> {len(train)} vectors")
    extra = {"copies": copies}

    index, secs = build_hnsw(train, M, EF_CONSTRUCTION)
    info = build_info(ds, "hnsw", f"M={M},efC={EF_CONSTRUCTION}", index, secs)
    rows += sweep(index, ds, set_ef_search, EF, {**info, **extra, "search_param_name": "efSearch"})

    index, secs = build_ivfflat(train, NLIST)
    info = build_info(ds, "ivfflat", f"nlist={NLIST}", index, secs)
    rows += sweep(index, ds, set_nprobe, NPROBE, {**info, **extra, "search_param_name": "nprobe"})

save_rows("part4", rows)
