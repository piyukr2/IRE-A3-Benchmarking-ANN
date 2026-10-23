"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ on both datasets."""
import config
import data
from bench import build_info, sweep
from indexes import (build_hnsw, build_ivfflat, build_ivfpq, set_ef_search, set_nprobe)
from results import save_rows

EF = [10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640, 960, 1280]
NPROBE = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]

if config.QUICK:   # 100K vectors: fewer, smaller settings
    HNSW_M, EF_CONSTRUCTION, NLISTS, PQ_NLIST = [16], 100, [256], 256
    PQ_M = {"sift": [16], "glove": [20]}
else:
    HNSW_M, EF_CONSTRUCTION, NLISTS, PQ_NLIST = [16, 32], 200, [1024, 4096], 1024
    PQ_M = {"sift": [16, 32], "glove": [20, 50]}      # m must divide the dimension (128 / 100)

rows = []
for name in config.DATASETS:
    ds = data.load(name)
    print(f"=== {name}: {len(ds.train)} vectors, dim {ds.train.shape[1]}")

    for M in HNSW_M:
        index, secs = build_hnsw(ds.train, M, EF_CONSTRUCTION)
        info = build_info(ds, "hnsw", f"M={M},efC={EF_CONSTRUCTION}", index, secs)
        rows += sweep(index, ds, set_ef_search, EF, {**info, "search_param_name": "efSearch"})

    for nlist in NLISTS:
        index, secs = build_ivfflat(ds.train, nlist)
        info = build_info(ds, "ivfflat", f"nlist={nlist}", index, secs)
        rows += sweep(index, ds, set_nprobe, NPROBE, {**info, "search_param_name": "nprobe"})

    for m in PQ_M[name]:
        index, secs = build_ivfpq(ds.train, PQ_NLIST, m)
        info = build_info(ds, "ivfpq", f"nlist={PQ_NLIST},m={m}", index, secs)
        rows += sweep(index, ds, set_nprobe, NPROBE[:-3], {**info, "search_param_name": "nprobe"})

    save_rows("part2", rows)       # saved after each dataset so a crash does not lose everything
