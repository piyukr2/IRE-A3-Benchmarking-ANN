"""Part 3 (Compression, on SIFT): vary the number of PQ sub-vectors m, and compare plain
IVF-PQ with IVF-PQ followed by exact re-ranking of the top 100 candidates."""
import config
import data
from bench import build_info, sweep
from indexes import build_ivfpq, set_nprobe
from results import save_rows

NLIST = 256 if config.QUICK else 1024
M_VALUES = [8, 16, 32, 64]
NPROBE = [1, 2, 4, 8, 16, 32, 64, 128]

ds = data.load("sift")
rows = []
for m in M_VALUES:
    for refine in (False, True):
        index, secs = build_ivfpq(ds.train, NLIST, m, refine=refine)
        name = "ivfpq+refine" if refine else "ivfpq"
        info = build_info(ds, name, f"nlist={NLIST},m={m}", index, secs)
        rows += sweep(index, ds, set_nprobe, NPROBE, {**info, "m": m, "search_param_name": "nprobe"})
save_rows("part3", rows)
