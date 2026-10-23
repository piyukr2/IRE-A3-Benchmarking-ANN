"""Our own recall@k and timing code."""
import time
from contextlib import contextmanager

import faiss
import numpy as np


def recall_at_k(found, gt_ids, gt_dists, k=10, rel_eps=1e-5):
    """Average fraction of the true k nearest neighbours that appear in `found[:, :k]`.

    Ties: several vectors can be exactly as far as the k-th true neighbour. Neighbours
    strictly closer than that distance must all be found; the tied ones are
    interchangeable, so they may fill whatever slots are left. Distances are compared
    with a tiny tolerance for float rounding. `gt_ids` / `gt_dists` need more than k
    columns (we store 100) so that ties beyond position k are visible.
    """
    total = 0
    for f, ids, d in zip(found[:, :k], gt_ids, gt_dists):
        tol = d[k - 1] * rel_eps + 1e-7
        strict = set(ids[d < d[k - 1] - tol].tolist())          # clearly closer than the k-th
        tied = set(ids[np.abs(d - d[k - 1]) <= tol].tolist())   # as far as the k-th
        f = set(f.tolist())                                     # a set: repeated ids count once
        total += len(f & strict) + min(len(f & tied), k - len(strict))
    return total / (len(found) * k)


@contextmanager
def single_thread():
    """Run the block with FAISS limited to one thread, then restore the old setting."""
    old = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(old)


def median_qps(search_one, queries, repeats=3):
    """Queries per second when searching ONE query per call. Median over `repeats` passes."""
    singles = [np.ascontiguousarray(q[None, :]) for q in queries]
    for q in singles[:20]:                            # warm-up
        search_one(q)
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in singles:
            search_one(q)
        rates.append(len(singles) / (time.perf_counter() - t0))
    return float(np.median(rates))


def median_batch_qps(search_batch, queries, repeats=3):
    """Queries per second when ALL queries are sent in one call. Median over `repeats`."""
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        search_batch(queries)
        rates.append(len(queries) / (time.perf_counter() - t0))
    return float(np.median(rates))
