"""Build the three index types with FAISS. Each builder returns (index, build_seconds)."""
import time

import faiss
import numpy as np

import config


def _timed(build):
    t0 = time.perf_counter()
    index = build()
    return index, time.perf_counter() - t0


def _sample_for_training(train, n):
    rng = np.random.default_rng(config.SEED)
    n = min(n, len(train))
    return train[rng.choice(len(train), n, replace=False)]


def build_flat(train):
    def build():
        index = faiss.IndexFlatL2(train.shape[1])
        index.add(train)
        return index
    return _timed(build)


def build_hnsw(train, M, ef_construction):
    def build():
        index = faiss.IndexHNSWFlat(train.shape[1], M)
        index.hnsw.efConstruction = ef_construction
        index.add(train)
        return index
    return _timed(build)


def build_ivfflat(train, nlist):
    def build():
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(train.shape[1]), train.shape[1], nlist)
        index.train(_sample_for_training(train, 100 * nlist))   # k-means for the cells
        index.add(train)
        return index
    return _timed(build)


def build_ivfpq(train, nlist, m, refine=False):
    """IVF + product quantisation: each vector becomes m one-byte codes.
    refine=True keeps the full vectors too and re-ranks the best 100 candidates exactly."""
    d = train.shape[1]

    def build():
        ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, m, 8)
        if refine:
            index = faiss.IndexRefineFlat(ivfpq)
            index.k_factor = 10                              # top 10 x 10 = 100 candidates
        else:
            index = ivfpq
        index.train(_sample_for_training(train, 100 * nlist))
        index.add(train)
        return index
    return _timed(build)


def size_bytes(index):
    """Size of the index as saved to disk (this includes the vectors for Flat/HNSW/refine)."""
    return faiss.serialize_index(index).nbytes


# --- search-time knobs --------------------------------------------------------------------
def set_ef_search(index, value):
    index.hnsw.efSearch = value


def set_nprobe(index, value):
    faiss.extract_index_ivf(index).nprobe = value            # also works through the refine wrapper
