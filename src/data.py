"""Load SIFT / GloVe, split the queries in half and compute exact ground truth."""
import urllib.request
from dataclasses import dataclass

import faiss
import h5py
import numpy as np

import config


def download(name):
    path = config.DATA_DIR / config.DATASETS[name]
    if not path.exists():
        print(f"downloading {path.name} ...")
        urllib.request.urlretrieve(config.BASE_URL + path.name, path)
    return path


def normalise(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@dataclass
class Dataset:
    name: str
    train: np.ndarray       # vectors to index
    tune_q: np.ndarray      # queries used to choose parameters
    final_q: np.ndarray     # queries used for the reported numbers
    gt_tune: tuple          # (ids, squared distances) of the 100 true neighbours
    gt_final: tuple
    stored_final: np.ndarray  # neighbours stored in the file (None for a subset of the data)


def exact_ground_truth(train, queries, k=config.GT_K):
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    dists, ids = index.search(queries, k)
    return ids, dists


def load(name):
    with h5py.File(download(name), "r") as f:
        train = np.array(f["train"], dtype="float32")
        test = np.array(f["test"], dtype="float32")
        stored = np.array(f["neighbors"])
    if name == "glove":                      # angular distance -> Euclidean on unit vectors
        train, test = normalise(train), normalise(test)

    n_total = len(test)                      # 10,000 queries: first half tunes, last half reports
    half = n_total // 2
    tune_idx = np.arange(config.N_HALF)
    final_idx = half + np.arange(config.N_HALF)

    full = config.N_TRAIN is None
    if not full:
        train = train[: config.N_TRAIN]
    train = np.ascontiguousarray(train)

    # exact ground truth for all queries, cached on disk (depends on how many vectors we index)
    cache = config.DATA_DIR / f"gt_{name}_{len(train)}.npz"
    if cache.exists():
        z = np.load(cache)
        ids, dists = z["ids"], z["dists"]
    else:
        ids, dists = exact_ground_truth(train, test)
        np.savez(cache, ids=ids, dists=dists)

    return Dataset(
        name=name,
        train=train,
        tune_q=np.ascontiguousarray(test[tune_idx]),
        final_q=np.ascontiguousarray(test[final_idx]),
        gt_tune=(ids[tune_idx], dists[tune_idx]),
        gt_final=(ids[final_idx], dists[final_idx]),
        stored_final=stored[final_idx] if full else None,
    )
