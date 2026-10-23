"""Synthetic data for Part 4 (seeded, so every run produces exactly the same vectors)."""
import numpy as np

import config


def near_duplicates(base, copies, sigma=1.0, seed=config.SEED):
    """Return `base` plus (copies - 1) noisy clones of EVERY base vector.

    Each clone is the original plus Gaussian noise with standard deviation `sigma` per
    coordinate. For SIFT (coordinates 0..255, typical nearest-neighbour distance ~ 150) a
    sigma of 1 puts clones ~16 away from their original, far closer than any other vector.
    copies=1 returns the untouched real vectors (the control).
    """
    rng = np.random.default_rng(seed)
    parts = [base]
    for _ in range(copies - 1):
        parts.append(base + rng.normal(0.0, sigma, base.shape).astype("float32"))
    return np.ascontiguousarray(np.vstack(parts), dtype="float32")
