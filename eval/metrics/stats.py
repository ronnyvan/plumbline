"""Statistical machinery (spec §10.3).

Clips from one subject are not independent, so intervals resample subjects, not clips.
Every headline number is reported as an interval with its N; a point estimate alone is
not a result.
"""
from __future__ import annotations

import numpy as np

MIN_POSITIVES = 30   # §10.3 minimum-N rule for cue metrics


def cluster_bootstrap(values, clusters, *, weights=None, n_resamples: int = 10_000,
                      alpha: float = 0.05, seed: int = 0) -> dict:
    """(Weighted) mean with a percentile CI from resampling whole clusters with replacement."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if values.size == 0:
        return {"value": None, "ci95": None, "n": 0, "clusters": 0, "method": "cluster_bootstrap"}
    w = np.ones_like(values) if weights is None else np.asarray(weights, dtype=np.float64)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values * w, minlength=len(uniq))
    counts = np.bincount(inv, weights=w, minlength=len(uniq))
    point = float(sums.sum() / counts.sum())

    rng = np.random.default_rng(seed)
    k = len(uniq)
    stats = np.empty(n_resamples)
    chunk = max(1, 2_000_000 // max(k, 1))
    for start in range(0, n_resamples, chunk):
        stop = min(start + chunk, n_resamples)
        draws = rng.integers(0, k, size=(stop - start, k))
        denom = counts[draws].sum(axis=1)
        stats[start:stop] = np.where(denom > 0, sums[draws].sum(axis=1) / np.maximum(denom, 1e-12), np.nan)
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": round(point, 6), "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "n": int(values.size), "clusters": int(k), "method": "cluster_bootstrap_percentile",
            "resamples": n_resamples}


def meets_minimum(n: int, minimum: int = MIN_POSITIVES) -> bool:
    return n >= minimum
