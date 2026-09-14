"""Class-agnostic repetition-period cross-check (spec §5.5 fallback, D-17).

This is a temporal self-similarity estimate over pose-derived features, not the RepNet
network. With dot-product similarity on standardised features, the mean of the k-th
diagonal of the self-similarity matrix is the autocorrelation at lag k, so the dominant
period is its strongest peak. It shares no code with the signal-based segmenter, which
is exactly what makes it useful as a disagreement detector.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _standardise(features: np.ndarray, min_range: float = 0.0) -> np.ndarray:
    """Z-score each column. Columns whose robust raw range is below `min_range` are dropped
    first: standardising a near-static signal scales its jitter up to the size of real
    motion, which makes rest look like movement."""
    f = np.asarray(features, dtype=np.float64)
    keep = np.isfinite(f).mean(axis=0) > 0.5
    if min_range > 0:
        with np.errstate(all="ignore"):
            rng = np.nanpercentile(np.where(keep, f, np.nan), 95, axis=0) \
                - np.nanpercentile(np.where(keep, f, np.nan), 5, axis=0)
        keep &= np.nan_to_num(rng, nan=0.0) >= min_range
    f = f[:, keep]
    if f.shape[1] == 0:
        return f
    mu, sd = np.nanmean(f, axis=0), np.nanstd(f, axis=0)
    ok = sd > 1e-9
    z = (f[:, ok] - mu[ok]) / sd[ok]
    return np.nan_to_num(z, nan=0.0)


def estimate_period(features: np.ndarray, fps: float, min_period_s: float = 0.8,
                    max_period_s: float = 10.0, min_range: float = 0.0) -> tuple[float | None, float]:
    """Return (period_seconds, normalised autocorrelation strength)."""
    z = _standardise(features, min_range)
    t, d = z.shape if z.ndim == 2 else (0, 0)
    min_lag, max_lag = max(1, int(min_period_s * fps)), min(int(max_period_s * fps), t // 2)
    if d == 0 or max_lag <= min_lag:
        return None, 0.0
    r0 = float(np.mean(np.sum(z * z, axis=1)))
    if r0 <= 0:
        return None, 0.0
    r = np.array([np.mean(np.sum(z[:-k] * z[k:], axis=1)) for k in range(1, max_lag + 1)]) / r0
    peaks, _ = find_peaks(r[min_lag - 1:], prominence=0.1)
    if len(peaks) == 0:
        return None, 0.0
    lags = peaks + min_lag
    heights = r[lags - 1]
    # Harmonic guard: the earliest peak that is nearly as strong as the strongest one.
    best = lags[np.flatnonzero(heights >= 0.9 * heights.max())[0]]
    return float(best / fps), float(r[best - 1])


def motion_span_frames(features: np.ndarray, activity_frac: float = 0.15,
                       min_range: float = 0.0) -> int:
    z = _standardise(features, min_range)
    if z.size == 0 or len(z) < 3:
        return 0
    speed = np.mean(np.abs(np.diff(z, axis=0)), axis=1)
    kernel = np.ones(5) / 5
    speed = np.convolve(speed, kernel, mode="same")
    active = np.flatnonzero(speed > activity_frac * np.percentile(speed, 95))
    return int(active[-1] - active[0] + 1) if len(active) else 0


def count_repetitions(features: np.ndarray, fps: float, min_range: float = 5.0) -> dict:
    """`min_range` is in the features' raw units (degrees for the default angle signals)."""
    period, strength = estimate_period(features, fps, min_range=min_range)
    span = motion_span_frames(features, min_range=min_range)
    count = int(round(span / (period * fps))) if period else 0
    return {"period_s": round(period, 3) if period else None,
            "strength": round(strength, 3), "count": count}
