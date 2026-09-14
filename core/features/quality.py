"""Perception quality Q over a cue window (spec §5.6, §7.1).

Q is computed per cue, over that cue's frames only: a clip can be adequate for ROM and
useless for symmetry. The four factors multiply, so any one of them can veto.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Variance-of-Laplacian at which sharpness reads 0.5. Provisional; checked in the L1 study.
SHARPNESS_HALF_POINT = 100.0


def sharpness_from_laplacian_var(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (v + SHARPNESS_HALF_POINT)


@dataclass(frozen=True)
class Quality:
    q: float
    confidence: float
    visibility: float
    sharpness: float
    stability: float

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def window_quality(signal: np.ndarray, confidence: np.ndarray, present: np.ndarray,
                   start: int, end: int, sharpness: np.ndarray | None = None) -> Quality:
    """Q over frames [start, end] inclusive."""
    sl = slice(max(start, 0), end + 1)
    finite = np.isfinite(signal[sl])
    if finite.size == 0:
        return Quality(0.0, 0.0, 0.0, 0.0, 0.0)
    visibility = float(finite.mean())
    conf = float(np.clip(np.mean(confidence[sl][finite]), 0.0, 1.0)) if finite.any() else 0.0
    sharp = float(np.mean(sharpness[sl])) if sharpness is not None else 1.0
    stability = float(np.mean(present[sl]))
    return Quality(conf * visibility * sharp * stability, conf, visibility, sharp, stability)
