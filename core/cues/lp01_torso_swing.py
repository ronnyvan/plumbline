"""LP-01 — lat pulldown torso swing / momentum (CMP, S-LAT). Rule form (D-08).

D-20 departs from the §6.2 pseudo-code, which used the median per-rep amplitude. The
dominant real pattern is a clean set that turns into swinging on its last few reps as the
lifter fatigues; a median over the set hides exactly that, and contradicts the §11.7
test the spec itself specifies. The statistic is instead the k-th largest per-rep
amplitude with k = max(min_reps, ceil(min_reps_frac * n)): a cluster of swinging reps
fires, a single noisy rep does not.
"""
from __future__ import annotations

import math

import numpy as np

from core.cues.contract import Detection, Evidence, NotComputable
from core.features.signals import FeatureSet
from core.segment.reps import Segmentation

CUE_ID = "LP-01"
SIGNAL = "torso_angle"


def detect(features: FeatureSet, seg: Segmentation, params: dict) -> Detection | NotComputable:
    if not seg.reps:
        return NotComputable("no_complete_reps")
    theta = features.signals[SIGNAL]
    amps = []
    for r in seg.reps:
        window = theta[r.start:r.end + 1]
        if np.isfinite(window).mean() < 0.5:
            amps.append(float("nan"))
        else:
            amps.append(float(np.nanmax(window) - np.nanmin(window)))
    finite = [a for a in amps if math.isfinite(a)]
    k = max(int(params["min_reps"]), math.ceil(params["min_reps_frac"] * len(seg.reps)))
    if len(finite) < k:
        return NotComputable("torso_not_visible_on_enough_reps")

    score = sorted(finite, reverse=True)[k - 1]
    threshold = float(params["threshold_deg"])
    fired = score > threshold
    severity = float(np.clip((score - threshold) / params["severity_span_deg"], 0, 1)) if fired else 0.0

    worst = int(np.nanargmax(amps))
    r = seg.reps[worst]
    local = theta[r.start:r.end + 1]
    peak_frame = r.start + int(np.nanargmax(np.abs(local - local[np.isfinite(local)][0])))
    return Detection(
        fired=fired, severity=severity, raw_score=score,
        evidence=Evidence((r.start, peak_frame, r.end), round(score, 2), threshold, "deg"),
        per_rep=tuple(round(a, 2) if math.isfinite(a) else float("nan") for a in amps),
        window=(seg.reps[0].start, seg.reps[-1].end), signal=SIGNAL,
    )
