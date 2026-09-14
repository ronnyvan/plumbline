"""Cue engine: registry dispatch plus the gates that exist today (spec §6, §7.1).

Order per cue: observability (geometry, before inference) -> detector -> quality ->
validation. A claim is displayable only if every stage passes. Until thresholds are
selected on a validation split (M4) and confidence is calibrated (M6), that is never —
and the report says so rather than showing uncalibrated numbers as findings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.cues import lp01_torso_swing
from core.cues.contract import Abstention, CueResult, NotComputable
from core.features.quality import window_quality
from core.features.signals import FeatureSet
from core.gates.observability import observability, remedy
from core.registry import Registry
from core.segment.reps import Segmentation

DETECTORS = {
    lp01_torso_swing.CUE_ID: lp01_torso_swing.detect,
}


@dataclass(frozen=True)
class EngineOutput:
    results: list[CueResult] = field(default_factory=list)       # everything computed (diagnostic)
    displayable: list[CueResult] = field(default_factory=list)   # passed every gate
    abstentions: list[Abstention] = field(default_factory=list)
    not_assessed: list[dict] = field(default_factory=list)


def run_cues(exercise_id: str, features: FeatureSet, seg: Segmentation, registry: Registry, *,
             station_used: str | None, station_verified: bool,
             sharpness: np.ndarray | None = None) -> EngineOutput:
    out = EngineOutput()
    for cue in registry.cues_for(exercise_id):
        detector = DETECTORS.get(cue.id)
        if cue.detector == "planned" or detector is None:
            out.not_assessed.append({"cue_id": cue.id, "reason": "detector_not_built",
                                     "expected_tier": cue.expected_tier})
            continue

        obs = observability(cue, station_used, station_verified)
        if obs.value == 0.0:
            out.abstentions.append(Abstention(cue.id, "observability", obs.reason, obs.detail,
                                              remedy(cue, obs, registry)))
            continue

        det = detector(features, seg, cue.params)
        if isinstance(det, NotComputable):
            out.abstentions.append(Abstention(cue.id, "quality", det.reason, {}, None))
            continue

        q = window_quality(features.signals[det.signal], features.confidence[det.signal],
                           features.present, *det.window, sharpness=sharpness)
        result = CueResult(cue.id, det.fired, det.severity, det.raw_score, None, obs.value, q.q,
                           det.evidence, det.per_rep)
        out.results.append(result)

        if q.q < cue.params["quality_threshold"]:
            out.abstentions.append(Abstention(cue.id, "quality", "low_perception_quality",
                                              q.as_dict(), None))
        elif cue.provisional or result.confidence is None:
            out.not_assessed.append({
                "cue_id": cue.id, "reason": "not_validated",
                "detail": {"threshold_provisional": cue.provisional,
                           "confidence_calibrated": result.confidence is not None},
                "expected_tier": cue.expected_tier})
        else:  # pragma: no cover - unreachable until M4/M6 ship calibrated, validated cues
            out.displayable.append(result)
    return out
