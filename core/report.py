"""Report assembly (spec §2.5).

Section order is the order a user should act in: set summary, fired cues, abstained cues
(visible, with remedies), coaching, trend. Abstentions are never hidden (§2.5.3).
"""
from __future__ import annotations

REPORT_SCHEMA_VERSION = 1

TIER_B_NOTE = ("Form-error detection isn't validated for this exercise yet. "
               "Showing rep quality metrics only.")
UNVALIDATED_METRICS_NOTE = ("These rep metrics haven't yet been checked against hand-annotated "
                            "clips, so treat them as provisional.")

SUBMISSION_COPY = {
    "multiple_subjects_ambiguous": "More than one person is moving in the working area, so we "
                                   "couldn't tell who was lifting. Reshoot with only you near the machine.",
    "no_subject_detected": "We couldn't find a person in this clip. Make sure your whole body is in frame.",
    "subject_intermittently_detected": "We kept losing sight of you during the set, so the rep count "
                                       "wouldn't be reliable. Reshoot with your whole body in frame "
                                       "and nothing between you and the phone.",
    "insufficient_signal": "The joints we track for this exercise were hidden for most of the clip. "
                           "Reshoot from the suggested camera position.",
    "no_motion": "We didn't see any repetitions in this clip.",
}


def submission_abstention(reason: str) -> dict:
    return {"reason": reason, "copy": SUBMISSION_COPY.get(reason, "We couldn't analyse this clip.")}


def build(*, registry, exercise_id: str, manifest_hash: str, set_summary: dict | None,
          engine=None, segmentation=None, selection=None, station: dict,
          abstained_reason: str | None = None) -> dict:
    ex = registry.exercises[exercise_id]
    promotion = registry.tier_b[exercise_id]
    fired = [r.as_dict() for r in engine.displayable] if engine else []
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest_hash": manifest_hash,
        "exercise": {
            "id": ex.id, "name": ex.name,
            "tier": "A" if fired else "B",
            "tier_validated": promotion.validated,
            "notes": [n for n in (None if fired else TIER_B_NOTE,
                                  None if promotion.validated else UNVALIDATED_METRICS_NOTE) if n],
        },
        "status": "abstained" if abstained_reason else "analysed",
        "submission_abstention": submission_abstention(abstained_reason) if abstained_reason else None,
        "set_summary": set_summary,
        "fired_cues": fired,
        "abstained_cues": [a.as_dict() for a in engine.abstentions] if engine else [],
        "not_assessed": list(engine.not_assessed) if engine else [],
        "coaching": None,   # M7 (§8)
        "trend": None,      # needs submission history (M8)
        "diagnostics": {
            "station": station,
            "subject_selection": selection,
            "segmentation": _segmentation_dict(segmentation),
            "cue_results": [r.as_dict() for r in engine.results] if engine else [],
        },
    }
    return report


def _segmentation_dict(seg) -> dict | None:
    if seg is None:
        return None
    return {
        "signal": seg.signal, "reason": seg.reason, "confidence": seg.confidence,
        "start_level": seg.start_level, "orientation_confident": seg.orientation_confident,
        "partial_reps": seg.partial_reps, "selfsim": seg.selfsim,
        "noise_sigma": seg.noise_sigma, "noise_to_range": seg.noise_to_range,
        "reps": [{"index": r.index, "start": r.start, "turn_arrive": r.turn_arrive,
                  "turn_leave": r.turn_leave, "end": r.end, "first_phase": r.first_phase}
                 for r in seg.reps],
    }
