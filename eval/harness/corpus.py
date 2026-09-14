"""Evaluation corpora.

Synthetic: generated deterministically from a pinned seed, never stored.
Real: hand-placed clips under data/clips/, each with a <name>.labels.json sidecar
(format in data/clips/README.md). A clip without recorded analysis consent is refused
before its bytes are read (§9.2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from eval.harness.splits import assign_split
from eval.harness.synthetic import TEMPLATES, SynthParams, generate

CORPUS_SEED = 20260913
GYMS = {  # gym -> (keypoint noise px, detector miss rate); gyms differ like lighting does
    "syn-g1": (1.0, 0.00), "syn-g2": (1.5, 0.01), "syn-g3": (2.5, 0.03), "syn-g4": (4.0, 0.05),
}


@dataclass(frozen=True)
class SyntheticItem:
    item_id: str
    subject_id: str
    gym_id: str
    exercise_id: str
    split: str
    params: SynthParams


def synthetic_items(n_subjects: int = 40, clips_per_subject: int = 3,
                    seed: int = CORPUS_SEED) -> list[SyntheticItem]:
    rng = np.random.default_rng(seed)
    exercises = sorted(TEMPLATES)
    gyms = sorted(GYMS)
    items = []
    for s in range(n_subjects):
        subject = f"syn-s{s:03d}"
        gym = gyms[int(rng.integers(len(gyms)))]
        noise, miss = GYMS[gym]
        torso_px = float(rng.uniform(240, 380))
        for c in range(clips_per_subject):
            ex = exercises[int(rng.integers(len(exercises)))]
            n_reps = int(rng.integers(4, 13))
            swing = ex == "lat_pulldown" and rng.random() < 0.25
            last_k = int(rng.integers(2, 4))
            params = SynthParams(
                exercise_id=ex, n_reps=n_reps,
                concentric_s=float(rng.uniform(0.8, 1.8)), eccentric_s=float(rng.uniform(1.2, 3.0)),
                turn_pause_s=float(rng.uniform(0.0, 0.8)), start_pause_s=float(rng.uniform(0.1, 1.5)),
                rom_decay_per_rep=float(rng.uniform(0.0, 0.04)),
                swing_deg=float(rng.uniform(12, 25)) if swing else 0.0,
                swing_reps=tuple(range(max(0, n_reps - last_k), n_reps)) if swing else (),
                noise_px=noise, miss_rate=miss, partial_first=bool(rng.random() < 0.15),
                bystander=bool(rng.random() < 0.20), torso_px=torso_px,
                hip_xy=(float(rng.uniform(380, 620)), float(rng.uniform(950, 1150))),
                seed=int(rng.integers(0, 2 ** 31)),
            )
            items.append(SyntheticItem(f"{subject}-c{c}", subject, gym, ex, assign_split(subject), params))
    return items


def synthetic_truth_reps(clip) -> list[dict]:
    return [{"start": r.start, "turn_arrive": r.turn_arrive, "turn_leave": r.turn_leave,
             "end": r.end, "first_phase": r.first_phase} for r in clip.truth.reps]


# --- real, hand-placed clips --------------------------------------------------------------

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Consent(_Strict):
    analysis: bool
    training: bool = False
    publication: bool = False


class LabeledRep(_Strict):
    start_s: float
    turn_arrive_s: float
    turn_leave_s: float
    end_s: float
    first_phase: Literal["concentric", "eccentric"]


class ClipLabels(_Strict):
    clip: str
    subject_id: str
    gym_id: str
    exercise: str
    station: str
    consent: Consent
    annotator: str
    reps: list[LabeledRep] = Field(min_length=1)
    partial_reps: int = 0


@dataclass(frozen=True)
class RealItem:
    labels_path: Path
    video_path: Path
    labels: ClipLabels
    split: str


def real_items(clips_dir: Path) -> tuple[list[RealItem], list[dict]]:
    items, refused = [], []
    for path in sorted(Path(clips_dir).glob("*.labels.json")):
        try:
            labels = ClipLabels(**json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:  # malformed labels are reported, not silently skipped
            refused.append({"labels": path.name, "reason": "invalid_labels", "detail": str(e)[:300]})
            continue
        if not labels.consent.analysis:
            refused.append({"labels": path.name, "reason": "no_analysis_consent"})
            continue
        video = path.parent / labels.clip
        if not video.exists():
            refused.append({"labels": path.name, "reason": "video_missing", "detail": labels.clip})
            continue
        items.append(RealItem(path, video, labels, assign_split(labels.subject_id)))
    return items, refused


def real_truth_reps(labels: ClipLabels, timestamps_s: list[float]) -> list[dict]:
    """Convert labeled seconds to output-frame indices using ingest's actual timestamps."""
    ts = np.asarray(timestamps_s)

    def frame(t: float) -> int:
        return int(np.argmin(np.abs(ts - t)))

    return [{"start": frame(r.start_s), "turn_arrive": frame(r.turn_arrive_s),
             "turn_leave": frame(r.turn_leave_s), "end": frame(r.end_s),
             "first_phase": r.first_phase} for r in labels.reps]


__all__ = ["synthetic_items", "synthetic_truth_reps", "real_items", "real_truth_reps", "generate",
           "SyntheticItem", "RealItem", "ClipLabels", "CORPUS_SEED"]
