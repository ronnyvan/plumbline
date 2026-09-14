"""The detector contract (spec §6.1, D-19).

The spec says a detector "never sets its own confidence, and never decides whether to
abstain". D-19 makes that a property of the types rather than a convention: detectors
return `Detection`, which has no confidence, observability or quality fields. Only the
engine constructs `CueResult`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class Evidence:
    frame_indices: tuple[int, ...]
    measured: float
    threshold: float
    units: str


@dataclass(frozen=True)
class Detection:
    """Everything a detector may say. Nothing about confidence or abstention."""
    fired: bool
    severity: float
    raw_score: float
    evidence: Evidence
    per_rep: tuple[float, ...]
    window: tuple[int, int]      # frames Q is computed over (§5.6)
    signal: str                  # feature signal Q is computed over


@dataclass(frozen=True)
class NotComputable:
    reason: str


@dataclass(frozen=True)
class CueResult:
    """§6.1. `confidence` stays None until calibration exists (§7.2, M6)."""
    cue_id: str
    fired: bool
    severity: float
    raw_score: float
    confidence: float | None
    observability: float
    quality: float
    evidence: Evidence
    per_rep: tuple[float, ...]

    def as_dict(self) -> dict:
        return asdict(self)


Gate = Literal["observability", "quality", "confidence"]


@dataclass(frozen=True)
class Abstention:
    """§7.5: every abstention carries its gate, its reason, and — where one exists — a remedy."""
    cue_id: str
    gate: Gate
    reason: str
    detail: dict
    remedy: dict | None

    def as_dict(self) -> dict:
        return asdict(self)
