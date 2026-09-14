"""Exercise, cue and station registry (spec §3.1, §3.3, §3.4).

The registry is data. This module loads it, validates every cross-reference, and
fails loudly on inconsistency — a cue that names a station that does not exist is a
spec bug, and it should break the build rather than silently abstain forever.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REGISTRY_ROOT = Path(__file__).resolve().parent.parent / "registry"

CueType = Literal["SET", "TMP", "ROM", "CMP", "SYM", "KIN"]
ExpectedTier = Literal["A", "B", "abstain"]
Direction = Literal["increasing", "decreasing"]
SignalName = Literal[
    "elbow_angle_mean", "knee_angle_mean", "hip_angle_mean",
    "shoulder_angle_mean", "wrist_separation",
]
ANY_STATION = "any"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Station(_Strict):
    id: str
    azimuth_deg: float
    elevation_deg: float
    lens_height: str
    distance_m: tuple[float, float]
    resolves: str
    placement: str


class Cue(_Strict):
    id: str
    exercise: str
    name: str
    type: CueType
    station: str
    expected_tier: ExpectedTier
    detector: Literal["planned", "implemented"] = "planned"
    params: dict | None = None

    @model_validator(mode="after")
    def _params_match_status(self) -> Cue:
        if self.detector == "implemented" and not self.params:
            raise ValueError(f"{self.id}: implemented detector needs params")
        if self.detector == "planned" and self.params:
            raise ValueError(f"{self.id}: planned cue must not carry params")
        if self.params and "provisional" not in self.params:
            raise ValueError(f"{self.id}: params must state provisional true/false (D-18)")
        return self

    @property
    def provisional(self) -> bool:
        return bool(self.params and self.params.get("provisional", True))


class RepPhase(_Strict):
    concentric: Direction
    eccentric: Direction

    @model_validator(mode="after")
    def _opposite(self) -> RepPhase:
        if self.concentric == self.eccentric:
            raise ValueError("concentric and eccentric must move in opposite directions")
        return self


class StationPlan(_Strict):
    required: str
    optional: list[str] = Field(default_factory=list)


class RomReference(_Strict):
    joint: str
    full_extension_deg: float | None
    normalisation: Literal["torso_length"]


class Exercise(_Strict):
    id: str
    name: str
    aliases: list[str]
    equipment_class: str
    dominant_plane: Literal["sagittal", "frontal", "transverse"]
    primary_joints: list[str]
    rep_signal: SignalName
    rep_phase: RepPhase
    stations: StationPlan
    anchors: list[str]
    rom_reference: RomReference
    cues: list[str]


class Promotion(_Strict):
    validated: bool
    evidence: str | None

    @model_validator(mode="after")
    def _evidence_required(self) -> Promotion:
        if self.validated and not self.evidence:
            raise ValueError("a validated promotion must cite committed evidence")
        return self


class Registry(_Strict):
    stations: dict[str, Station]
    cues: dict[str, Cue]
    exercises: dict[str, Exercise]
    tier_b: dict[str, Promotion]
    tier_a_cues: dict[str, Promotion]
    content_hash: str

    def cues_for(self, exercise_id: str) -> list[Cue]:
        return [self.cues[c] for c in self.exercises[exercise_id].cues]


def _validate(stations: dict, cues: dict, exercises: dict) -> None:
    problems: list[str] = []
    for cue in cues.values():
        if cue.exercise not in exercises:
            problems.append(f"{cue.id}: unknown exercise {cue.exercise}")
        if cue.station != ANY_STATION and cue.station not in stations:
            problems.append(f"{cue.id}: unknown station {cue.station}")
    for ex in exercises.values():
        for sid in [ex.stations.required, *ex.stations.optional]:
            if sid not in stations:
                problems.append(f"{ex.id}: unknown station {sid}")
        for cid in ex.cues:
            if cid not in cues:
                problems.append(f"{ex.id}: lists unknown cue {cid}")
            elif cues[cid].exercise != ex.id:
                problems.append(f"{ex.id}: lists {cid}, which belongs to {cues[cid].exercise}")
    for cue in cues.values():
        if cue.exercise in exercises and cue.id not in exercises[cue.exercise].cues:
            problems.append(f"{cue.id}: not listed by exercise {cue.exercise}")
    if problems:
        raise ValueError("registry is inconsistent:\n  " + "\n  ".join(problems))


def _hash_files(root: Path, paths: list[Path]) -> str:
    """Line-ending-insensitive, so Windows and Linux checkouts produce the same hash."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def load_registry(root: Path = REGISTRY_ROOT) -> Registry:
    station_file = root / "stations.yaml"
    cue_file = root / "cues.yaml"
    exercise_files = sorted((root / "exercises").glob("*.yaml"))

    stations = {s["id"]: Station(**s) for s in yaml.safe_load(station_file.read_text(encoding="utf-8"))["stations"]}
    cue_rows = yaml.safe_load(cue_file.read_text(encoding="utf-8"))["cues"]
    cues: dict[str, Cue] = {}
    for row in cue_rows:
        if row["id"] in cues:
            raise ValueError(f"duplicate cue id {row['id']}")
        cues[row["id"]] = Cue(**row)
    exercises: dict[str, Exercise] = {}
    for f in exercise_files:
        ex = Exercise(**yaml.safe_load(f.read_text(encoding="utf-8")))
        if ex.id != f.stem:
            raise ValueError(f"{f.name}: id {ex.id} must match file name")
        exercises[ex.id] = ex

    _validate(stations, cues, exercises)
    validation_file = root / "validation.yaml"
    validation = yaml.safe_load(validation_file.read_text(encoding="utf-8"))
    tier_b = {k: Promotion(**v) for k, v in validation["tier_b"].items()}
    tier_a = {k: Promotion(**v) for k, v in (validation.get("tier_a_cues") or {}).items()}
    if set(tier_b) != set(exercises):
        raise ValueError("validation.yaml tier_b must list exactly the registry's exercises")
    unknown = set(tier_a) - set(cues)
    if unknown:
        raise ValueError(f"validation.yaml tier_a_cues names unknown cues: {sorted(unknown)}")
    return Registry(
        stations=stations, cues=cues, exercises=exercises, tier_b=tier_b, tier_a_cues=tier_a,
        content_hash=_hash_files(root, [station_file, cue_file, validation_file, *exercise_files]),
    )


@lru_cache(maxsize=1)
def default_registry() -> Registry:
    return load_registry()
