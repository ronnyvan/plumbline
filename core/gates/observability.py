"""Observability gate O (spec §7.1, §4.3).

O is geometry and runs before any detector, so an unobservable cue never pays for
inference. Until equipment anchoring exists (M5), the station a clip was shot from
cannot be verified — and §4.3 says station-dependent cues must then abstain. They do.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.registry import ANY_STATION, Cue, Registry


@dataclass(frozen=True)
class Observability:
    value: float
    reason: str | None = None
    detail: dict = field(default_factory=dict)


def observability(cue: Cue, station_used: str | None, station_verified: bool) -> Observability:
    if cue.station == ANY_STATION:
        return Observability(1.0)
    if not station_verified:
        return Observability(0.0, "station_unverified",
                             {"required": cue.station, "claimed": station_used})
    if station_used != cue.station:
        return Observability(0.0, "requires_station",
                             {"required": cue.station, "detected": station_used})
    return Observability(1.0)


def remedy(cue: Cue, obs: Observability, registry: Registry) -> dict | None:
    if obs.reason == "requires_station":
        target = registry.stations[cue.station]
        return {"action": "reposition_camera", "target_station": target.id,
                "copy": f"Checking “{cue.name.lower()}” needs a different camera position. "
                        f"{target.placement}"}
    if obs.reason == "station_unverified":
        # Honest: the user cannot fix this; the capability does not exist yet (M5).
        return {"action": "none",
                "copy": "We can't yet confirm where the camera was, so checks that depend "
                        "on camera position were skipped."}
    return None
