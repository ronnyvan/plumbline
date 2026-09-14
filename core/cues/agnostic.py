"""Tier B, class-agnostic metrics (spec §6.3).

One implementation for all twelve exercises, no labels required. Every metric carries
its own availability and reason, because a metric silently missing is indistinguishable
from a metric silently wrong.
"""
from __future__ import annotations

import numpy as np

from core.features.signals import SIGNAL_UNITS, FeatureSet
from core.segment.reps import Segmentation

AGNOSTIC_VERSION = "3"   # v2: timing reliability gate; v3: phase-label gate (F-04)
MIN_REPS_CONSISTENCY = 3
MIN_REPS_SLOPE = 4
SYMMETRY_STATIONS = ("S-FRONT", "S-REAR")
# Rep counts survive noise that ruins phase timing. Above this noise-to-range ratio, the
# synthetic sweep puts worst-case p95 boundary error past 300 ms, so timing metrics are
# withheld. Provisional: re-derive against hand-annotated boundaries at M1.
TIMING_MAX_NOISE_TO_RANGE = 0.015


def _metric(value, units: str, reason: str | None = None) -> dict:
    if value is None:
        return {"value": None, "units": units, "available": False, "reason": reason}
    if isinstance(value, float):
        value = round(value, 4)
    elif isinstance(value, list):
        value = [round(v, 4) if isinstance(v, float) else v for v in value]
    return {"value": value, "units": units, "available": True, "reason": None}


def _rom(signal: np.ndarray, seg: Segmentation) -> list[float]:
    out = []
    for r in seg.reps:
        base = np.nanmean([signal[r.start], signal[r.end]])
        turn = np.nanmean(signal[r.turn_arrive:r.turn_leave + 1])
        out.append(float(abs(turn - base)))
    return out


def _slope(values: list[float]) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.asarray(values, float), 1)[0])


def compute(features: FeatureSet, seg: Segmentation, *, station_used: str | None,
            station_verified: bool) -> dict:
    units = SIGNAL_UNITS[seg.signal]
    n = len(seg.reps)
    out: dict = {"reps": _metric(n, "count"), "partial_reps": _metric(seg.partial_reps, "count")}
    if n == 0:
        reason = seg.reason or "no_complete_reps"
        for key, u in (("rom_per_rep", units), ("rom_cv", "ratio"), ("tempo", "s"), ("tut", "s"),
                       ("fatigue_slope_rom", f"{units}/rep"), ("fatigue_slope_rom_rel", "fraction/rep"),
                       ("fatigue_slope_velocity", f"{units}/s per rep"), ("asymmetry_rom", "ratio")):
            out[key] = _metric(None, u, reason)
        return out

    rom = _rom(features.signals[seg.signal], seg)
    out["rom_per_rep"] = _metric(rom, units)
    if n >= MIN_REPS_CONSISTENCY and np.mean(rom) > 0:
        out["rom_cv"] = _metric(float(np.std(rom) / np.mean(rom)), "ratio")
    else:
        out["rom_cv"] = _metric(None, "ratio", f"needs_{MIN_REPS_CONSISTENCY}_reps")

    timing_ok = seg.noise_to_range is not None and seg.noise_to_range <= TIMING_MAX_NOISE_TO_RANGE
    # When the start position is ambiguous (clip opens mid-rep), the phases may be swapped.
    # TUT sums both phases, so it survives; anything that says which phase is which does not.
    phases_ok = seg.orientation_confident
    timing_reason = "too_noisy_for_timing" if not timing_ok else "set_start_ambiguous"
    if timing_ok and phases_ok:
        tempo = []
        for i, r in enumerate(seg.reps):
            nxt = seg.reps[i + 1].start if i + 1 < n else None
            tempo.append({"concentric_s": round(r.concentric_s, 3), "eccentric_s": round(r.eccentric_s, 3),
                          "turnaround_pause_s": round(r.turnaround_pause_s, 3),
                          "start_pause_s": round((nxt - r.end) / features.fps, 3) if nxt is not None else None,
                          "first_phase": r.first_phase})
        out["tempo"] = _metric(tempo, "s")
    else:
        out["tempo"] = _metric(None, "s", timing_reason)
    if timing_ok:
        out["tut"] = _metric(float(sum(r.concentric_s + r.eccentric_s for r in seg.reps)), "s")
    else:
        out["tut"] = _metric(None, "s", "too_noisy_for_timing")

    if n >= MIN_REPS_SLOPE:
        velocity = [rom[i] / r.concentric_s for i, r in enumerate(seg.reps) if r.concentric_s > 0]
        out["fatigue_slope_rom"] = _metric(_slope(rom), f"{units}/rep")
        out["fatigue_slope_rom_rel"] = _metric(_slope(rom) / float(np.mean(rom)), "fraction/rep")
        if not (timing_ok and phases_ok):
            out["fatigue_slope_velocity"] = _metric(None, f"{units}/s per rep", timing_reason)
        elif len(velocity) >= MIN_REPS_SLOPE:
            out["fatigue_slope_velocity"] = _metric(_slope(velocity), f"{units}/s per rep")
        else:
            out["fatigue_slope_velocity"] = _metric(None, f"{units}/s per rep", "zero_duration_phase")
    else:
        for key, u in (("fatigue_slope_rom", f"{units}/rep"), ("fatigue_slope_rom_rel", "fraction/rep"),
                       ("fatigue_slope_velocity", f"{units}/s per rep")):
            out[key] = _metric(None, u, f"needs_{MIN_REPS_SLOPE}_reps")

    out["asymmetry_rom"] = _asymmetry(features, seg, station_used, station_verified)
    return out


def _asymmetry(features: FeatureSet, seg: Segmentation, station_used: str | None,
               station_verified: bool) -> dict:
    if not seg.signal.endswith("_angle_mean"):
        return _metric(None, "ratio", "signal_has_no_sides")
    if station_used not in SYMMETRY_STATIONS:
        return _metric(None, "ratio", "requires_frontal_or_rear_station")
    if not station_verified:
        return _metric(None, "ratio", "station_unverified")
    joint = seg.signal.removesuffix("_angle_mean")
    left, right = features.signals[f"{joint}_angle_left"], features.signals[f"{joint}_angle_right"]
    if np.isfinite(left).mean() < 0.8 or np.isfinite(right).mean() < 0.8:
        return _metric(None, "ratio", "one_side_not_visible")
    rl, rr = _rom(left, seg), _rom(right, seg)
    idx = [abs(a - b) / ((a + b) / 2) for a, b in zip(rl, rr, strict=True) if (a + b) > 0]
    return _metric(float(np.median(idx)), "ratio") if idx else _metric(None, "ratio", "zero_rom")
