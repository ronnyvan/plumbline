"""Signal-based repetition segmentation (spec §5.5, D-16).

A rep is a cycle that leaves the *observed* start position, reaches a turnaround, and
returns. Which half is concentric comes from the registry's `rep_phase` (anatomy: an
elbow flexes during a curl, always). Which half comes first is read from the signal
(behaviour: a leg press starts at lockout, so its first phase is eccentric). The spec's
original output list assumed concentric-first; D-16 records why that was wrong.

Boundaries are velocity-based rather than position-based: a position threshold at 10%
of travel lags true motion onset by ~20% of the phase duration, while 8% of peak
velocity lags it by ~2.5%.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from scipy.signal import butter, find_peaks, savgol_filter, sosfiltfilt

from core.features.signals import MIN_REP_AMPLITUDE, SIGNAL_UNITS, FeatureSet
from core.segment.selfsim import count_repetitions
from core.staging import StageSpec

SEGMENT_VERSION = "4"   # v2 F-01 transit-bounded velocity; v3 F-02 noise-adaptive prominence; v4 F-04 edge dwell
Phase = Literal["concentric", "eccentric"]

SELFSIM_SIGNALS = ("elbow_angle_mean", "knee_angle_mean", "hip_angle_mean",
                   "shoulder_angle_mean", "torso_angle")


@dataclass(frozen=True)
class SegmentConfig:
    lowpass_hz: float = 3.0
    min_period_s: float = 0.8
    # A cycle counts as a rep when its prominence clears BOTH a fraction of the set's range
    # and a multiple of this clip's own noise (F-02). A fixed fraction cannot serve both
    # heavy fatigue (short final reps) and noisy capture (jitter mistaken for reps).
    # Chosen by sweep on the synthetic stress family (seeds 3000+), never on the eval corpus
    # (CORPUS_SEED) and never on real data. Re-check on hand-annotated clips at M1.
    prominence_frac: float = 0.12
    prominence_noise_k: float = 8.0
    velocity_frac: float = 0.08
    start_window_s: float = 0.5
    orientation_margin_frac: float = 0.20
    edge_valley_frac: float = 0.35
    edge_band_frac: float = 0.20     # "at an extreme" means within 20% of range of it
    dwell_margin_s: float = 0.2
    transit_frac: float = 0.10       # bounds the peak-velocity search to 10-100% of travel
    max_count_disagreement: int = 1

    def stage_spec(self) -> StageSpec:
        return StageSpec("segment", SEGMENT_VERSION, asdict(self))


@dataclass(frozen=True)
class Rep:
    index: int
    start: int          # motion away from the start position begins
    turn_arrive: int    # turnaround reached (end of first phase)
    turn_leave: int     # motion back toward the start position begins
    end: int            # start position re-reached
    first_phase: Phase
    fps: float

    @property
    def concentric(self) -> tuple[int, int]:
        return (self.start, self.turn_arrive) if self.first_phase == "concentric" \
            else (self.turn_leave, self.end)

    @property
    def eccentric(self) -> tuple[int, int]:
        return (self.turn_leave, self.end) if self.first_phase == "concentric" \
            else (self.start, self.turn_arrive)

    @property
    def concentric_s(self) -> float:
        a, b = self.concentric
        return (b - a) / self.fps

    @property
    def eccentric_s(self) -> float:
        a, b = self.eccentric
        return (b - a) / self.fps

    @property
    def turnaround_pause_s(self) -> float:
        return (self.turn_leave - self.turn_arrive) / self.fps


@dataclass(frozen=True)
class Segmentation:
    signal: str
    reps: tuple[Rep, ...]
    partial_reps: int
    start_level: Literal["low", "high"] | None
    orientation_confident: bool
    confidence: float
    reason: str | None
    selfsim: dict = field(default_factory=dict)
    noise_sigma: float | None = None       # robust residual noise, signal units
    noise_to_range: float | None = None

    @property
    def window(self) -> tuple[int, int] | None:
        return (self.reps[0].start, self.reps[-1].end) if self.reps else None


def _empty(signal: str, reason: str) -> Segmentation:
    return Segmentation(signal, (), 0, None, False, 0.0, reason)


def _start_is_low(x: np.ndarray, finite: np.ndarray, fps: float, lo: float, hi: float,
                  cfg: SegmentConfig) -> tuple[bool, bool]:
    """Decide which extreme is the start position. Returns (start_low, confident).

    A set begins and ends at rest in the start position, so both clip edges normally agree.
    When they disagree, the clip begins or ends mid-rep, and the edge is lying; the
    longest still period (the lead-in or lead-out rest) is then the better witness.
    """
    mid, amp = (lo + hi) / 2, hi - lo
    w = max(1, int(cfg.start_window_s * fps))
    idx = np.flatnonzero(finite)
    head = float(np.median(x[idx[0]:idx[0] + w]))
    tail = float(np.median(x[max(idx[-1] - w + 1, 0):idx[-1] + 1]))
    margin = cfg.orientation_margin_frac * amp
    if (head < mid) == (tail < mid):
        return head < mid, abs(head - mid) > margin and abs(tail - mid) > margin

    # Edges disagree: one of them is mid-rep. A clip opening at the turnaround dwells there only
    # for the rest of one turnaround pause; the set's real lead-in or lead-out rest is longer.
    # Contiguous dwell is measured, not "stillness", because jitter fragments still periods
    # on noisy clips (the failure that motivated this).
    band = cfg.edge_band_frac * amp
    near_low, near_high = x <= lo + band, x >= hi - band
    head_mask = near_low if head < mid else near_high
    tail_mask = near_low if tail < mid else near_high
    head_dwell = int(np.argmin(head_mask[idx[0]:])) if not head_mask[idx[0]:].all() else len(x)
    tail_dwell = int(np.argmin(tail_mask[:idx[-1] + 1][::-1])) if not tail_mask[:idx[-1] + 1].all() else len(x)
    if abs(head_dwell - tail_dwell) >= cfg.dwell_margin_s * fps:
        return (head < mid) if head_dwell > tail_dwell else (tail < mid), False
    return bool(near_low.sum() >= near_high.sum()), False


def segment_reps(features: FeatureSet, signal_name: str, concentric_direction: str,
                 cfg: SegmentConfig = SegmentConfig()) -> Segmentation:
    raw = features.signals[signal_name]
    fps = features.fps
    n = len(raw)
    finite = np.isfinite(raw)
    if n < 2 * fps or finite.mean() < 0.5:
        return _empty(signal_name, "insufficient_signal")

    idx = np.arange(n)
    x = np.interp(idx, idx[finite], raw[finite])
    sos = butter(2, cfg.lowpass_hz, fs=fps, output="sos")
    x = sosfiltfilt(sos, x, padlen=min(15, n - 1))

    lo, hi = np.percentile(x, [5, 95])
    amp = hi - lo
    if amp < MIN_REP_AMPLITUDE[SIGNAL_UNITS[signal_name]]:
        return _empty(signal_name, "no_motion")
    raw_interp = np.interp(idx, idx[finite], raw[finite])
    resid = (raw_interp - x)[finite]
    noise_sigma = float(1.4826 * np.median(np.abs(resid - np.median(resid))))

    start_low, orientation_confident = _start_is_low(x, finite, fps, lo, hi, cfg)
    start_low, orientation_confident = bool(start_low), bool(orientation_confident)
    y = x if start_low else -x
    y_lo = lo if start_low else -hi

    distance = max(1, int(cfg.min_period_s * fps))
    prominence = max(cfg.prominence_frac * amp, cfg.prominence_noise_k * noise_sigma)
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance)
    valleys = list(find_peaks(-y, prominence=prominence, distance=distance)[0])
    edge = y_lo + cfg.edge_valley_frac * amp
    if y[0] < edge and (not valleys or valleys[0] > 0):
        valleys.insert(0, 0)
    if y[-1] < edge and (not valleys or valleys[-1] < n - 1):
        valleys.append(n - 1)

    window = min(n - (1 - n % 2), max(5, int(0.2 * fps) | 1))
    vel = savgol_filter(y, window, 2, deriv=1) * fps

    raw_first = "increasing" if start_low else "decreasing"
    first_phase: Phase = "concentric" if raw_first == concentric_direction else "eccentric"

    reps: list[Rep] = []
    bracketed: set[int] = set()
    for v0, v1 in zip(valleys[:-1], valleys[1:], strict=True):
        inside = peaks[(peaks > v0) & (peaks < v1)]
        if len(inside) == 0:
            continue
        bracketed.update(inside.tolist())
        p = int(inside[np.argmax(y[inside])])

        # Peak speed is searched only while the limb is in transit. On the last rep v1 is the
        # clip edge, and an unbounded search would include the whole lead-out rest, where a
        # noise spike can out-run the real movement (failure F-01).
        travel_in = np.flatnonzero(y[v0:p + 1] <= y[v0] + cfg.transit_frac * (y[p] - y[v0]))
        a = v0 + int(travel_in[-1]) if len(travel_in) else v0
        travel_out = np.flatnonzero(y[p:v1 + 1] <= y[v1] + cfg.transit_frac * (y[p] - y[v1]))
        b = p + int(travel_out[0]) if len(travel_out) else v1
        i_max = a + int(np.argmax(vel[a:p + 1]))
        v_up = vel[i_max]
        j_min = p + int(np.argmin(vel[p:b + 1]))
        v_down = vel[j_min]
        if v_up <= 0 or v_down >= 0:
            continue
        up_thr, down_thr = cfg.velocity_frac * v_up, cfg.velocity_frac * v_down

        below = np.flatnonzero(vel[v0:i_max + 1] < up_thr)
        start = v0 + int(below[-1]) if len(below) else v0
        settle = np.flatnonzero(vel[i_max:p + 1] < up_thr)
        turn_arrive = i_max + int(settle[0]) if len(settle) else p
        still = np.flatnonzero(vel[p:j_min + 1] > down_thr)
        turn_leave = p + int(still[-1]) if len(still) else p
        stop = np.flatnonzero(vel[j_min:v1 + 1] > down_thr)
        end = j_min + int(stop[0]) if len(stop) else v1

        reps.append(Rep(len(reps), int(start), int(turn_arrive), int(turn_leave), int(end),
                        first_phase, fps))

    # Partial reps are excluded from set statistics (§5.5) but still counted.
    loose = sorted(set(peaks.tolist()) - bracketed)
    first_v, last_v = (valleys[0], valleys[-1]) if valleys else (n, -1)
    partial_start = bool(y[0] >= edge or any(pk < first_v for pk in loose))
    partial_end = bool(y[-1] >= edge or any(pk > last_v for pk in loose))
    between = sum(1 for pk in loose if first_v < pk < last_v)
    partial = int(partial_start) + int(partial_end) + between

    selfsim_features = np.stack([features.signals[s] for s in SELFSIM_SIGNALS], axis=1)
    cross = count_repetitions(selfsim_features, fps)
    observed = len(reps) + partial
    agrees = abs(cross["count"] - observed) <= cfg.max_count_disagreement
    cross["agrees"] = bool(agrees)

    confidence = 0.0
    if reps:
        durations = np.array([(r.end - r.start) / fps for r in reps])
        cv = float(durations.std() / durations.mean()) if durations.mean() > 0 else 1.0
        conf_sig = features.confidence[signal_name]
        rep_frames = np.concatenate([np.arange(r.start, r.end + 1) for r in reps])
        signal_conf = float(np.clip(np.mean(conf_sig[rep_frames]), 0.0, 1.0))
        confidence = ((1.0 if orientation_confident else 0.7)
                      * (1.0 - 0.5 * min(cv, 1.0))
                      * signal_conf
                      * (1.0 if agrees else 0.5))

    return Segmentation(signal_name, tuple(reps), partial, "low" if start_low else "high",
                        orientation_confident, round(confidence, 4), None, cross,
                        round(noise_sigma, 4), round(noise_sigma / amp, 5))
