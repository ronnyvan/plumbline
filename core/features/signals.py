"""Keypoints to features (spec §5.6).

Honesty constraints, enforced here rather than in documentation:
- Angles are image-plane angles. Nothing in this module produces a 3D anatomical angle.
- Gaps longer than `max_gap_frames` stay NaN. Motion is never invented.
- Smoothing is a centred Savitzky-Golay filter, which has zero group delay offline.
  `phase_lag_frames` records that explicitly so tempo code never "corrects" a lag
  that does not exist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import savgol_filter

from core.keypoints import HIP_CENTRE, INDEX, NECK, TRIPLETS
from core.pose.types import SubjectTrack
from core.staging import StageSpec

FEATURE_VERSION = "1"

SIGNAL_UNITS = {
    "elbow_angle_mean": "deg", "knee_angle_mean": "deg", "hip_angle_mean": "deg",
    "shoulder_angle_mean": "deg", "wrist_separation": "torso_lengths", "torso_angle": "deg",
    **{f"{j}_angle_{s}": "deg" for j in TRIPLETS for s in ("left", "right")},
}
# Below this peak-to-peak range there is no repetition to segment.
MIN_REP_AMPLITUDE = {"deg": 15.0, "torso_lengths": 0.25}


@dataclass(frozen=True)
class FeatureConfig:
    keypoint_floor: float = 0.3      # provisional (Q-08)
    max_gap_frames: int = 4          # §5.6
    sg_window: int = 9               # §5.6, odd, ~0.3 s at 30 fps
    sg_order: int = 2                # §5.6
    side_dominance: float = 1.5      # use one side alone when it is this much more confident

    def stage_spec(self) -> StageSpec:
        return StageSpec("features", FEATURE_VERSION, asdict(self))


@dataclass(frozen=True)
class FeatureSet:
    fps: float
    n_frames: int
    signals: dict[str, np.ndarray]
    confidence: dict[str, np.ndarray]
    present: np.ndarray
    torso_length_px: float
    config: FeatureConfig
    phase_lag_frames: float = 0.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-exclusive (start, stop) spans where mask is True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2], strict=True))


def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate interior NaN gaps of at most `max_gap` samples."""
    out = x.astype(np.float64).copy()
    finite = np.isfinite(out)
    for start, stop in _runs(~finite):
        if start == 0 or stop == len(out) or stop - start > max_gap:
            continue
        a, b = out[start - 1], out[stop]
        out[start:stop] = a + (b - a) * (np.arange(1, stop - start + 1) / (stop - start + 1))
    return out


def smooth(x: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay over each finite run long enough to hold the window."""
    out = x.astype(np.float64).copy()
    for start, stop in _runs(np.isfinite(out)):
        if stop - start >= window:
            out[start:stop] = savgol_filter(out[start:stop], window, order)
    return out


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b in degrees, image plane. NaN where any point is missing."""
    v1, v2 = a - b, c - b
    dot = np.einsum("...i,...i", v1, v2)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def fuse_sides(left: np.ndarray, cl: np.ndarray, right: np.ndarray, cr: np.ndarray,
               dominance: float) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-weighted side fusion. A side much less confident than the other is dropped,
    because from a lateral station the far side is an occluded guess, not a measurement."""
    lv, rv = np.isfinite(left), np.isfinite(right)
    cl = np.where(lv, cl, 0.0)
    cr = np.where(rv, cr, 0.0)
    use_l = lv & ~(rv & (cr > dominance * cl))
    use_r = rv & ~(lv & (cl > dominance * cr))
    wl, wr = np.where(use_l, cl, 0.0), np.where(use_r, cr, 0.0)
    total = wl + wr
    with np.errstate(invalid="ignore", divide="ignore"):
        fused = (np.where(use_l, left, 0.0) * wl + np.where(use_r, right, 0.0) * wr) / total
    # Equal zero weights but a finite value (confidence 0) still count as observed.
    only = (total == 0) & (use_l | use_r)
    fused = np.where(only, np.where(use_l, left, right), fused)
    fused = np.where(use_l | use_r, fused, np.nan)
    return fused, np.maximum(wl, wr)


def extract_features(track: SubjectTrack, cfg: FeatureConfig = FeatureConfig()) -> FeatureSet:
    kp = track.kp.astype(np.float64).copy()
    sc = track.kp_score.astype(np.float64)
    missing = (sc < cfg.keypoint_floor) | ~np.isfinite(kp).all(axis=-1)
    kp[missing] = np.nan
    for k in range(kp.shape[1]):
        for d in range(2):
            kp[:, k, d] = smooth(fill_gaps(kp[:, k, d], cfg.max_gap_frames), cfg.sg_window, cfg.sg_order)

    signals: dict[str, np.ndarray] = {}
    confidence: dict[str, np.ndarray] = {}
    for joint, sides in TRIPLETS.items():
        per_side = {}
        for side, (a, b, c) in sides.items():
            ia, ib, ic = INDEX[a], INDEX[b], INDEX[c]
            ang = joint_angle(kp[:, ia], kp[:, ib], kp[:, ic])
            conf = np.minimum(np.minimum(sc[:, ia], sc[:, ib]), sc[:, ic])
            signals[f"{joint}_angle_{side}"], confidence[f"{joint}_angle_{side}"] = ang, conf
            per_side[side] = (ang, conf)
        (left, cl), (right, cr) = per_side["left"], per_side["right"]
        signals[f"{joint}_angle_mean"], confidence[f"{joint}_angle_mean"] = \
            fuse_sides(left, cl, right, cr, cfg.side_dominance)

    axis = kp[:, NECK] - kp[:, HIP_CENTRE]
    torso_len = np.linalg.norm(axis, axis=-1)
    torso_length_px = float(np.nanmedian(torso_len)) if np.isfinite(torso_len).any() else float("nan")
    # 0 deg = upright; sign follows image +x. Facing direction is unknown until M5.
    signals["torso_angle"] = np.degrees(np.arctan2(axis[:, 0], -axis[:, 1]))
    confidence["torso_angle"] = np.minimum(sc[:, NECK], sc[:, HIP_CENTRE])

    lw, rw = INDEX["left_wrist"], INDEX["right_wrist"]
    signals["wrist_separation"] = np.linalg.norm(kp[:, lw] - kp[:, rw], axis=-1) / torso_length_px
    confidence["wrist_separation"] = np.minimum(sc[:, lw], sc[:, rw])

    return FeatureSet(track.fps, track.n_frames, signals, confidence, track.present.copy(),
                      torso_length_px, cfg)
