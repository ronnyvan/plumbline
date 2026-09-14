"""Tracking and subject selection (spec §5.2).

Gyms contain bystanders. Picking the wrong person is a silent, catastrophic failure,
so the choice is an explicit scored decision with an abstention path — never
`max(area)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.keypoints import N_KEYPOINTS
from core.pose.types import PoseFrames, SubjectTrack
from core.staging import StageSpec

TRACK_VERSION = "2"   # v2: fragment stitching, co-existence ambiguity, coverage gate (F-03)


@dataclass(frozen=True)
class TrackConfig:
    iou_threshold: float = 0.3
    max_gap_frames: int = 8
    # A lifter on a machine barely moves, so fragments that never co-exist and line up across
    # a detection gap are one person, not two (F-03).
    stitch_gap_frames: int = 45
    ambiguity_min_overlap: float = 0.2   # tracks must share this fraction of the shorter one's frames
    min_coverage: float = 0.6            # provisional: detected frames / frames spanned by the track
    min_persistence: float = 0.10
    keypoint_floor: float = 0.3      # provisional: scores are not probabilities (Q-08)
    # §5.2 weights. anchor_proximity stays 0 until equipment detection exists (M5).
    w_area: float = 0.25
    w_centrality: float = 0.20
    w_motion: float = 0.25
    w_persistence: float = 0.30
    w_anchor_proximity: float = 0.0
    margin: float = 0.15

    def stage_spec(self) -> StageSpec:
        return StageSpec("track", TRACK_VERSION, self.__dict__.copy())


@dataclass(frozen=True)
class SubjectSelection:
    track: SubjectTrack | None
    ambiguous: bool
    reason: str | None
    scores: list[dict] = field(default_factory=list)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build_tracks(pf: PoseFrames, cfg: TrackConfig) -> list[list[tuple[int, int]]]:
    """Greedy IoU association. Each track is a list of (frame, row) pairs."""
    tracks: list[list[tuple[int, int]]] = []
    last_box: list[np.ndarray] = []
    for i in range(pf.n_frames):
        rows = pf.rows_for_frame(i)
        if len(rows) == 0:
            continue
        live = [t for t in range(len(tracks)) if i - tracks[t][-1][0] <= cfg.max_gap_frames]
        pairs = sorted(((_iou(pf.bbox[r], last_box[t]), t, r) for t in live for r in rows),
                       reverse=True)
        used_t, used_r = set(), set()
        for iou, t, r in pairs:
            if iou < cfg.iou_threshold or t in used_t or r in used_r:
                continue
            tracks[t].append((i, int(r)))
            last_box[t] = pf.bbox[r]
            used_t.add(t)
            used_r.add(r)
        for r in rows:
            if r not in used_r:
                tracks.append([(i, int(r))])
                last_box.append(pf.bbox[r])
    return _stitch(pf, tracks, cfg)


def _stitch(pf: PoseFrames, tracks: list[list[tuple[int, int]]], cfg: TrackConfig
            ) -> list[list[tuple[int, int]]]:
    tracks = sorted(tracks, key=lambda t: t[0][0])
    merged = True
    while merged:
        merged = False
        best = None
        for a in range(len(tracks)):
            end_a = tracks[a][-1]
            for b in range(len(tracks)):
                if a == b:
                    continue
                start_b = tracks[b][0]
                gap = start_b[0] - end_a[0]
                if not 0 < gap <= cfg.stitch_gap_frames:
                    continue
                iou = _iou(pf.bbox[end_a[1]], pf.bbox[start_b[1]])
                if iou >= cfg.iou_threshold and (best is None or gap < best[0]):
                    best = (gap, a, b)
        if best:
            _, a, b = best
            tracks[a] = tracks[a] + tracks[b]
            del tracks[b]
            merged = True
    return tracks


def _overlap(t1: list[tuple[int, int]], t2: list[tuple[int, int]]) -> float:
    f1, f2 = {f for f, _ in t1}, {f for f, _ in t2}
    return len(f1 & f2) / max(1, min(len(f1), len(f2)))


def _terms(pf: PoseFrames, track: list[tuple[int, int]], cfg: TrackConfig) -> dict:
    rows = np.array([r for _, r in track])
    frames = np.array([f for f, _ in track])
    boxes = pf.bbox[rows]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    area = float(np.median(w * h) / (pf.width * pf.height))
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    half_diag = 0.5 * np.hypot(pf.width, pf.height)
    centrality = float(1.0 - np.median(np.hypot(cx - pf.width / 2, cy - pf.height / 2)) / half_diag)
    motion = 0.0
    if len(rows) > 1:
        consecutive = np.flatnonzero(np.diff(frames) == 1)
        if len(consecutive):
            a, b = rows[consecutive], rows[consecutive + 1]
            ok = (pf.kp_score[a] > cfg.keypoint_floor) & (pf.kp_score[b] > cfg.keypoint_floor)
            disp = np.linalg.norm(pf.kp[b] - pf.kp[a], axis=-1)
            per_step = np.where(ok.any(axis=1),
                                np.nansum(np.where(ok, disp, np.nan), axis=1) / np.maximum(ok.sum(1), 1),
                                np.nan) / np.maximum(h[consecutive], 1.0)
            motion = float(np.nanmedian(per_step)) if np.isfinite(per_step).any() else 0.0
    return {"area": area, "centrality": centrality, "motion": motion,
            "persistence": len(set(frames.tolist())) / pf.n_frames, "anchor_proximity": 0.0}


def select_subject(pf: PoseFrames, cfg: TrackConfig = TrackConfig()) -> SubjectSelection:
    tracks = build_tracks(pf, cfg)
    candidates = [(t, _terms(pf, t, cfg)) for t in tracks]
    candidates = [(t, terms) for t, terms in candidates if terms["persistence"] >= cfg.min_persistence]
    if not candidates:
        return SubjectSelection(None, False, "no_subject_detected")

    keys = ("area", "centrality", "motion", "persistence", "anchor_proximity")
    weights = {"area": cfg.w_area, "centrality": cfg.w_centrality, "motion": cfg.w_motion,
               "persistence": cfg.w_persistence, "anchor_proximity": cfg.w_anchor_proximity}
    maxima = {k: max(terms[k] for _, terms in candidates) for k in keys}
    scored = []
    for t, terms in candidates:
        norm = {k: (terms[k] / maxima[k] if maxima[k] > 0 else 0.0) for k in keys}
        scored.append((sum(weights[k] * norm[k] for k in keys), t, terms))
    scored.sort(key=lambda s: s[0], reverse=True)
    report = [{"score": round(s, 4), **{k: round(v, 4) for k, v in terms.items()},
               "detections": len(t)} for s, t, terms in scored]

    # Only people present at the same time can be confused for each other.
    rivals = [s for s in scored[1:] if _overlap(scored[0][1], s[1]) >= cfg.ambiguity_min_overlap]
    if rivals and scored[0][0] - rivals[0][0] < cfg.margin:
        return SubjectSelection(None, True, "multiple_subjects_ambiguous", report)

    best = scored[0][1]
    frames = sorted({f for f, _ in best})
    coverage = len(frames) / (frames[-1] - frames[0] + 1)
    if coverage < cfg.min_coverage:
        return SubjectSelection(None, False, "subject_intermittently_detected",
                                [{**report[0], "coverage": round(coverage, 4)}, *report[1:]])
    kp = np.full((pf.n_frames, N_KEYPOINTS, 2), np.nan, np.float32)
    sc = np.zeros((pf.n_frames, N_KEYPOINTS), np.float32)
    present = np.zeros(pf.n_frames, bool)
    for f, r in best:
        kp[f], sc[f], present[f] = pf.kp[r], pf.kp_score[r], True
    return SubjectSelection(SubjectTrack(pf.fps, pf.width, pf.height, kp, sc, present),
                            False, None, report)
