"""Pose data containers.

`PoseFrames` holds every detection in every frame, exactly as the backend emitted it.
`SubjectTrack` is the dense per-frame skeleton of the one person selected as the lifter.
Keeping the raw detections means subject selection can be re-run without re-running
the GPU (§4.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.keypoints import N_KEYPOINTS


@dataclass(frozen=True)
class PoseFrames:
    fps: float
    width: int
    height: int
    n_frames: int
    frame: np.ndarray        # (N,) int32 — output-frame index of each detection
    bbox: np.ndarray         # (N, 4) float32 — x1, y1, x2, y2
    kp: np.ndarray           # (N, 26, 2) float32 — image pixels
    kp_score: np.ndarray     # (N, 26) float32 — raw SimCC response, NOT a probability
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.frame)
        if self.bbox.shape != (n, 4) or self.kp.shape != (n, N_KEYPOINTS, 2) \
                or self.kp_score.shape != (n, N_KEYPOINTS):
            raise ValueError("PoseFrames arrays have inconsistent shapes")

    def rows_for_frame(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.frame == i)

    def to_parquet(self, path: Path) -> None:
        cols = {
            "frame": pa.array(self.frame.astype(np.int32)),
            **{f"b{j}": pa.array(self.bbox[:, j].astype(np.float32)) for j in range(4)},
        }
        for k in range(N_KEYPOINTS):
            cols[f"x{k}"] = pa.array(self.kp[:, k, 0].astype(np.float32))
            cols[f"y{k}"] = pa.array(self.kp[:, k, 1].astype(np.float32))
            cols[f"s{k}"] = pa.array(self.kp_score[:, k].astype(np.float32))
        table = pa.table(cols)
        import json
        table = table.replace_schema_metadata({
            "plumbline": json.dumps({"fps": self.fps, "width": self.width, "height": self.height,
                                     "n_frames": self.n_frames, "meta": self.meta}, default=str)})
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

    @classmethod
    def from_parquet(cls, path: Path) -> PoseFrames:
        import json
        table = pq.read_table(path)
        info = json.loads(table.schema.metadata[b"plumbline"])
        d = table.to_pydict()
        n = len(d["frame"])
        kp = np.zeros((n, N_KEYPOINTS, 2), np.float32)
        sc = np.zeros((n, N_KEYPOINTS), np.float32)
        for k in range(N_KEYPOINTS):
            kp[:, k, 0], kp[:, k, 1], sc[:, k] = d[f"x{k}"], d[f"y{k}"], d[f"s{k}"]
        return cls(fps=info["fps"], width=info["width"], height=info["height"],
                   n_frames=info["n_frames"], frame=np.asarray(d["frame"], np.int32),
                   bbox=np.stack([d[f"b{j}"] for j in range(4)], axis=1).astype(np.float32)
                   if n else np.zeros((0, 4), np.float32),
                   kp=kp, kp_score=sc, meta=info["meta"])


@dataclass(frozen=True)
class SubjectTrack:
    """Dense skeleton of the selected subject. Absent frames are NaN with score 0."""
    fps: float
    width: int
    height: int
    kp: np.ndarray           # (T, 26, 2)
    kp_score: np.ndarray     # (T, 26)
    present: np.ndarray      # (T,) bool

    @property
    def n_frames(self) -> int:
        return len(self.present)

    def reversed(self) -> SubjectTrack:
        return SubjectTrack(self.fps, self.width, self.height,
                            self.kp[::-1].copy(), self.kp_score[::-1].copy(), self.present[::-1].copy())

    def scaled(self, factor: float) -> SubjectTrack:
        return SubjectTrack(self.fps, int(round(self.width * factor)), int(round(self.height * factor)),
                            self.kp * factor, self.kp_score.copy(), self.present.copy())
