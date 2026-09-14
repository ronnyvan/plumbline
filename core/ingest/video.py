"""Video ingest (spec §5.1, D-15).

Phone video is usually variable frame rate. Output frames are chosen by timestamp —
nearest source frame to each k/30 s — rather than by index, so tempo stays true to wall
time. Decoded frames are not written to disk (D-15): the ingest artifact stores which
source frame backs each output frame, and later stages re-decode deterministically.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from core.staging import StageSpec, sha256_file

INGEST_VERSION = "1"
MAX_DURATION_S = 90.0
MIN_DURATION_S = 2.0
MAX_BYTES = 4 * 1024 ** 3


class IngestRejected(Exception):
    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


@dataclass(frozen=True)
class IngestConfig:
    target_fps: float = 30.0
    max_side: int = 1080
    stats_side: int = 360
    activity_pad_s: float = 1.0
    activity_frac: float = 0.15
    phash_samples: int = 8

    def stage_spec(self) -> StageSpec:
        return StageSpec("ingest", INGEST_VERSION, asdict(self))


@dataclass(frozen=True)
class IngestResult:
    source_sha256: str
    size_bytes: int
    native_fps: float
    source_frames: int
    duration_s: float
    width: int                  # output frame size, after rotation and scaling
    height: int
    scale: float
    target_fps: float
    source_index: list[int]     # source frame backing each output frame
    timestamps_s: list[float]
    sharpness_raw: list[float]  # variance of Laplacian per output frame
    motion: list[float]
    active: tuple[int, int]     # output-frame window containing the set, padded
    phash: str
    orientation_meta: float | None

    @property
    def n_frames(self) -> int:
        return len(self.source_index)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> IngestResult:
        return cls(**{**d, "active": tuple(d["active"])})


def _stats_gray(frame: np.ndarray, side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _dhash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def ingest(path: Path, cfg: IngestConfig = IngestConfig()) -> IngestResult:
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise IngestRejected("file_too_large", {"size_bytes": size, "max_bytes": MAX_BYTES})
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IngestRejected("unreadable_video")
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if native_fps <= 0:
            raise IngestRejected("unknown_frame_rate")
        if declared and declared / native_fps > MAX_DURATION_S * 1.05:
            raise IngestRejected("too_long", {"duration_s": round(declared / native_fps, 2),
                                              "max_s": MAX_DURATION_S})
        orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META) \
            if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else None

        source_index, timestamps, sharp, motion, samples = [], [], [], [], []
        prev = None           # (index, ts, frame)
        prev_gray = None
        width = height = 0
        k, i, t0, last_ts = 0, 0, None, -1.0

        def emit(idx: int, ts: float, frame: np.ndarray) -> None:
            nonlocal prev_gray
            gray = _stats_gray(frame, cfg.stats_side)
            source_index.append(idx)
            timestamps.append(round(ts - t0, 6))
            sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            motion.append(float(np.mean(cv2.absdiff(gray, prev_gray))) if prev_gray is not None else 0.0)
            prev_gray = gray
            samples.append(gray)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= last_ts:                      # broken or missing timestamps
                ts = i / native_fps
            last_ts = ts
            if t0 is None:
                t0 = ts
                height, width = frame.shape[:2]
            while t0 + k / cfg.target_fps <= ts:
                target = t0 + k / cfg.target_fps
                if prev is not None and abs(prev[1] - target) <= abs(ts - target):
                    emit(prev[0], prev[1], prev[2])
                else:
                    emit(i, ts, frame)
                k += 1
            prev = (i, ts, frame)
            i += 1
        # One trailing target may sit just past the last frame's timestamp.
        if prev is not None and t0 + k / cfg.target_fps - prev[1] <= 0.5 / native_fps:
            emit(prev[0], prev[1], prev[2])
    finally:
        cap.release()

    if i == 0:
        raise IngestRejected("no_frames")
    duration = (last_ts - t0) + 1.0 / native_fps
    if duration > MAX_DURATION_S:
        raise IngestRejected("too_long", {"duration_s": round(duration, 2), "max_s": MAX_DURATION_S})
    if duration < MIN_DURATION_S:
        raise IngestRejected("too_short", {"duration_s": round(duration, 2), "min_s": MIN_DURATION_S})

    scale = min(1.0, cfg.max_side / max(width, height))
    m = np.convolve(np.asarray(motion), np.ones(5) / 5, mode="same")
    moving = np.flatnonzero(m > cfg.activity_frac * np.percentile(m, 95)) if m.max() > 0 else []
    pad = int(cfg.activity_pad_s * cfg.target_fps)
    n = len(source_index)
    active = (max(0, int(moving[0]) - pad), min(n - 1, int(moving[-1]) + pad)) if len(moving) else (0, n - 1)
    picks = np.linspace(0, n - 1, min(cfg.phash_samples, n)).astype(int)

    return IngestResult(
        source_sha256=sha256_file(path), size_bytes=size, native_fps=round(native_fps, 4),
        source_frames=i, duration_s=round(duration, 4),
        width=int(round(width * scale)), height=int(round(height * scale)), scale=scale,
        target_fps=cfg.target_fps, source_index=source_index, timestamps_s=timestamps,
        sharpness_raw=[round(s, 3) for s in sharp], motion=[round(x, 4) for x in motion],
        active=active, phash="".join(_dhash(samples[p]) for p in picks),
        orientation_meta=orientation,
    )


def iter_frames(path: Path, ing: IngestResult, window: tuple[int, int] | None = None
                ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (output_index, BGR frame) for output frames in `window`, resized per ingest."""
    lo, hi = window or (0, ing.n_frames - 1)
    wanted = ing.source_index[lo:hi + 1]
    cap = cv2.VideoCapture(str(path))
    try:
        src, k = -1, 0
        frame = None
        while k < len(wanted):
            while src < wanted[k]:
                ok, frame = cap.read()
                if not ok:
                    return
                src += 1
            out = frame if ing.scale == 1.0 else cv2.resize(
                frame, (ing.width, ing.height), interpolation=cv2.INTER_AREA)
            yield lo + k, out
            k += 1
    finally:
        cap.release()
