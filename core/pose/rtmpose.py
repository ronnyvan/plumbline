"""RTMPose-m (Halpe-26) backend via rtmlib + ONNX Runtime (D-04, D-13, D-14).

Two behaviours of the underlying library are deliberately overridden here:

1. Silent CPU fallback. ONNX Runtime drops to CPU when CUDA libraries are missing and
   only logs a warning. CPU and GPU produce slightly different numbers and very
   different latency, so a CUDA request that lands on CPU is a hard error unless the
   caller explicitly allows it. The provider actually used is stamped into metadata.

2. Hallucinated subjects. rtmlib's RTMPose, given zero boxes, runs on the whole image
   and returns a skeleton for nobody. Frames with no detection get no pose.
"""
from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np

from core.keypoints import N_KEYPOINTS
from core.pose.types import PoseFrames
from core.staging import StageSpec, sha256_file

BACKEND_VERSION = "1"


class PoseBackendError(RuntimeError):
    pass


def verify_providers(requested: str, actual: list[str], allow_cpu_fallback: bool) -> str:
    """Return the provider in use; raise if CUDA was requested but not granted."""
    in_use = actual[0] if actual else "none"
    if requested.startswith("cuda") and in_use != "CUDAExecutionProvider" and not allow_cpu_fallback:
        raise PoseBackendError(
            "CUDA was requested but ONNX Runtime is running on "
            f"{in_use}. Install the CUDA runtime with "
            "`pip install \"onnxruntime-gpu[cuda,cudnn]\"`, or pass allow_cpu_fallback=True "
            "to accept slower, numerically different CPU results.")
    return in_use


class RTMPoseBackend:
    MODE = "balanced"  # rtmlib BodyWithFeet: YOLOX-m detector + RTMPose-m Halpe-26 256x192

    def __init__(self, device: str = "cuda", allow_cpu_fallback: bool = False):
        try:
            import onnxruntime as ort
            from rtmlib import YOLOX, RTMPose
            from rtmlib.tools.solution.body_with_feet import BodyWithFeet
        except ImportError as e:  # pragma: no cover - exercised only without the pose extra
            raise PoseBackendError("pose dependencies missing; run `make install-pose`") from e

        if device.startswith("cuda") and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # loads CUDA/cuDNN from the nvidia-* wheels
        cfg = BodyWithFeet.MODE[self.MODE]
        self._det = YOLOX(cfg["det"], model_input_size=cfg["det_input_size"],
                          backend="onnxruntime", device=device)
        self._pose = RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"],
                             backend="onnxruntime", device=device)
        providers = {verify_providers(device, t.session.get_providers(), allow_cpu_fallback)
                     for t in (self._det, self._pose)}
        self.meta = {
            "backend_version": BACKEND_VERSION,
            "mode": self.MODE,
            "detector": cfg["det"].rsplit("/", 1)[-1],
            "detector_sha256": sha256_file(self._det.onnx_model),
            "pose_model": cfg["pose"].rsplit("/", 1)[-1],
            "pose_sha256": sha256_file(self._pose.onnx_model),
            "provider": sorted(providers),
            "onnxruntime": ort.__version__,
            "score_semantics": "SimCC heatmap response clipped to [0.001, 50]; not a probability",
        }

    def stage_spec(self) -> StageSpec:
        # Provider is part of the identity: CPU and GPU results are not interchangeable.
        return StageSpec("pose", BACKEND_VERSION, {k: self.meta[k] for k in
                         ("mode", "detector_sha256", "pose_sha256", "provider")})

    def infer(self, frames: Iterable[tuple[int, np.ndarray]], *, n_frames: int,
              width: int, height: int, fps: float) -> PoseFrames:
        rows_frame, rows_bbox, rows_kp, rows_sc = [], [], [], []
        t0 = time.perf_counter()
        for i, image in frames:
            boxes = np.asarray(self._det(image), dtype=np.float32).reshape(-1, 4)
            if len(boxes) == 0:
                continue  # never let rtmlib fall back to a whole-image "person"
            kp, sc = self._pose(image, bboxes=boxes.tolist())
            rows_frame.extend([i] * len(boxes))
            rows_bbox.append(boxes)
            rows_kp.append(np.asarray(kp, np.float32))
            rows_sc.append(np.asarray(sc, np.float32))
        elapsed = time.perf_counter() - t0
        n = len(rows_frame)
        return PoseFrames(
            fps=fps, width=width, height=height, n_frames=n_frames,
            frame=np.asarray(rows_frame, np.int32),
            bbox=np.concatenate(rows_bbox) if n else np.zeros((0, 4), np.float32),
            kp=np.concatenate(rows_kp) if n else np.zeros((0, N_KEYPOINTS, 2), np.float32),
            kp_score=np.concatenate(rows_sc) if n else np.zeros((0, N_KEYPOINTS), np.float32),
            meta={**self.meta, "inference_s": round(elapsed, 3)},
        )
