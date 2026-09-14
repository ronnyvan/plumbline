"""The single analysis entry point (spec §4.1, D-11).

`analyze_pose` is what both production and the eval harness call. `analyze_video` only
adds ingest and pose estimation in front of it, with content-addressed caching (§4.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.cues.agnostic import AGNOSTIC_VERSION
from core.cues.agnostic import compute as compute_agnostic
from core.cues.engine import DETECTORS, run_cues
from core.features.quality import sharpness_from_laplacian_var
from core.features.signals import FeatureConfig, extract_features
from core.ingest.video import IngestConfig, IngestResult, ingest, iter_frames
from core.pose.track import TrackConfig, select_subject
from core.pose.types import PoseFrames
from core.registry import Registry, default_registry
from core.report import build as build_report
from core.segment.reps import SegmentConfig, segment_reps
from core.staging import ArtifactStore, StageSpec, manifest

CUES_VERSION = "1"


@dataclass(frozen=True)
class PipelineConfig:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)


def analysis_stages(cfg: PipelineConfig, registry: Registry) -> list[StageSpec]:
    """Stages after pose estimation — shared by every input source."""
    return [
        cfg.track.stage_spec(),
        cfg.features.stage_spec(),
        cfg.segment.stage_spec(),
        StageSpec("agnostic", AGNOSTIC_VERSION),
        StageSpec("cues", CUES_VERSION, {"registry": registry.content_hash,
                                         "detectors": sorted(DETECTORS)}),
    ]


def analyze_pose(pose: PoseFrames, exercise_id: str, *, station_requested: str,
                 station_verified: bool = False, registry: Registry | None = None,
                 cfg: PipelineConfig = PipelineConfig(), sharpness: np.ndarray | None = None,
                 upstream: list[StageSpec] | None = None) -> dict:
    registry = registry or default_registry()
    if exercise_id not in registry.exercises:
        raise KeyError(f"exercise {exercise_id!r} is not in the registry (Tier C)")
    ex = registry.exercises[exercise_id]
    man = manifest([*(upstream or []), *analysis_stages(cfg, registry)])
    station = {"requested": station_requested, "verified": station_verified,
               "verification": "unavailable until equipment anchoring (M5)"}

    selection = select_subject(pose, cfg.track)
    sel = {"reason": selection.reason, "ambiguous": selection.ambiguous, "candidates": selection.scores}
    if selection.track is None:
        return build_report(registry=registry, exercise_id=exercise_id, manifest_hash=man["manifest_hash"],
                            set_summary=None, selection=sel, station=station,
                            abstained_reason=selection.reason)

    features = extract_features(selection.track, cfg.features)
    seg = segment_reps(features, ex.rep_signal, ex.rep_phase.concentric, cfg.segment)
    summary = compute_agnostic(features, seg, station_used=station_requested,
                               station_verified=station_verified)
    if not seg.reps:
        return build_report(registry=registry, exercise_id=exercise_id, manifest_hash=man["manifest_hash"],
                            set_summary=summary, segmentation=seg, selection=sel, station=station,
                            abstained_reason=seg.reason or "no_motion")

    engine = run_cues(exercise_id, features, seg, registry, station_used=station_requested,
                      station_verified=station_verified, sharpness=sharpness)
    return build_report(registry=registry, exercise_id=exercise_id, manifest_hash=man["manifest_hash"],
                        set_summary=summary, engine=engine, segmentation=seg, selection=sel,
                        station=station)


def ingest_cached(path: Path, store: ArtifactStore, cfg: PipelineConfig) -> tuple[str, IngestResult]:
    from core.staging import sha256_file
    source_hash = sha256_file(path)
    spec = cfg.ingest.stage_spec()
    cached = store.get_json(source_hash, spec)
    if cached is not None:
        return source_hash, IngestResult.from_dict(cached)
    result = ingest(path, cfg.ingest)
    store.put_json(source_hash, spec, result.as_dict())
    return source_hash, result


def pose_cached(path: Path, source_hash: str, ing: IngestResult, backend, store: ArtifactStore,
                cfg: PipelineConfig) -> PoseFrames:
    spec = backend.stage_spec()
    target = store.path(source_hash, StageSpec(spec.name, spec.version,
                                               {**spec.config, "ingest": cfg.ingest.stage_spec().config_hash}),
                        "parquet")
    if target.exists():
        return PoseFrames.from_parquet(target)
    lo, hi = ing.active
    pose = backend.infer(iter_frames(path, ing, (lo, hi)), n_frames=ing.n_frames,
                         width=ing.width, height=ing.height, fps=ing.target_fps)
    pose.to_parquet(target)
    return pose


def analyze_video(path: Path, exercise_id: str, *, station_requested: str, store: ArtifactStore,
                  backend=None, registry: Registry | None = None,
                  cfg: PipelineConfig = PipelineConfig()) -> dict:
    registry = registry or default_registry()
    if backend is None:
        from core.pose.rtmpose import RTMPoseBackend
        backend = RTMPoseBackend()
    source_hash, ing = ingest_cached(Path(path), store, cfg)
    pose = pose_cached(Path(path), source_hash, ing, backend, store, cfg)
    report = analyze_pose(pose, exercise_id, station_requested=station_requested,
                          station_verified=False, registry=registry, cfg=cfg,
                          sharpness=sharpness_from_laplacian_var(np.asarray(ing.sharpness_raw)),
                          upstream=[cfg.ingest.stage_spec(), backend.stage_spec()])
    report["diagnostics"]["ingest"] = {
        "source_sha256": source_hash, "native_fps": ing.native_fps, "duration_s": ing.duration_s,
        "frames": ing.n_frames, "active": list(ing.active), "size": [ing.width, ing.height],
        "pose_inference_s": pose.meta.get("inference_s"), "provider": pose.meta.get("provider"),
    }
    return report
