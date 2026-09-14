"""Shared test helpers: run the production pipeline on a synthetic clip and score it."""
from __future__ import annotations

from core.features.signals import extract_features
from core.pipeline import analyze_pose
from core.registry import default_registry
from core.segment.reps import segment_reps
from eval.harness.corpus import synthetic_truth_reps
from eval.harness.synthetic import SynthParams, generate
from eval.metrics.segmentation import item_metrics

EXERCISES = ("lat_pulldown", "preacher_curl", "leg_extension", "leg_press_quad_bias", "tricep_pushdown")


def segment(params: SynthParams):
    clip = generate(params)
    ex = default_registry().exercises[params.exercise_id]
    features = extract_features(clip.track)
    return clip, features, segment_reps(features, ex.rep_signal, ex.rep_phase.concentric)


def run(params: SynthParams, **kwargs):
    clip = generate(params)
    report = analyze_pose(clip.pose, params.exercise_id, station_requested="S-LAT", **kwargs)
    seg = report["diagnostics"].get("segmentation")
    metrics = item_metrics(seg["reps"] if seg else [], synthetic_truth_reps(clip), clip.truth.fps)
    return clip, report, metrics


def max_boundary_ms(metrics: dict) -> float:
    values = sum(metrics["boundary_error_ms"].values(), [])
    return max(values) if values else 0.0
