"""Staging (§4.2), splits and leakage (§10.2), statistics (§10.3), pose provider guard (D-14)."""
from dataclasses import replace

import numpy as np
import pytest

from core.pose.rtmpose import PoseBackendError, verify_providers
from core.pose.types import PoseFrames
from core.staging import ArtifactStore, StageSpec, manifest
from eval.harness.splits import FRACTIONS, assign_split, check_leakage
from eval.harness.synthetic import SynthParams, generate
from eval.metrics.stats import cluster_bootstrap


def test_stage_key_depends_on_version_and_config(tmp_path):
    store = ArtifactStore(tmp_path)
    base = StageSpec("features", "1", {"window": 9})
    assert store.key("abc", base) == store.key("abc", StageSpec("features", "1", {"window": 9}))
    assert store.key("abc", base) != store.key("abc", replace(base, version="2"))
    assert store.key("abc", base) != store.key("abc", StageSpec("features", "1", {"window": 11}))


def test_manifest_hash_changes_with_any_stage():
    a = manifest([StageSpec("a", "1"), StageSpec("b", "1")])
    b = manifest([StageSpec("a", "1"), StageSpec("b", "2")])
    assert a["manifest_hash"] != b["manifest_hash"]


def test_withdrawal_deletes_every_derived_artifact(tmp_path):
    # §9.2: withdrawal must actually delete. This is the tested deletion path R-11 demands.
    store = ArtifactStore(tmp_path)
    for name in ("ingest", "pose", "segment"):
        store.put_json("deadbeef" * 8, StageSpec(name, "1"), {"x": 1})
    store.put_json("cafebabe" * 8, StageSpec("ingest", "1"), {"keep": True})
    assert store.delete_input("deadbeef" * 8) == 3
    assert store.get_json("deadbeef" * 8, StageSpec("pose", "1")) is None
    assert store.get_json("cafebabe" * 8, StageSpec("ingest", "1")) == {"keep": True}


def test_pose_frames_parquet_roundtrip(tmp_path):
    pose = generate(SynthParams(n_reps=2, seed=61, bystander=True)).pose
    path = tmp_path / "pose.parquet"
    pose.to_parquet(path)
    back = PoseFrames.from_parquet(path)
    assert back.n_frames == pose.n_frames and back.meta == pose.meta
    np.testing.assert_allclose(back.kp, pose.kp, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(back.frame, pose.frame)


def test_split_assignment_is_deterministic_and_proportional():
    subjects = [f"s{i}" for i in range(20_000)]
    splits = [assign_split(s) for s in subjects]
    assert splits == [assign_split(s) for s in subjects]
    for name, frac in FRACTIONS:
        assert splits.count(name) / len(splits) == pytest.approx(frac, abs=0.015)


def test_leakage_check_flags_a_subject_in_two_splits():
    rows = [("s1", "train"), ("s2", "test"), ("s1", "test")]
    assert check_leakage(rows) == [{"subject_id": "s1", "splits": ["test", "train"]}]


def test_cluster_bootstrap_interval_covers_truth_and_widens_with_fewer_subjects():
    rng = np.random.default_rng(0)
    def sample(n_subjects):
        vals, subs = [], []
        for s in range(n_subjects):
            offset = rng.normal(0, 1)          # subjects differ: clips are not independent
            for _ in range(5):
                vals.append(10 + offset + rng.normal(0, 0.5))
                subs.append(s)
        return cluster_bootstrap(vals, subs, n_resamples=2000, seed=1)
    many, few = sample(200), sample(10)
    assert many["ci95"][0] < 10 < many["ci95"][1]
    assert (few["ci95"][1] - few["ci95"][0]) > (many["ci95"][1] - many["ci95"][0])


def test_cluster_bootstrap_is_reproducible():
    vals, subs = list(range(30)), [i % 6 for i in range(30)]
    assert cluster_bootstrap(vals, subs, seed=3) == cluster_bootstrap(vals, subs, seed=3)


def test_cuda_request_landing_on_cpu_is_an_error():
    # D-14: ONNX Runtime silently falls back to CPU when CUDA libraries are missing.
    with pytest.raises(PoseBackendError, match="running on CPUExecutionProvider"):
        verify_providers("cuda", ["CPUExecutionProvider"], allow_cpu_fallback=False)
    assert verify_providers("cuda", ["CPUExecutionProvider"], allow_cpu_fallback=True) == "CPUExecutionProvider"
    assert verify_providers("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"], False) == "CUDAExecutionProvider"
