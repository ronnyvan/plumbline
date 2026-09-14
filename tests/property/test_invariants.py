"""Property tests named in spec §11.7.

- Reversing a clip swaps concentric and eccentric durations.
- Scaling the frame leaves normalised features and angles unchanged.
- Degrading keypoint confidence never raises Q.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.features.quality import window_quality
from core.features.signals import extract_features
from core.pose.types import SubjectTrack
from core.registry import default_registry
from core.segment.reps import segment_reps
from eval.harness.synthetic import SynthParams, generate

REG = default_registry()
EXERCISES = st.sampled_from(["lat_pulldown", "preacher_curl", "leg_extension", "tricep_pushdown"])


def _seg(track: SubjectTrack, exercise: str):
    ex = REG.exercises[exercise]
    f = extract_features(track)
    return f, segment_reps(f, ex.rep_signal, ex.rep_phase.concentric)


@settings(max_examples=20, deadline=None)
@given(exercise=EXERCISES, seed=st.integers(0, 10_000),
       conc=st.floats(0.8, 1.6), ecc=st.floats(1.8, 3.0))
def test_reversing_a_clip_swaps_concentric_and_eccentric(exercise, seed, conc, ecc):
    clip = generate(SynthParams(exercise_id=exercise, n_reps=5, seed=seed, concentric_s=conc,
                                eccentric_s=ecc, tempo_jitter=0.0))
    _, fwd = _seg(clip.track, exercise)
    _, rev = _seg(clip.track.reversed(), exercise)
    assert len(fwd.reps) == len(rev.reps) == 5
    fwd_c = np.mean([r.concentric_s for r in fwd.reps])
    fwd_e = np.mean([r.eccentric_s for r in fwd.reps])
    rev_c = np.mean([r.concentric_s for r in rev.reps])
    rev_e = np.mean([r.eccentric_s for r in rev.reps])
    assert rev_c == pytest.approx(fwd_e, abs=0.2)
    assert rev_e == pytest.approx(fwd_c, abs=0.2)


@settings(max_examples=20, deadline=None)
@given(exercise=EXERCISES, seed=st.integers(0, 10_000), factor=st.floats(0.3, 3.0))
def test_scaling_the_frame_leaves_angles_and_normalised_features_unchanged(exercise, seed, factor):
    clip = generate(SynthParams(exercise_id=exercise, n_reps=3, seed=seed))
    base, _ = _seg(clip.track, exercise)
    scaled, _ = _seg(clip.track.scaled(factor), exercise)
    for name in ("elbow_angle_mean", "knee_angle_mean", "torso_angle", "wrist_separation"):
        np.testing.assert_allclose(scaled.signals[name], base.signals[name], rtol=1e-4, atol=1e-3)
    assert scaled.torso_length_px == pytest.approx(base.torso_length_px * factor, rel=1e-4)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), a=st.floats(0.0, 1.0), b=st.floats(0.0, 1.0))
def test_lowering_keypoint_confidence_never_raises_quality(seed, a, b):
    hi, lo = max(a, b), min(a, b)
    clip = generate(SynthParams(n_reps=3, seed=seed))
    t = clip.track

    def q(scale):
        track = SubjectTrack(t.fps, t.width, t.height, t.kp, t.kp_score * scale, t.present)
        f = extract_features(track)
        return window_quality(f.signals["torso_angle"], f.confidence["torso_angle"], f.present,
                              0, f.n_frames - 1).q
    assert q(lo) <= q(hi) + 1e-12
