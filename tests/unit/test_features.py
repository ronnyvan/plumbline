"""Signal conditioning (§5.6)."""
import numpy as np
import pytest

from core.features.quality import window_quality
from core.features.signals import FeatureConfig, extract_features, fill_gaps, fuse_sides, joint_angle, smooth
from eval.harness.synthetic import SynthParams, generate


def test_joint_angle_right_angle_and_straight():
    a, b = np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]])
    assert joint_angle(a, b, np.array([[0.0, 1.0]]))[0] == pytest.approx(90.0)
    assert joint_angle(a, b, np.array([[-1.0, 0.0]]))[0] == pytest.approx(180.0)


def test_fill_gaps_fills_short_interior_gaps_only():
    x = np.array([0, 1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, np.nan, 10.0])
    out = fill_gaps(x, max_gap=4)
    assert out[2:4] == pytest.approx([2.0, 3.0])
    assert np.isnan(out[5:10]).all()          # five-frame gap is never invented


def test_fill_gaps_never_extrapolates_edges():
    out = fill_gaps(np.array([np.nan, np.nan, 2.0, 3.0]), max_gap=4)
    assert np.isnan(out[:2]).all()


def test_savgol_preserves_lines_and_has_no_lag():
    x = np.linspace(0, 50, 100)
    assert smooth(x, 9, 2) == pytest.approx(x)


def test_fuse_sides_drops_the_much_less_confident_side():
    left, right = np.array([100.0]), np.array([160.0])
    fused, _ = fuse_sides(left, np.array([0.9]), right, np.array([0.3]), dominance=1.5)
    assert fused[0] == pytest.approx(100.0)
    fused, _ = fuse_sides(left, np.array([0.8]), right, np.array([0.8]), dominance=1.5)
    assert fused[0] == pytest.approx(130.0)


def test_extracted_elbow_angle_matches_commanded_kinematics():
    clip = generate(SynthParams(noise_px=0.0, n_reps=3, seed=1))
    f = extract_features(clip.track, FeatureConfig())
    elbow = f.signals["elbow_angle_mean"]
    assert np.nanmax(elbow) == pytest.approx(165, abs=1.5)
    assert np.nanmin(elbow) == pytest.approx(75, abs=1.5)


def test_quality_falls_with_visibility_and_confidence():
    signal = np.ones(100)
    conf = np.full(100, 0.9)
    present = np.ones(100, bool)
    full = window_quality(signal, conf, present, 0, 99).q
    holed = signal.copy()
    holed[:50] = np.nan
    assert window_quality(holed, conf, present, 0, 99).q < full
    assert window_quality(signal, conf * 0.5, present, 0, 99).q < full
