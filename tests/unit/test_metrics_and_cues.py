"""Tier B metrics (§6.3), LP-01 (§11.7, D-20) and the engine's gating (§7.1, D-18, D-19)."""
import dataclasses

import numpy as np
import pytest

from core.cues import lp01_torso_swing
from core.cues.agnostic import compute
from core.cues.contract import CueResult, Detection
from core.cues.engine import run_cues
from core.registry import default_registry
from eval.harness.synthetic import SynthParams
from tests._util import run, segment

REG = default_registry()
LP01 = REG.cues["LP-01"].params


def test_fatigue_slope_recovers_known_rom_loss():
    clip, f, seg = segment(SynthParams(exercise_id="preacher_curl", n_reps=10, rom_decay_per_rep=0.04, seed=31))
    out = compute(f, seg, station_used="S-LAT", station_verified=False)
    true_rom = [r.rom_deg for r in clip.truth.reps]
    true_slope = np.polyfit(np.arange(len(true_rom)), true_rom, 1)[0]
    assert out["fatigue_slope_rom"]["value"] == pytest.approx(true_slope, abs=0.8)
    assert out["fatigue_slope_rom_rel"]["value"] < 0


def test_tut_matches_ground_truth():
    clip, f, seg = segment(SynthParams(n_reps=8, seed=32))
    out = compute(f, seg, station_used="S-LAT", station_verified=False)
    truth = sum(r.concentric_s + r.eccentric_s for r in clip.truth.reps)
    assert out["tut"]["value"] == pytest.approx(truth, rel=0.05)


def test_clean_set_has_small_rom_variation():
    _, f, seg = segment(SynthParams(n_reps=8, seed=33))
    assert compute(f, seg, station_used="S-LAT", station_verified=False)["rom_cv"]["value"] < 0.03


def test_asymmetry_is_unavailable_from_the_side_with_a_reason():
    _, f, seg = segment(SynthParams(n_reps=6, seed=34))
    out = compute(f, seg, station_used="S-LAT", station_verified=True)["asymmetry_rom"]
    assert out["available"] is False
    assert out["reason"] == "requires_frontal_or_rear_station"


def test_noisy_clip_withholds_timing_but_keeps_counts():
    clip, report, _ = run(SynthParams(exercise_id="lat_pulldown", noise_px=25, seed=35))
    summary = report["set_summary"]
    assert summary["tempo"]["reason"] == "too_noisy_for_timing"
    assert summary["tut"]["available"] is False
    assert summary["reps"]["value"] == len(clip.truth.reps)
    assert summary["rom_per_rep"]["available"] is True


# --- LP-01, the §11.7 cases ---------------------------------------------------------------

def _lp01(**kw):
    _, f, seg = segment(SynthParams(n_reps=9, seed=11, **kw))
    return lp01_torso_swing.detect(f, seg, LP01), seg


def test_lp01_clean_set_does_not_fire():
    det, _ = _lp01()
    assert not det.fired
    assert det.raw_score < 8.0


def test_lp01_fires_on_22_degree_swing_on_reps_7_to_9():
    det, seg = _lp01(swing_deg=20.0, swing_reps=(6, 7, 8))
    assert det.fired
    assert det.raw_score == pytest.approx(22.0, abs=2.0)
    assert det.severity == pytest.approx((22.0 - 15.0) / 15.0, abs=0.12)
    first, last = seg.reps[6].start, seg.reps[8].end
    assert all(first <= f <= last for f in det.evidence.frame_indices)


def test_lp01_ignores_a_single_outlier_rep():
    # D-20: the k-th largest amplitude needs a cluster of swinging reps.
    det, _ = _lp01(swing_deg=20.0, swing_reps=(4,))
    assert not det.fired


def test_lp01_median_statistic_would_have_missed_late_set_swing():
    # Documents why D-20 replaced the §6.2 median: the spec's own §11.7 case defeats it.
    det, _ = _lp01(swing_deg=20.0, swing_reps=(6, 7, 8))
    assert np.nanmedian(det.per_rep) < LP01["threshold_deg"] < det.raw_score


# --- engine gating ------------------------------------------------------------------------

def test_detections_cannot_carry_confidence():
    # D-19: the type a detector returns has no field for confidence, observability or quality.
    names = {f.name for f in dataclasses.fields(Detection)}
    assert not names & {"confidence", "observability", "quality"}
    assert {"confidence", "observability", "quality"} <= {f.name for f in dataclasses.fields(CueResult)}


def test_unverified_station_abstains_with_honest_remedy():
    _, f, seg = segment(SynthParams(n_reps=9, seed=11, swing_deg=20.0, swing_reps=(6, 7, 8)))
    out = run_cues("lat_pulldown", f, seg, REG, station_used="S-LAT", station_verified=False)
    assert out.results == []
    [ab] = out.abstentions
    assert (ab.cue_id, ab.gate, ab.reason) == ("LP-01", "observability", "station_unverified")
    assert ab.remedy["action"] == "none"


def test_wrong_station_abstains_with_reposition_remedy():
    _, f, seg = segment(SynthParams(n_reps=6, seed=12))
    out = run_cues("lat_pulldown", f, seg, REG, station_used="S-FRONT", station_verified=True)
    [ab] = out.abstentions
    assert ab.reason == "requires_station"
    assert ab.remedy["target_station"] == "S-LAT"
    assert "to your side" in ab.remedy["copy"]


def test_provisional_detector_is_computed_but_never_displayable():
    _, f, seg = segment(SynthParams(n_reps=9, seed=11, swing_deg=20.0, swing_reps=(6, 7, 8)))
    out = run_cues("lat_pulldown", f, seg, REG, station_used="S-LAT", station_verified=True)
    assert len(out.results) == 1 and out.results[0].fired
    assert out.results[0].confidence is None
    assert out.displayable == []
    assert {n["cue_id"]: n["reason"] for n in out.not_assessed}["LP-01"] == "not_validated"


def test_planned_cues_are_listed_as_not_built():
    _, f, seg = segment(SynthParams(exercise_id="preacher_curl", n_reps=5, seed=13))
    out = run_cues("preacher_curl", f, seg, REG, station_used="S-LAT", station_verified=True)
    assert {n["cue_id"] for n in out.not_assessed} == set(REG.exercises["preacher_curl"].cues)
