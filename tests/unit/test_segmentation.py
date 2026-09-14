"""Rep segmentation (§5.5), including the §11.7 cases and regression cases F-01..F-04."""
import pytest

from eval.harness.synthetic import SynthParams
from tests._util import EXERCISES, max_boundary_ms, run, segment


@pytest.mark.parametrize("exercise", EXERCISES)
def test_clean_eight_rep_set(exercise):
    clip, _, seg = segment(SynthParams(exercise_id=exercise, n_reps=8, seed=21))
    assert len(seg.reps) == 8
    errors = []
    for truth, pred in zip(clip.truth.reps, seg.reps, strict=True):
        assert pred.first_phase == truth.first_phase
        for b in ("start", "turn_arrive", "turn_leave", "end"):
            errors.append(abs(getattr(pred, b) - getattr(truth, b)))
    # Bounds come from the measured clean-clip distribution (worst 133 ms = 4 frames), not from
    # intuition: every boundary within 5 frames, and the average within 2.
    assert max(errors) <= 5
    assert sum(errors) / len(errors) <= 2.0


def test_leg_press_is_eccentric_first_and_phases_map_correctly():
    # D-16: sets start at lockout; the spec's concentric-first assumption would swap tempo.
    clip, _, seg = segment(SynthParams(exercise_id="leg_press_quad_bias", n_reps=6, seed=22))
    assert all(r.first_phase == "eccentric" for r in seg.reps)
    for truth, pred in zip(clip.truth.reps, seg.reps, strict=True):
        assert pred.concentric_s == pytest.approx(truth.concentric_s, abs=0.15)
        assert pred.eccentric_s == pytest.approx(truth.eccentric_s, abs=0.15)


def test_partial_first_rep_is_excluded_and_counted():
    clip, _, seg = segment(SynthParams(n_reps=8, seed=3, partial_first=True))
    assert len(seg.reps) == 8
    assert seg.partial_reps == 1
    assert all(r.first_phase == "concentric" for r in seg.reps)


def test_fidgeting_below_the_amplitude_floor_is_not_a_set():
    # 2% of a 90-degree range is ~2 degrees of movement: under MIN_REP_AMPLITUDE.
    _, _, seg = segment(SynthParams(n_reps=4, seed=4, rom_scale=0.02))
    assert seg.reps == ()
    assert seg.reason == "no_motion"


def test_selfsim_cross_check_agrees_on_clean_sets():
    for seed, ex in enumerate(EXERCISES):
        _, _, seg = segment(SynthParams(exercise_id=ex, n_reps=7, seed=40 + seed))
        assert seg.selfsim["agrees"], ex
        assert abs(seg.selfsim["count"] - 7) <= 1


def test_f01_last_rep_boundary_survives_noise_in_lead_out():
    # F-01: an unbounded velocity search let noise in the lead-out rest end the last rep 3.2 s late.
    _, _, m = run(SynthParams(exercise_id="tricep_pushdown", seed=1004, noise_px=8))
    assert m["abs_count_error"] == 0
    assert max_boundary_ms(m) < 600


def test_f02_heavy_fatigue_keeps_short_final_reps():
    # F-02: a fixed 30%-of-range prominence silently dropped the last two reps.
    _, _, m = run(SynthParams(exercise_id="preacher_curl", n_reps=12, rom_decay_per_rep=0.08, seed=3001))
    assert m["count_error"] == 0


def test_f02_noise_does_not_invent_reps():
    for seed, ex in enumerate(EXERCISES):
        _, _, m = run(SynthParams(exercise_id=ex, noise_px=25, seed=3100 + seed))
        assert m["count_error"] <= 0, ex


def test_f04_ambiguous_start_is_flagged_not_trusted():
    # F-04: a long turnaround pause plus a short lead-out makes the start position unknowable.
    clip, report, _ = run(SynthParams(exercise_id="tricep_pushdown", seed=6044, partial_first=True,
                                      turn_pause_s=0.81, lead_out_s=0.92, start_pause_s=0.64, noise_px=2.2))
    seg = report["diagnostics"]["segmentation"]
    assert seg["orientation_confident"] is False
    summary = report["set_summary"]
    assert summary["tempo"]["available"] is False
    assert summary["tempo"]["reason"] == "set_start_ambiguous"
    assert summary["reps"]["value"] == len(clip.truth.reps)   # counts are unaffected
