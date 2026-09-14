"""Subject selection (§5.2) and regression case F-03."""
import numpy as np

from core.pose.track import TrackConfig, select_subject
from core.pose.types import PoseFrames
from eval.harness.synthetic import SynthParams, generate
from tests._util import run


def _shifted_copy(p: PoseFrames, dx: float) -> PoseFrames:
    return PoseFrames(p.fps, p.width, p.height, p.n_frames, np.concatenate([p.frame, p.frame]),
                      np.concatenate([p.bbox, p.bbox + np.array([dx, 0, dx, 0], np.float32)]),
                      np.concatenate([p.kp, p.kp + np.array([dx, 0], np.float32)]),
                      np.concatenate([p.kp_score, p.kp_score]))


def test_lifter_is_chosen_over_a_passing_bystander():
    clip = generate(SynthParams(bystander=True, seed=51))
    sel = select_subject(clip.pose)
    assert sel.track is not None and not sel.ambiguous
    assert sel.track.present.mean() > 0.95
    assert len(sel.scores) == 2


def test_two_equally_plausible_lifters_abstain():
    clip = generate(SynthParams(seed=52))
    sel = select_subject(_shifted_copy(clip.pose, 420.0))
    assert sel.track is None
    assert sel.reason == "multiple_subjects_ambiguous"


def test_f03_detection_gaps_are_stitched_not_mistaken_for_two_people():
    clip = generate(SynthParams(seed=53))
    p = clip.pose
    keep = ~((p.frame >= 200) & (p.frame < 230))      # a 1-second detector dropout
    gapped = PoseFrames(p.fps, p.width, p.height, p.n_frames, p.frame[keep], p.bbox[keep],
                        p.kp[keep], p.kp_score[keep])
    sel = select_subject(gapped)
    assert sel.track is not None
    assert len(sel.scores) == 1


def test_f03_sequential_people_are_not_ambiguous():
    clip = generate(SynthParams(seed=54))
    p = clip.pose
    first = p.frame < p.n_frames // 2
    second = PoseFrames(p.fps, p.width, p.height, p.n_frames,
                        np.concatenate([p.frame[first], p.frame[~first]]),
                        np.concatenate([p.bbox[first], p.bbox[~first] + np.array([500, 0, 500, 0], np.float32)]),
                        np.concatenate([p.kp[first], p.kp[~first] + np.array([500, 0], np.float32)]),
                        np.concatenate([p.kp_score[first], p.kp_score[~first]]))
    assert select_subject(second).reason != "multiple_subjects_ambiguous"


def test_f03_intermittent_detection_abstains_truthfully():
    for seed in range(5):
        _, report, _ = run(SynthParams(miss_rate=0.5, seed=4000 + seed))
        assert report["status"] == "abstained"
        assert report["submission_abstention"]["reason"] == "subject_intermittently_detected"


def test_f03_moderate_detection_misses_still_count_correctly():
    for seed in range(5):
        _, report, m = run(SynthParams(miss_rate=0.3, seed=4100 + seed))
        assert report["status"] == "analysed"
        assert m["count_error"] == 0


def test_no_detections_abstain():
    clip = generate(SynthParams(seed=55))
    p = clip.pose
    empty = PoseFrames(p.fps, p.width, p.height, p.n_frames, p.frame[:0], p.bbox[:0], p.kp[:0], p.kp_score[:0])
    assert select_subject(empty, TrackConfig()).reason == "no_subject_detected"
