"""Eval harness behaviour (§10.6, §10.7)."""
import json

from eval.harness import corpus, run
from eval.harness.splits import assign_split


def test_fast_results_are_strict_json_and_deterministic():
    a = run.build_results(fast=True, include_real=False)
    b = run.build_results(fast=True, include_real=False)
    json.dumps(a, allow_nan=False)                  # raises on NaN or numpy types
    assert run._diff(run._comparable(a), run._comparable(b)) == []


def test_synthetic_evidence_can_never_meet_a_milestone():
    results = run.build_results(fast=True, include_real=False)
    assert results["corpora"]["synthetic"]["admissible"] is False
    assert results["milestones"]["M0"]["met"] is False
    assert results["milestones"]["M1"]["met"] is False
    assert results["layers"]["L3"]["status"] == "not_implemented"


def test_compare_refuses_mismatched_manifests(tmp_path, capsys):
    base = run.build_results(fast=True, include_real=False)
    other = json.loads(json.dumps(base))
    other["manifest"]["manifest_hash"] = "0" * 64
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(base), encoding="utf-8")
    pb.write_text(json.dumps(other), encoding="utf-8")
    assert run.main(["compare", str(pa), str(pb)]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert run.main(["compare", str(pa), str(pb), "--allow-manifest-mismatch"]) == 0


def test_diff_tolerates_float_noise_but_not_real_changes():
    assert run._diff({"x": 1.0}, {"x": 1.0 + 1e-9}) == []
    assert run._diff({"x": 1.0}, {"x": 1.01}) != []
    assert run._diff({"n": 3}, {"n": 4}) != []


def test_real_clips_without_consent_are_refused_before_reading(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    labels = {"clip": "a.mp4", "subject_id": "s1", "gym_id": "g1", "exercise": "lat_pulldown",
              "station": "S-LAT", "consent": {"analysis": False}, "annotator": "r1",
              "reps": [{"start_s": 1, "turn_arrive_s": 2, "turn_leave_s": 2.2, "end_s": 3,
                        "first_phase": "concentric"}]}
    (tmp_path / "a.labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (tmp_path / "b.labels.json").write_text("{not json", encoding="utf-8")
    items, refused = corpus.real_items(tmp_path)
    assert items == []
    assert {r["reason"] for r in refused} == {"no_analysis_consent", "invalid_labels"}


def test_labeled_seconds_map_to_nearest_output_frame():
    labels = corpus.ClipLabels(clip="x.mp4", subject_id="s", gym_id="g", exercise="lat_pulldown",
                               station="S-LAT", consent={"analysis": True}, annotator="r",
                               reps=[{"start_s": 0.10, "turn_arrive_s": 0.5, "turn_leave_s": 0.52,
                                      "end_s": 1.0, "first_phase": "concentric"}])
    ts = [i / 30 for i in range(40)]
    [rep] = corpus.real_truth_reps(labels, ts)
    assert (rep["start"], rep["turn_arrive"], rep["end"]) == (3, 15, 30)


def test_corpus_subjects_are_assigned_by_id_alone():
    for item in corpus.synthetic_items(n_subjects=10):
        assert item.split == assign_split(item.subject_id)
