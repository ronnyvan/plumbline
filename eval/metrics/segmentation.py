"""L2 — segmentation metrics (spec §10.1).

Truth and prediction are matched on turnaround time, the most sharply defined instant in
a rep. Boundary errors are reported per boundary, because a segmenter that is exact at
the turnaround and 300 ms late at motion onset produces tempo numbers that are wrong in
a specific, diagnosable way.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from eval.metrics.stats import cluster_bootstrap

BOUNDARIES = ("start", "turn_arrive", "turn_leave", "end")
GATE_MAE = 0.5          # §13 M1 exit criterion
GATE_OFF_BY_ONE = 0.90


def match_reps(pred: list[dict], truth: list[dict], fps: float) -> list[tuple[dict, dict]]:
    if not pred or not truth:
        return []
    periods = np.diff([t["turn_arrive"] for t in truth])
    tolerance = 0.5 * float(np.median(periods)) if len(periods) else 2.0 * fps
    pairs = sorted(((abs(p["turn_arrive"] - t["turn_arrive"]), i, j)
                    for i, p in enumerate(pred) for j, t in enumerate(truth)))
    used_p, used_t, out = set(), set(), []
    for dist, i, j in pairs:
        if dist > tolerance or i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        out.append((pred[i], truth[j]))
    return sorted(out, key=lambda pt: pt[1]["turn_arrive"])


def item_metrics(pred: list[dict], truth: list[dict], fps: float) -> dict:
    n_pred, n_true = len(pred), len(truth)
    pairs = match_reps(pred, truth, fps)
    errors = {b: [1000.0 * abs(p[b] - t[b]) / fps for p, t in pairs] for b in BOUNDARIES}
    phase_ok = [p["first_phase"] == t["first_phase"] for p, t in pairs]
    return {"n_pred": n_pred, "n_true": n_true, "count_error": n_pred - n_true,
            "abs_count_error": abs(n_pred - n_true), "off_by_one_ok": abs(n_pred - n_true) <= 1,
            "exact_ok": n_pred == n_true, "matched": len(pairs),
            "boundary_error_ms": errors, "phase_correct": phase_ok}


def aggregate(items: list[dict], *, n_resamples: int, seed: int) -> dict:
    """items: dicts with subject_id, exercise_id and the output of item_metrics."""
    if not items:
        return {"clips": 0}

    def summary(rows: list[dict]) -> dict:
        subj = [r["subject_id"] for r in rows]
        out = {
            "clips": len(rows),
            "subjects": len(set(subj)),
            "rep_count_mae": cluster_bootstrap([r["abs_count_error"] for r in rows], subj,
                                               n_resamples=n_resamples, seed=seed),
            "off_by_one_accuracy": cluster_bootstrap([float(r["off_by_one_ok"]) for r in rows], subj,
                                                     n_resamples=n_resamples, seed=seed),
            "exact_count_accuracy": cluster_bootstrap([float(r["exact_ok"]) for r in rows], subj,
                                                      n_resamples=n_resamples, seed=seed),
            "mean_count_error": round(float(np.mean([r["count_error"] for r in rows])), 4),
        }
        phase_v, phase_s = [], []
        for r in rows:
            phase_v += [float(x) for x in r["phase_correct"]]
            phase_s += [r["subject_id"]] * len(r["phase_correct"])
        out["phase_label_accuracy"] = cluster_bootstrap(phase_v, phase_s, n_resamples=n_resamples, seed=seed)
        out["boundary_error_ms"] = {}
        for b in BOUNDARIES:
            vals, subs = [], []
            for r in rows:
                vals += r["boundary_error_ms"][b]
                subs += [r["subject_id"]] * len(r["boundary_error_ms"][b])
            out["boundary_error_ms"][b] = cluster_bootstrap(vals, subs, n_resamples=n_resamples, seed=seed)
        mae, obo = out["rep_count_mae"]["value"], out["off_by_one_accuracy"]["value"]
        out["l2_gate"] = {"rep_count_mae_lt": GATE_MAE, "off_by_one_gt": GATE_OFF_BY_ONE,
                          "passes": bool(mae is not None and mae < GATE_MAE and obo > GATE_OFF_BY_ONE)}
        return out

    by_ex: dict[str, list[dict]] = defaultdict(list)
    for r in items:
        by_ex[r["exercise_id"]].append(r)
    return {**summary(items), "by_exercise": {ex: summary(rows) for ex, rows in sorted(by_ex.items())}}
