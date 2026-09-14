"""Golden cases (spec §11.7): fixed inputs whose full reports are committed.

Regenerate deliberately with `make golden`, then review the diff like any other code change.
A golden diff you cannot explain is a bug.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from core.pipeline import analyze_pose
from eval.harness.run import _clean
from eval.harness.synthetic import SynthParams, generate

GOLDEN_DIR = Path(__file__).resolve().parent / "reports"

CASES = {
    "lat_pulldown_late_swing": SynthParams(n_reps=9, seed=11, swing_deg=20.0, swing_reps=(6, 7, 8)),
    "leg_press_eccentric_first": SynthParams(exercise_id="leg_press_quad_bias", n_reps=6, seed=22),
    "preacher_curl_fatigue": SynthParams(exercise_id="preacher_curl", n_reps=10, rom_decay_per_rep=0.04, seed=31),
    "pushdown_bystander_partial": SynthParams(exercise_id="tricep_pushdown", seed=71, bystander=True,
                                              partial_first=True),
    "intermittent_detection": SynthParams(miss_rate=0.5, seed=4000),
}


def build(name: str) -> dict:
    params = CASES[name]
    report = analyze_pose(generate(params).pose, params.exercise_id, station_requested="S-LAT")
    return _clean(report)


def main() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name in CASES:
        payload = json.dumps(build(name), indent=1, sort_keys=True) + "\n"
        (GOLDEN_DIR / f"{name}.json").write_text(payload, encoding="utf-8")
        print(f"wrote {name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
