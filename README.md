# Plumbline

Monocular video analysis for machine and cable hypertrophy training that reports form errors
**only where the camera geometry, perception quality and calibrated confidence support the
claim** — and says plainly, with a fix, when they don't.

The design lives in [`spec/plumbline.html`](spec/plumbline.html). It is the source of truth:
change the spec, record the decision, then change the code. Section references like `§7.4` and
IDs like `D-16` or `F-02` point into it and into [`failures.md`](failures.md).

## Where this is

To pick up where the last session left off, start with [`HANDOFF.md`](HANDOFF.md).

| Milestone (§13) | State | Exit criterion |
|---|---|---|
| **M0** skeleton, registry, eval harness, CI | built; exit **not met** | `make eval` on 3 hand-placed **real** clips — none yet |
| **M1** ingest → pose → segmentation → Tier B metrics | built and tested on synthetic data; exit **not met** | rep-count MAE < 0.5, off-by-one > 0.90 on 100 hand-annotated clips |
| M2 labeling tool, rubrics, rater onboarding | not started | |
| M3–M10 data, cues, anchoring, calibration, coaching, product | not started | |

The committed results in [`eval/results/REPORT.md`](eval/results/REPORT.md) say the same thing,
because they are generated, not written.

**Nothing here has touched real gym video yet.** The pipeline has run end to end on synthetic
lifts with exact ground truth, and the GPU pose backend has been verified to load on CUDA — but
no number in this repository is evidence about how it performs on a real person. The next step
that matters is filming three sets of yourself and labelling them (below).

## What works today

- **Registry as data** — 12 exercises, 56 cues, 5 camera stations, validated cross-references
  (`registry/`). Tier promotions require committed evidence (`registry/validation.yaml`).
- **Ingest** — variable-frame-rate phone video resampled to 30 fps by timestamp, sharpness and
  motion statistics, activity trimming, duration and size limits.
- **Pose** — RTMPose-m (Halpe-26) on ONNX Runtime GPU. A CUDA request that silently lands on
  CPU is an error, and frames with no detected person produce no skeleton (D-14).
- **Subject selection** — scored choice between people, fragment stitching, and truthful
  abstention when the lifter is ambiguous or intermittently visible (F-03).
- **Rep segmentation** — phase order inferred rather than assumed, so eccentric-first lifts
  like the leg press are labelled correctly (D-16); noise-adaptive rep detection (F-02); an
  independent self-similarity count as a cross-check (D-17).
- **Tier B metrics** for every exercise — reps, ROM per rep, ROM consistency, tempo, TUT,
  fatigue slopes, asymmetry — each withheld with a reason when it can't be trusted (F-01, F-04).
- **Cue engine** with the observability gate and quality gate. One rule detector (LP-01, torso
  swing) exists with a provisional threshold; it computes but can never be shown to a user until
  its threshold is validated and its confidence calibrated (D-18).
- **Evaluation** — subject-disjoint splits, clustered bootstrap intervals, L2 segmentation
  metrics, error analysis, a reproducibility gate, and a leakage gate.

## Setup

Python 3.12. On Windows, `make` and a POSIX `sh` (Git for Windows provides one) are needed.

```sh
make install          # .venv + runtime and dev dependencies
make test             # 84 tests, no GPU needed
make eval             # writes eval/results/eval.json
make verify           # fails unless committed results regenerate exactly
make report           # writes eval/results/REPORT.md
```

For real video you need the pose stack and an NVIDIA GPU:

```sh
make install-pose     # ONNX Runtime GPU + CUDA/cuDNN wheels + rtmlib (without its deps)
make test-pose        # confirms CUDA is actually in use
make analyze VIDEO=my_set.mp4 EXERCISE=lat_pulldown STATION=S-LAT
```

`rtmlib` is installed with `--no-deps` on purpose: its declared dependencies pull in three
OpenCV builds and CPU ONNX Runtime, which overwrite the headless and GPU builds this project
uses.

## Adding your first real clips

This is what closes M0. See [`data/clips/README.md`](data/clips/README.md) for the label format.

1. Film a set from the station the exercise asks for (`registry/exercises/<name>.yaml`).
   Start recording before your first rep and stop after your last.
2. Put the video in `data/clips/` (videos are git-ignored and never committed).
3. Write `<clip>.labels.json` with the rep boundaries in seconds and your consent.
4. `make eval`. The M0 row of the report flips when three clips process.

## Layout

```
spec/            the design; source of truth
registry/        exercises, cues, stations, tier promotions (data, not code)
core/            the shared analysis core — production and eval both call this (D-11)
  ingest/ pose/ features/ segment/ cues/ gates/
  pipeline.py    the single entry point
eval/            harness, metrics, committed results
tests/           unit, property and golden tests
data/            clip labels and split manifests only — never media
failures.md      every failure mode found, with its regression test
DATASHEET.md     dataset card (§9.4)
```

Not yet present, by milestone: `rubrics/` content (M2), `labeling/` (M2), `core/anchor/` (M5),
`coaching/` (M7), `api/`, `worker/`, `web/` (M8), `eval/slices/` (M10).
