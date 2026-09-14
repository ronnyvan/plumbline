# Plumbline — session handoff

**Session date:** 2026-09-13
**Read this first when resuming.** Then read [`failures.md`](failures.md) and the parts of
[`spec/plumbline.html`](spec/plumbline.html) that the next task touches. The published copy of the spec
is at <https://claude.ai/code/artifact/b62fc641-33be-4117-9f55-ae24f5be279f> (version 2, matches the repo).

---

## 1. Where things stand

The code for milestones M0 and M1 exists and passes its tests. **Neither milestone's exit criterion is met,
and nothing has been run on real video.** Every number in the repository comes from synthetic lifts with
known ground truth, and the results file marks them `admissible: false`.

| Milestone (spec §13) | Code | Exit criterion | Met? |
|---|---|---|---|
| M0 — skeleton, registry, eval harness, CI | Built | `make eval` on 3 hand-placed real clips | **No.** 0 real clips exist |
| M1 — ingest → pose → segmentation → Tier B metrics | Built, tested on synthetic data | Rep-count MAE < 0.5 and off-by-one > 0.90 on 100 hand-annotated clips | **No** |
| M2–M10 | Not started | | |

**Repository state:**

- Location: `C:\Project`
- `git init` has run, but there are **no commits**. A first commit would contain 90 files. I checked that list: no video, no model weights, no caches, nothing that looks like a secret.
- 84 tests pass (`make test`). The 2 GPU tests pass (`make test-pose`). Lint is clean. `make verify` confirms the committed results regenerate exactly.
- The spec was amended from v0.1 to v0.2 and republished (§3 below).

---

## 2. What was built

### Environment

| Item | Value |
|---|---|
| Python | 3.12.0, virtual environment in `.venv/` |
| GPU | RTX 4090, 24 GB, driver 591.86 |
| Pose runtime | ONNX Runtime GPU 1.30.0 with CUDA 13 and cuDNN 9 wheels; rtmlib 0.0.16 |
| Pose models | YOLOX-m person detector + RTMPose-m Halpe-26, cached in `~/.cache/rtmlib/hub/checkpoints/` |
| Pinned dependencies | `requirements.txt`, `requirements-dev.txt`, `requirements-pose.txt` |

### Components

| Area | Files | What it does |
|---|---|---|
| Registry | `registry/stations.yaml`, `registry/cues.yaml`, `registry/exercises/*.yaml`, `registry/validation.yaml`, `core/registry.py` | 5 camera stations, 56 cues and 12 exercises, stored as data. Every cross-reference is validated on load. Tier promotions must cite evidence. |
| Keypoints | `core/keypoints.py` | The Halpe-26 schema and joint-angle triplets |
| Ingest | `core/ingest/video.py` | Resamples variable-frame-rate video to 30 fps by timestamp, caps the long side at 1080 px, trims to the active window, and records sharpness, motion and a perceptual hash. Rejects clips under 2 s or over 90 s. |
| Pose | `core/pose/rtmpose.py`, `core/pose/types.py` | RTMPose on the GPU. It errors if CUDA silently falls back to CPU, and never produces a skeleton for a frame with no detected person. Pose output is stored as Parquet. |
| Subject selection | `core/pose/track.py` | Scores each tracked person, stitches broken tracks, and abstains when the lifter is ambiguous or seen too rarely. |
| Features | `core/features/signals.py`, `core/features/quality.py` | Image-plane joint angles, confidence-weighted side fusion, gap filling of 4 frames or fewer, Savitzky–Golay smoothing, torso angle, and the per-window quality score *Q* |
| Segmentation | `core/segment/reps.py`, `core/segment/selfsim.py` | Signal-based rep detection with noise-adaptive prominence, velocity-based phase boundaries, and phase order read from the signal. A separate self-similarity rep count serves as a cross-check. |
| Tier B metrics | `core/cues/agnostic.py` | Reps, partial reps, ROM per rep, ROM variation, tempo, TUT, fatigue slopes and asymmetry. Each is withheld, with a reason, when unreliable. |
| Cue engine | `core/cues/contract.py`, `core/cues/engine.py`, `core/gates/observability.py`, `core/cues/lp01_torso_swing.py` | Detector contract, observability gate with remedy text, quality gate, and one rule detector (LP-01) |
| Report | `core/report.py` | Report JSON ordered as spec §2.5. Abstentions stay visible and carry user-facing copy. |
| Pipeline | `core/pipeline.py`, `core/staging.py` | `analyze_pose` is the single entry point used by both production and evaluation. `analyze_video` adds cached ingest and pose in front of it. Every result carries a content-addressed cache and a manifest hash. |
| CLI | `core/cli.py` | `make analyze VIDEO=… EXERCISE=… STATION=…` |
| Evaluation | `eval/harness/*.py`, `eval/metrics/*.py` | Synthetic generator, synthetic and real corpora, subject-disjoint splits, cluster bootstrap intervals, L2 segmentation metrics, and the reproducibility and leakage gates |
| Results | `eval/results/eval.json`, `eval/results/REPORT.md` | Committed and generated, never written by hand |
| Tests | `tests/unit`, `tests/property`, `tests/golden` | 84 tests: every failure found this session, the spec §11.7 cases, property tests and 5 golden reports |
| CI | `.github/workflows/ci.yml` | Lint, tests, leakage gate and reproducibility gate. It has not run yet, because the repository has no remote. |
| Documentation | `README.md`, `failures.md`, `DATASHEET.md`, `data/clips/README.md`, `rubrics/_TEMPLATE.md` | |

### Numbers measured this session

These are real measurements, but none of them was taken on real video.

| Measurement | Result | Caveat |
|---|---|---|
| Person detector latency | 15.0 ms per frame | Random 1280×720 image on the 4090 |
| Pose model latency | 3.1 ms per person | Same conditions |
| Rep-count MAE, synthetic eval corpus | 0.000 [0.000, 0.000], 120 clips, 40 subjects | Inadmissible |
| Phase-label accuracy, synthetic eval corpus | 1.000 [1.000, 1.000] | Inadmissible |
| Phase-boundary error, synthetic eval corpus | 33–53 ms mean, depending on the boundary | Inadmissible |
| Stress limits | Counts stay exact up to 25 px of keypoint noise and a 30% detector miss rate. At 40% misses or more the system abstains instead of answering. | Synthetic |

---

## 3. Decisions, errata and open questions added to the spec (v0.2)

All of these are recorded in spec §0.2, §0.4 and §15.

**Decisions**

| ID | Decision |
|---|---|
| D-13 | The keypoint schema is Halpe-26, not COCO-17, because feet are needed for the foot-placement cues. |
| D-14 | rtmlib is installed with `--no-deps`. A CUDA request that lands on CPU is an error, and the provider actually used is part of the manifest. |
| D-15 | Decoded frames are never stored. Ingest stores a frame index, and later stages re-decode. |
| D-16 | Phase order is read from the signal. Leg press sets start with the eccentric phase. |
| D-17 | The M1 cross-check is a self-similarity period estimate, not the RepNet network. |
| D-18 | Detectors with provisional thresholds may run, but can never produce a user-visible claim. |
| D-19 | Detectors return a type with no confidence field. Only the engine constructs `CueResult`. |
| D-20 | LP-01 fires on the k-th largest per-rep amplitude, not the median. The median failed the spec's own test case. |
| D-21 | Tier promotions live in `validation.yaml`, must cite evidence, and are made by hand. |
| D-22 | Tracking stitches fragments, and abstains when the lifter is detected in under 60% of frames. |
| D-23 | Tier B metrics have their own reliability gates for timing noise and phase ambiguity. |
| D-24 | Rep-detection prominence adapts to each clip's noise. |

**Errata**

| ID | Correction |
|---|---|
| E-01 | The cue counts in the prose were wrong. The table has 56 cues split 37/13/6; all six expected abstentions are joint-angle (KIN) cues; and LP-04 is a KIN Tier A candidate. |
| E-02 | 81.1 AP was misattributed to ViTPose-H. |
| E-03 | A centred Savitzky–Golay filter has no phase lag, so there is nothing to correct for. |
| E-04 | The median statistic in §6.2 contradicted the test case in §11.7. |

**Open questions:** Q-07 (pec fly has no single camera position that serves it), Q-08 (keypoint scores are not
probabilities), Q-09 (when does a short final rep count?).

**Failures found and fixed:** F-01 to F-04. The details and regression tests are in [`failures.md`](failures.md).

---

## 4. Known gaps and untested paths

Read this before trusting anything. Every item is something a reasonable person could assume works when it
has not been shown to.

1. **The real-video path has never run end to end.** `analyze_video`, `make analyze` and the harness's
   real-clip branch (`run_real`) are written but unexercised. Ingest was tested on OpenCV-generated videos.
   The pose backend was tested on blank and random frames. Expect the first real clip to break something.
2. **Segmentation is only tested on 5 of the 12 exercises.** The synthetic generator has templates for
   lat pulldown, preacher curl, leg extension, leg press (quad bias) and tricep pushdown. Pec fly,
   chest-supported row, upper-back row, leg curl, leg press (glute bias), overhead tricep extension and
   incline press have registry entries but no test coverage. Spec success criterion SC-9 (12 of 12
   exercises) is not demonstrated.
3. **LP-01 always abstains on real clips today.** The camera position can't be verified until equipment
   anchoring (M5) exists, so every station-dependent cue abstains with `station_unverified`. This is the
   intended behaviour, not a bug.
4. **Several thresholds are provisional**, set from synthetic sweeps rather than real data:
   - keypoint confidence floor 0.3 (Q-08)
   - `TIMING_MAX_NOISE_TO_RANGE = 0.015`
   - prominence `0.12 × range` or `8 × noise`
   - subject-coverage minimum 0.6
   - sharpness half-point 100
   - LP-01 threshold 15°
5. **The synthetic generator is a flat 2D stick figure** with a fixed far-side offset. Real occlusion,
   clothing, lens distortion and camera shake are not modelled.
6. **Not built:** calibration and confidence (M6), equipment anchoring (M5), labeling tool and Krippendorff α
   (M2), coaching layer (M7), API, worker and web app (M8), slice analysis (M10), and the L0, L1, L3, L4 and
   L5 evaluation layers.
7. **Not committed**, and CI has never run on a remote.

---

## 5. How to resume

```sh
cd C:\Project
make test          # expect: 84 passed, 2 deselected
make test-pose     # expect: 2 passed (confirms CUDA is in use)
make verify        # expect: OK: eval.json reproduces exactly
```

If `make test-pose` fails with a CUDA error, run `make install-pose` again. The CUDA libraries come from
the `nvidia-*` wheels, which `core/pose/rtmpose.py` loads via `ort.preload_dlls()`.

### Rules that keep the project honest

These are for you, and for any future Claude session working on the repo.

- **Change the spec first.** Record new choices as a `D-##` entry and mistakes as an `E-##` entry. Never renumber existing IDs.
- **Bump a stage's `*_VERSION`** whenever its output can change. Then run `make eval`, `make golden` and `make report`, and commit the regenerated files in the same commit. CI's `verify` step fails otherwise.
- **Synthetic results never promote a tier or satisfy a milestone.** Promotions are hand edits to `registry/validation.yaml` that cite admissible evidence.
- **Every new failure mode** gets an `F-##` entry in `failures.md` and a regression test.
- **Choose parameters on a separate stress seed family** (seeds 3000 and up), never on the eval corpus (`CORPUS_SEED`) or on test data.
- **No media in git.** Clips stay in `data/clips/`, which is ignored.

---

## 6. Next steps, in order

### Step 0 — this week

- [ ] **Commit the work.** `git add -A`, then run `git status` and check that no video or weights are included, then commit. Optionally push to a private GitHub repository so CI runs.
- [ ] **File the IRB determination (R-05).** It has weeks of lead time and blocks any publication use of clips of other people. Clips of yourself do not need it.
- [ ] **Draft the consent form** with separate analysis, training and publication checkboxes (spec §9.2), before filming anyone else.

### Step 1 — close M0: three real clips of yourself

- [ ] Film three sets from the lateral station (`S-LAT`): one lat pulldown, one preacher curl, and one leg press or tricep pushdown. Phone at mid-torso height, about 3 m away. Start recording before the first rep, stop after the last, nobody else in frame.
- [ ] Before labeling anything, run one clip:
  `make analyze VIDEO=data/clips/<clip>.mp4 EXERCISE=lat_pulldown STATION=S-LAT`
  Open the `.report.json`. Check the rep count against what you did, check which person was selected, and check the pose provider says CUDA.
- [ ] **Fix what breaks.** This is the first contact with real data. Record each surprise as an `F-##` entry with a test. Likely trouble spots:
  - the far-side limb from a lateral view being confidently wrong rather than low-confidence (side fusion)
  - equipment parts detected as people (subject selection)
  - the activity window trimming off a slow first rep
  - the timing gate withholding tempo on every real clip, which would mean the 0.015 threshold is wrong
- [ ] Label all three clips using [`data/clips/README.md`](data/clips/README.md). Watch at quarter speed and read times from a player that shows milliseconds.
- [ ] Run `make eval` and `make report`. The M0 row flips to met. Commit the labels, `eval.json` and `REPORT.md` together.

### Step 2 — reach M1

- [ ] **Add synthetic templates for the other 7 exercises** in `eval/harness/synthetic.py`. Pec fly needs a frontal-view template, because its rep signal is wrist separation (Q-07). Add them to the stress sweeps and golden cases. This closes gap 2 before spending time on real footage of those exercises.
- [ ] **Collect about 100 hand-annotated real clips** across as many exercises and people as possible. Record `subject_id` and `gym_id` from the first clip. Aim for variety in clothing fit, lighting and phone distance (spec §9.1).
- [ ] **Label rep boundaries with a second person on a subset** so boundary disagreement is measured. Rater disagreement puts a floor under how well segmentation can ever score.
- [ ] **Run the L1 keypoint study (Q-08).** Hand-annotate keypoints on about 200 frames. Relate RTMPose scores to actual pixel error. Replace the 0.3 floor with a calibrated value. Optionally run ViTPose-H as a reference ceiling (the 4090 can handle it).
- [ ] **Re-derive the provisional thresholds** (gap 4) on real data, using a validation split and never the test split.
- [ ] **Promote exercises that pass the gate.** Where per-exercise rep-count MAE < 0.5 and off-by-one > 0.90 on real clips, set `validated: true` in `registry/validation.yaml` with the evidence path. Each promotion is its own reviewed commit.
- [ ] **Settle Q-09** (does a grinding partial count as a rep?) in the rep-boundary rubric.

### Step 3 — M2: labeling infrastructure

- [ ] Choose the first three Tier A cues (Q-01). The spec recommends lat pulldown, preacher curl and incline press.
- [ ] Write their rubrics from `rubrics/_TEMPLATE.md` (spec §3.5).
- [ ] Build or adopt a labeling tool (Label Studio is fine): rep scrubbing, rubric shown inline, a borderline option, blind double labeling.
- [ ] Implement Krippendorff's α in `eval/metrics/`, and a gold-item onboarding check for raters.
- [ ] Recruit raters from the org. Run a 30-clip dry run per cue. Ship gate: α ≥ 0.67.

### Step 4 onwards — spec roadmap

| Milestone | Work | Spec sections |
|---|---|---|
| M3 | Collect at least 150 clips and 25 subjects for each of the first three exercises; freeze splits | §9 |
| M4 | Rule detectors for the chosen cues; L3 per-cue F1 with bootstrap intervals; the first real LP-01 threshold | §6, §10 |
| M5 | Equipment landmark detector, camera pose, station classifier (unblocks every station-dependent cue), weight-stack scale | §5.4 |
| M6 | Calibration split, isotonic calibration, conformal thresholds, risk–coverage curves, **random-abstention control** | §7 |
| M7 | Evidence corpus, grounded coaching renderer, deterministic verifier, human-validated LLM judge | §8 |
| M8 | API, queue, GPU worker, web app with guided capture, first real users | §2, §11 |
| M9 | Learned detectors that replace rules only after passing McNemar with Benjamini–Hochberg correction | §6.2, §10.3 |
| M10 | Slice and gym-disjoint analysis, dataset card, workshop paper draft | §10.4, §14 |

### Decisions only you can make

- **Q-01** — which three exercises get Tier A treatment first
- **Q-04** — whether to run the two-phone multi-view study (the strongest upgrade to the research claim)
- **Q-07** — how pec fly is captured: two submissions per set, or a lateral-view rep signal
- **Q-09** — what counts as a rep

---

## 7. File map

```
HANDOFF.md            this file
README.md             project overview and setup
failures.md           F-01..F-04, each with its regression test
DATASHEET.md          dataset card template (§9.4)
spec/plumbline.html   the design, v0.2 — source of truth
registry/             stations, 56 cues, 12 exercises, tier promotions
core/                 shared analysis core
  ingest/ pose/ features/ segment/ cues/ gates/
  pipeline.py         analyze_pose / analyze_video
  cli.py              make analyze
  report.py           report JSON
  staging.py          content-addressed cache and manifest
eval/
  harness/            synthetic generator, corpora, splits, run.py (make eval, verify, report)
  metrics/            L2 segmentation metrics, cluster bootstrap
  results/            eval.json, REPORT.md (committed)
tests/                unit, property, golden
data/clips/           your clips (git-ignored) and their labels (committed)
rubrics/              rubric template (M2)
artifacts/            stage cache (git-ignored, safe to delete)
```
