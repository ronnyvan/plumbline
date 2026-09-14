# Failure taxonomy (§10.5)

Every failure mode found, how it was found, what fixed it, and the regression test that keeps
it fixed. A failure that isn't here hasn't been understood yet.

All four below were found on **synthetic** data by stress-testing the signal path on
2026-09-13. None has yet been checked against real video — each one's real-world rate is an
open measurement for M1.

---

## F-01 — Tempo collapses under noise while rep counts stay perfect

- **Found by:** stress sweep. At 8 px keypoint noise (≈2.5% of torso length) rep counts were
  exact but some phase boundaries were off by **3.2 s**.
- **Why it was dangerous:** silent. A correct rep count sits next to wrong tempo numbers and
  nothing looks off.
- **Cause:** on the last rep the return phase's velocity search ran to the clip edge, across the
  whole lead-out rest, where a noise spike could out-run the real movement.
- **Fix:** search for peak velocity only while the limb is in transit (segment v2). Worst
  boundary error at 8 px fell from 3,233 ms to 467 ms. Residual degradation at extreme noise is
  handled by a timing gate: above a noise-to-range ratio of 0.015, tempo and TUT are withheld
  with `too_noisy_for_timing` (agnostic v2).
- **Tests:** `test_f01_last_rep_boundary_survives_noise_in_lead_out`,
  `test_noisy_clip_withholds_timing_but_keeps_counts`.
- **Open:** the 0.015 threshold comes from a synthetic sweep. Re-derive it against hand-annotated
  boundaries.

## F-02 — Heavy fatigue silently drops the final reps

- **Found by:** stress sweep. At 8% ROM loss per rep over 12 reps, every clip was exactly 2 reps
  short (off-by-one accuracy 0.00).
- **Why it was dangerous:** the dropped reps are the fatigue evidence the fatigue-slope metric
  exists to capture.
- **Cause:** rep detection required prominence of 30% of the set's range. Lowering it fixed
  fatigue but invented reps under noise — no fixed threshold serves both.
- **Fix:** noise-adaptive prominence: `max(0.12 × range, 8 × this clip's noise σ)` (segment v3).
  Parameters chosen by sweep on a separate stress seed family, never on the eval corpus.
- **Tests:** `test_f02_heavy_fatigue_keeps_short_final_reps`, `test_f02_noise_does_not_invent_reps`.
- **Open:** Q-09 — whether a rep at 12% of the set's range *should* count is a product
  definition, not a signal-processing question.

## F-03 — One lifter split into fragments, reported as two people

- **Found by:** stress sweep at 50% detector miss rate.
- **Why it was dangerous:** two ways. Five clips abstained telling the user "more than one person
  is moving" when there was one — a false abstention reason, which §7.5 forbids. One clip picked
  a fragment covering 32% of the set and confidently reported 5 reps out of 8.
- **Cause:** gaps longer than 8 frames started new tracks; ambiguity was judged between any two
  tracks, including ones never on screen together.
- **Fix:** stitch non-overlapping fragments that line up across a gap; judge ambiguity only
  between tracks that co-exist; abstain with `subject_intermittently_detected` when the chosen
  track covers under 60% of its span (track v2). Zero wrong answers at 50–70% miss rates; 30%
  miss rate now analyses every clip correctly.
- **Tests:** `test_f03_*` in `tests/unit/test_tracking.py`.

## F-04 — A clip that starts mid-rep can have its phases swapped

- **Found by:** the full 120-clip eval corpus, not the stress sweep. Phase-label accuracy was
  0.954 and boundary error 130–180 ms on a corpus that looked perfect by rep count.
- **Cause:** when the two clip edges disagree about the start position, the tie-breaker was "the
  longest still period", which jitter fragments on noisy clips.
- **Fix:** break the tie by contiguous dwell at each edge, then total dwell (segment v4). Eval
  corpus phase accuracy 1.0, boundary error 33–53 ms.
- **Residual:** on a held-out family of hard mid-rep starts, 3/60 clips still swap — all with a
  turnaround pause over 0.8 s and a lead-out under 1 s, which is genuinely ambiguous from the
  signal. All three were flagged `orientation_confident = false`. So phase-labelled metrics
  (per-phase tempo, concentric velocity) are withheld with `set_start_ambiguous` whenever
  orientation is unconfident; TUT and counts are unaffected (agnostic v3).
- **Tests:** `test_f04_ambiguous_start_is_flagged_not_trusted`, `test_partial_first_rep_is_excluded_and_counted`.
- **Note:** guided capture (§2.4) starts recording before the first rep, which should make this
  rare in the product. Measure how rare.
