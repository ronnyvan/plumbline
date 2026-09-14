# Dataset card (§9.4)

**Status: no real data collected.** This card is a template with the parts that are already
decided filled in. Fill in the rest as collection happens; the known-gaps section is the part
reviewers will read first.

## Motivation

To measure, per form cue, how accurately monocular phone video of machine and cable exercises
supports a claim — and where it cannot — for a coaching product that abstains rather than
guesses (spec §1).

## Composition

- **Unit:** one set of one exercise by one subject, filmed from one camera station.
- **Labels:** rep boundaries (seconds), and later per-cue rubric labels (M2).
- **Ground truth is experienced-lifter consensus**, not clinical or credentialed truth (D-05).
- **Planned targets per exercise:** 150–200 clips, ≥ 25 subjects, ≥ 4 gyms, ≥ 3 machine variants (§9.1).
- **Demographic composition:** _to be filled from collection records._ Tracked from clip one:
  body size, limb proportion, skin tone (Monk scale), clothing fit, gym lighting, device and
  distance, training experience.

## Collection

- Guided capture from the station each exercise prescribes (`registry/exercises/`).
- Written, granular consent: analysis, training and publication separately (§9.2).
- Bystanders: none identifiable in the working area.
- IRB determination: _not yet filed._ Required before any publication use (R-05).

## Preprocessing

Ingest resamples to 30 fps by timestamp and caps the longest side at 1080 px. Pose is RTMPose-m
Halpe-26. Pipeline versions are recorded in every result's manifest.

## Uses

- In scope: evaluating and calibrating this system's cues.
- Out of scope: identifying people, medical or injury assessment, training models for purposes
  the subject did not consent to.

## Distribution

Not distributed. Media never enters the repository; labels do.

## Maintenance

Withdrawal deletes the source video and every derived artifact (`ArtifactStore.delete_input`,
tested in `test_withdrawal_deletes_every_derived_artifact`).

## Known gaps

- No real data yet. Every published number so far comes from synthetic clips and is marked inadmissible.
- Keypoint confidence scores from RTMPose are not probabilities; the confidence floor and the
  quality score's confidence term are provisional (Q-08).
- _Add every gap discovered during collection here, especially under-collected slices._
