# Hand-placed clips

Each clip is a video plus a labels file beside it:

```
data/clips/
  lat_pulldown_2026-09-14_a.mp4           never committed (git-ignored)
  lat_pulldown_2026-09-14_a.labels.json   committed
```

## Consent comes first (§9.2)

The harness reads the labels file before it opens the video, and refuses any clip without
`"analysis": true`. If the person in the clip is not you, you need their written consent for
each use you tick — and `training` and `publication` are separate from `analysis` for a reason.

Nobody else may be identifiable in the working area. If they are, don't label the clip.

## Label format

```json
{
  "clip": "lat_pulldown_2026-09-14_a.mp4",
  "subject_id": "s001",
  "gym_id": "g01",
  "exercise": "lat_pulldown",
  "station": "S-LAT",
  "consent": {"analysis": true, "training": false, "publication": false},
  "annotator": "r01",
  "reps": [
    {"start_s": 2.10, "turn_arrive_s": 3.30, "turn_leave_s": 3.55, "end_s": 5.40, "first_phase": "concentric"}
  ],
  "partial_reps": 0
}
```

- **`subject_id`** is a stable pseudonym. Never a name. The same person always gets the same
  id, because splits are assigned by it (D-10) and the same person in train and test inflates
  every number.
- **`exercise`** and **`station`** must exist in `registry/`.
- **Times are seconds from the start of the file**, read off a video player that shows
  milliseconds. The harness maps them to frames using the ingest timestamps, so frame rate
  doesn't matter.
- Per rep:
  - `start_s` — the first instant the working limb moves away from the start position
  - `turn_arrive_s` — it reaches the turnaround (top of a curl, bottom of a pulldown)
  - `turn_leave_s` — it starts moving back (equal to `turn_arrive_s` if there's no pause)
  - `end_s` — it is back at the start position
  - `first_phase` — `concentric` if the set starts with the lifting effort (a curl, a pulldown),
    `eccentric` if it starts with the lowering (a leg press from lockout)
- List only **complete** reps. Count reps the clip cuts through in `partial_reps`.

Label by watching at quarter speed. Where you genuinely can't tell when a rep started, note it
in the commit message rather than guessing: those clips are the ones the M1 study needs to know
about.
