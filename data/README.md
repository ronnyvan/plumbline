# data/

Manifests, labels and consent records only. **No media, ever** (§9.2, §11.2): videos, frames
and derived pose files are git-ignored, and derived artifacts live in `artifacts/`, which can
be deleted per input with `ArtifactStore.delete_input` when someone withdraws.

- `clips/` — hand-placed clips and their labels. See `clips/README.md`.
- `manifests/` — optional `*.csv` files with `subject_id,split` columns. The leakage gate reads
  every row and fails if a subject appears in more than one split.
