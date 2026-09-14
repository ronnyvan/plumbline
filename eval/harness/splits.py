"""Subject-disjoint splits and the leakage gate (spec §10.2, §10.7, D-10).

Assignment is a pure function of subject_id, so a subject can never land in two splits
by construction. The leakage check exists anyway, for manifests edited by hand.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

SPLIT_SALT = "plumbline-splits-v1"
FRACTIONS = (("train", 0.55), ("validation", 0.15), ("calibration", 0.15), ("test", 0.15))


def assign_split(subject_id: str, salt: str = SPLIT_SALT) -> str:
    digest = hashlib.sha256(f"{salt}:{subject_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2 ** 64
    cumulative = 0.0
    for name, frac in FRACTIONS:
        cumulative += frac
        if u < cumulative:
            return name
    return FRACTIONS[-1][0]


def check_leakage(rows: Iterable[tuple[str, str]]) -> list[dict]:
    """rows: (subject_id, split). Returns one violation per subject found in several splits."""
    seen: dict[str, set[str]] = defaultdict(set)
    for subject, split in rows:
        seen[subject].add(split)
    return [{"subject_id": s, "splits": sorted(v)} for s, v in sorted(seen.items()) if len(v) > 1]


def manifest_rows(manifest_dir: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted(Path(manifest_dir).glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                rows.append((rec["subject_id"], rec["split"]))
    return rows
