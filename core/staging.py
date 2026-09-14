"""Content-addressed stage artifacts and the pipeline manifest (spec §4.2).

Every artifact is keyed by (input_hash, stage name, stage version, config hash). Two
results are comparable only when their manifest hashes match (§10.7).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage's identity. Bump `version` whenever its output can change."""
    name: str
    version: str
    config: dict = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return hash_obj(self.config)

    def as_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "config": self.config,
                "config_hash": self.config_hash}


def manifest(stages: list[StageSpec], extra: dict | None = None) -> dict:
    body = {"stages": [s.as_dict() for s in stages], "extra": extra or {}}
    return {**body, "manifest_hash": hash_obj(body)}


class ArtifactStore:
    """Filesystem cache: <root>/<stage>/<input_hash>/<key>.<ext>. Never holds source media."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def key(self, input_hash: str, stage: StageSpec) -> str:
        return hash_obj([input_hash, stage.name, stage.version, stage.config_hash])[:24]

    def path(self, input_hash: str, stage: StageSpec, ext: str) -> Path:
        return self.root / stage.name / input_hash[:16] / f"{self.key(input_hash, stage)}.{ext}"

    def get_json(self, input_hash: str, stage: StageSpec) -> dict | None:
        p = self.path(input_hash, stage, "json")
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put_json(self, input_hash: str, stage: StageSpec, payload: dict) -> Path:
        p = self.path(input_hash, stage, "json")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        tmp.replace(p)
        return p

    def delete_input(self, input_hash: str) -> int:
        """Remove every artifact derived from one input (§9.2 withdrawal). Returns files removed."""
        removed = 0
        if not self.root.exists():
            return 0
        for stage_dir in self.root.iterdir():
            target = stage_dir / input_hash[:16]
            if target.is_dir():
                for f in target.iterdir():
                    f.unlink()
                    removed += 1
                target.rmdir()
        return removed
