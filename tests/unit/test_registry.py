"""Registry integrity (§3.1, §3.4). These tests are how spec erratum E-01 was found."""
import shutil
from collections import Counter
from pathlib import Path

import pytest
import yaml

from core.cues.engine import DETECTORS
from core.registry import ANY_STATION, load_registry

REG = load_registry()


def test_twelve_exercises_and_fifty_six_cues():
    # E-01: the §3.4 table has 56 rows. The spec prose said 52.
    assert len(REG.exercises) == 12
    assert len(REG.cues) == 56


def test_preregistered_tier_distribution_matches_table():
    # E-01: prose claimed 32 / 14 / 6. The table itself is 37 / 13 / 6.
    assert Counter(c.expected_tier for c in REG.cues.values()) == {"A": 37, "B": 13, "abstain": 6}


def test_every_expected_abstention_is_kinematic():
    # E-01: prose said "five of those six are KIN". All six are.
    assert {c.type for c in REG.cues.values() if c.expected_tier == "abstain"} == {"KIN"}


def test_lp04_is_the_only_kinematic_tier_a_candidate():
    # E-01: prose said every Tier A candidate is SET/TMP/ROM/CMP. LP-04 is KIN.
    odd = [c.id for c in REG.cues.values()
           if c.expected_tier == "A" and c.type not in {"SET", "TMP", "ROM", "CMP"}]
    assert odd == ["LP-04"]


def test_cross_references_are_consistent():
    for ex in REG.exercises.values():
        assert ex.stations.required in REG.stations
        for cid in ex.cues:
            assert REG.cues[cid].exercise == ex.id
    for cue in REG.cues.values():
        assert cue.station == ANY_STATION or cue.station in REG.stations


def test_implemented_detectors_match_registry_and_are_provisional():
    implemented = {c.id for c in REG.cues.values() if c.detector == "implemented"}
    assert implemented == set(DETECTORS)
    assert all(REG.cues[c].provisional for c in implemented)  # D-18: nothing is validated yet


def test_no_exercise_is_promoted_without_evidence():
    assert not any(p.validated for p in REG.tier_b.values())
    assert REG.tier_a_cues == {}


def test_rep_phases_are_anatomical_opposites():
    for ex in REG.exercises.values():
        assert ex.rep_phase.concentric != ex.rep_phase.eccentric


def _copy_registry(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    shutil.copytree(Path(__file__).resolve().parents[2] / "registry", root)
    return root


def test_unknown_station_breaks_loading(tmp_path):
    root = _copy_registry(tmp_path)
    data = yaml.safe_load((root / "cues.yaml").read_text(encoding="utf-8"))
    data["cues"][0]["station"] = "S-NOWHERE"
    (root / "cues.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown station"):
        load_registry(root)


def test_promotion_without_evidence_breaks_loading(tmp_path):
    root = _copy_registry(tmp_path)
    data = yaml.safe_load((root / "validation.yaml").read_text(encoding="utf-8"))
    data["tier_b"]["lat_pulldown"] = {"validated": True, "evidence": None}
    (root / "validation.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="cite committed evidence"):
        load_registry(root)


def test_registry_hash_ignores_line_endings(tmp_path):
    root = _copy_registry(tmp_path)
    for f in root.rglob("*.yaml"):
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert load_registry(root).content_hash == REG.content_hash
