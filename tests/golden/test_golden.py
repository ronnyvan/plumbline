import json

import pytest

from eval.harness.run import _diff
from tests.golden.cases import CASES, GOLDEN_DIR, build


@pytest.mark.parametrize("name", sorted(CASES))
def test_report_matches_golden(name):
    path = GOLDEN_DIR / f"{name}.json"
    assert path.exists(), f"missing golden file; run `make golden` and review {path.name}"
    problems = _diff(json.loads(path.read_text(encoding="utf-8")), build(name))
    assert not problems, "report changed:\n" + "\n".join(problems[:25])
