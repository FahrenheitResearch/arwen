"""Compiled original initialization blocks and labeled geometry corrections."""
from pathlib import Path
import json
import pytest
from conftest import requires_gpu

ROOT = Path(__file__).parents[1] / "tools/sfire_coupled_ideal/initialization/fixtures"
CASES = tuple(json.loads((ROOT / "receipt.json").read_text())["cases"])


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES)
def test_native_ideal_geometry_words(case):
    from tools.sfire_coupled_ideal.initialization.grade_geometry import replay
    for name, grade in replay(ROOT, case).items():
        assert grade["different_words"] == 0, (case, name, grade)
