"""Whole native landuse_init controls, including retained lookup state."""
from pathlib import Path
import json
import pytest
from conftest import requires_gpu

ROOT=Path(__file__).parents[1]/"tools/sfire_coupled_ideal/initialization/landuse_fixtures"
CASES=tuple(json.loads((ROOT/"receipt.json").read_text())["cases"])


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case",CASES)
def test_complete_native_landuse_words(case):
    from tools.sfire_coupled_ideal.initialization.grade_landuse import replay
    for name,grade in replay(ROOT,case).items():
        assert grade["different_words"]==0,(case,name,grade)
