"""Native driver cadence and first-actual-advance initialization controls."""
from pathlib import Path
import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case", range(1, 13))
@pytest.mark.parametrize("corpus", ["fixtures", "fixtures_nfmc7"])
def test_moisture_driver_matches_corrected_native_schedule(case, corpus):
    from tools.sfire_coupled_ideal.moisture_driver.grade import replay
    fixture_root = Path(__file__).parents[1] / "tools/sfire_coupled_ideal/moisture_driver" / corpus
    for step in replay(case, fixture_root):
        for name, grade in step.items():
            assert grade["different_words"] == 0, (case, name, grade)
