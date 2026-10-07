"""The actual open geopotential path matches unmodified compiled WRF words."""
from pathlib import Path
import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_native_open_geopotential_full_control_corpus(tmp_path):
    from tools.sfire_coupled_ideal.rhs_ph_open.grade import grade
    root = Path(__file__).parents[1] / "tools/sfire_coupled_ideal/rhs_ph_open/fixtures"
    grade(root, tmp_path / "open-geopotential.json")
