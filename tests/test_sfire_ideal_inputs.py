"""Native initializer control geometry and harness allocation contracts."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest


def test_native_allocated_classes_changes_only_extent(tmp_path):
    from tools.sfire_coupled_ideal.moisture_driver.allocated_classes import emit
    source = Path(__file__).parents[1] / "tools/sfire_coupled_ideal/moisture_driver/run.F90"
    output = tmp_path / "run.F90"
    receipt = emit(source, output, 7)
    assert output.read_text().replace("nc=7", "nc=5") == source.read_text()
    assert receipt["allocated_classes"] == 7
    assert receipt["active_class_limit"] == 5
    with pytest.raises(ValueError, match="five active"):
        emit(source, output, 4)


def test_native_history_corpus_keeps_original_geometry_defects_separate():
    from tools.sfire_coupled_ideal.initialization.grade_geometry import load
    import numpy as np
    root = Path(__file__).parents[1] / "tools/sfire_coupled_ideal/initialization/fixtures"
    for name in ("soil_3", "soil_5"):
        f = load(root, "ideal_geometry/" + name)
        assert np.count_nonzero(f["original_dzs"] != f["dzs"]) == 1
        assert f["original_dzs"][1] > f["dzs"][1]
    f = load(root, "ideal_geometry/interpolation_edges")
    assert np.count_nonzero(f["original_zero"] != f["original_sentinel"]) > 0
    assert np.all(f["corrected"] != -999.)
