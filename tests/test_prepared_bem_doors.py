"""Prepared runners read urban columns before their automatic admission."""

from dataclasses import replace

import pytest

from gpuwm.core import preflight as pf, streaming as st
from test_urban_workspace_price import _city_tree, _land_use, MODIS
from test_resident_admission import (
    _ConstructorReached, _stand_in_card, _stub_prepared_single_door,
    _tree_door)


@pytest.mark.parametrize("automatic", [False, True])
def test_single_prepared_door_admits_the_actual_workspace_before_restore(
        tmp_path, monkeypatch, automatic):
    exp = _city_tree(tmp_path)
    exp = replace(exp, domains=(exp.root,))
    if automatic:
        exp = replace(exp, tiles=st.StreamingOptions(
            mode="auto", resident_context=st.ResidentAdmissionContext(exp)))
    count = 500
    prepared = pf.estimate_experiment(exp, forcing_intervals=1,
                                      urban_columns={1: count})
    bound = pf.estimate_experiment(exp, forcing_intervals=1)
    _stand_in_card(monkeypatch,
        (prepared.peak_envelope_bytes + bound.peak_envelope_bytes) // 2
        + pf.EXTERNAL_MARGIN_BYTES)
    door, inputs = _stub_prepared_single_door(
        monkeypatch, tmp_path, exp, intervals=1, source="gfs")
    inputs.static = {"LU_INDEX": _land_use((216, 216), count)}
    inputs.landuse_identity = {"MMINLU": MODIS}
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(_ConstructorReached):
        door.run_prepared_forecast(inputs, output_directory=output)


@pytest.mark.parametrize("automatic", [False, True])
def test_tree_prepared_door_admits_the_actual_workspace_before_allocation(
        tmp_path, monkeypatch, automatic):
    exp = _city_tree(tmp_path)
    if automatic:
        exp = replace(exp, tiles=st.StreamingOptions(
            mode="auto", resident_context=st.ResidentAdmissionContext(exp)))
    counts = {1: 500, 2: 500}
    prepared = pf.estimate_experiment(exp, forcing_intervals=6,
                                      urban_columns=counts)
    bound = pf.estimate_experiment(exp, forcing_intervals=6)
    _stand_in_card(monkeypatch,
        (prepared.peak_envelope_bytes + bound.peak_envelope_bytes) // 2
        + pf.EXTERNAL_MARGIN_BYTES)
    door, inputs = _tree_door(monkeypatch, tmp_path, exp)
    for bundle, dc in zip(inputs.domains, exp.domains):
        bundle.static_fields = {"LU_INDEX": _land_use(
            (dc.run.ny, dc.run.nx), counts[dc.grid_id])}
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(_ConstructorReached):
        door.run_prepared_tree(inputs, output_directory=output,
                               io_mode="none")
