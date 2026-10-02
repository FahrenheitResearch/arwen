"""WRF input reservation reads its urban planes before taking a shared card."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import netcdf_bridge, supervisor
from gpuwm.core import preflight as pf
from gpuwm.wrfinput_forecast import wrfinput_urban_columns
from test_urban_workspace_price import _city_tree, _land_use, MODIS


class _Plane:
    def __init__(self, name, value, reads, dimensions=(
            "Time", "south_north", "west_east")):
        self.name, self.value, self.reads = name, value, reads
        self.dimensions = dimensions

    def __getitem__(self, key):
        self.reads.append(self.name)
        return self.value


def _run(exp):
    return SimpleNamespace(experiment=exp, metadata={
        dc.grid_id: SimpleNamespace(
            path=Path(f"wrfinput_d{dc.grid_id:02d}"),
            global_attributes={"MMINLU": MODIS})
        for dc in exp.domains})


def test_shared_reservation_uses_only_wrfinput_urban_land_cover(
        tmp_path, monkeypatch):
    exp = _city_tree(tmp_path)
    reads = []
    datasets = {}
    counts = {1: 500, 2: 750}
    for dc in exp.domains:
        shape = (dc.run.ny, dc.run.nx)
        variables = {
            "LU_INDEX": _Plane("LU_INDEX", _land_use(
                shape, counts[dc.grid_id])[None], reads),
            "FRC_URB2D": _Plane("FRC_URB2D", np.zeros(
                (1,) + shape, np.float32), reads),
            # The complete restore reads this; the reservation must not.
            "T": _Plane("T", None, reads),
        }
        datasets[f"wrfinput_d{dc.grid_id:02d}"] = SimpleNamespace(
            variables=variables)

    @contextmanager
    def opened(path):
        yield datasets[str(path)]

    monkeypatch.setattr(netcdf_bridge, "open_dataset", opened)
    actual = wrfinput_urban_columns(_run(exp))
    assert actual == counts
    assert reads == ["LU_INDEX", "FRC_URB2D"] * 2
    prepared = supervisor.priced_reservation_bytes(exp, urban_columns=actual)
    bound = supervisor.priced_reservation_bytes(exp)
    assert prepared == pf.admission_estimate(
        exp, urban_columns=counts).peak_envelope_bytes
    assert prepared < bound


def test_missing_land_cover_keeps_the_bound_and_plain_domains_read_nothing(
        tmp_path, monkeypatch):
    exp = _city_tree(tmp_path)
    run = _run(exp)
    run.metadata[1].global_attributes = {}
    opened_paths = []

    @contextmanager
    def opened(path):
        opened_paths.append(path)
        yield SimpleNamespace(variables={})

    monkeypatch.setattr(netcdf_bridge, "open_dataset", opened)
    assert wrfinput_urban_columns(run) is None
    assert opened_paths == [Path("wrfinput_d02")]
    plain = replace(exp, domains=tuple(replace(
        dc, run=replace(dc.run, sf_urban_physics=0)) for dc in exp.domains))
    opened_paths.clear()
    assert wrfinput_urban_columns(_run(plain)) is None
    assert opened_paths == []


@pytest.mark.parametrize("bad", ["shape", "time", "nonfinite"])
def test_supplied_urban_plane_is_checked_before_reservation(
        tmp_path, monkeypatch, bad):
    exp = _city_tree(tmp_path)
    run = _run(exp)
    shape = (exp.root.run.ny, exp.root.run.nx)
    value = _land_use(shape, 100)[None].astype(np.float32)
    if bad == "shape":
        value = value[:, :-1]
    elif bad == "time":
        value = np.repeat(value, 2, axis=0)
    else:
        value[0, 0, 0] = np.nan

    @contextmanager
    def opened(path):
        yield SimpleNamespace(variables={
            "LU_INDEX": _Plane("LU_INDEX", value, [])})

    monkeypatch.setattr(netcdf_bridge, "open_dataset", opened)
    reason = {"shape": "shape mismatch", "time": "exactly one Time",
              "nonfinite": "non-finite"}[bad]
    with pytest.raises(ValueError, match=reason):
        wrfinput_urban_columns(run)


def test_an_unknown_reservation_count_preserves_the_existing_price_call(
        tmp_path, monkeypatch):
    exp = _city_tree(tmp_path)
    calls = []

    def price(value, *, source=None):
        calls.append(value)
        return SimpleNamespace(peak_envelope_bytes=123)

    monkeypatch.setattr(pf, "admission_estimate", price)
    assert supervisor.priced_reservation_bytes(exp) == 123
    assert calls == [exp]
