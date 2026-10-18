"""The boundary after an analysis replaces the start: WRFDA's da_update_bc on real files."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from analysis_bundle_fixtures import raw_fields, write_wrf_state, write_wrfbdy
from gpuwm.da import analysis_boundary as wrfbdy_update  # noqa: N813
from gpuwm.ingest.wrfinput import _check_initial_boundary_pair, _wrfbdy_side_table

WIDTH = 2
SIDES = {"XS": "west", "XE": "east", "YS": "south", "YE": "north"}
KEYS = {"U": "u", "V": "v", "T": "theta", "PH": "phi", "MU": "mu", "QVAPOR": "qv"}


def _restored(path: Path):
    return SimpleNamespace(raw=raw_fields(path), global_attributes={"USE_THETA_M": 0}, path=Path(path))


def _tables(path: Path, record: int = 0):
    tables = {key: {} for key in KEYS.values()}
    with netCDF4.Dataset(path) as dataset:
        for base, key in KEYS.items():
            for suffix, side in SIDES.items():
                tables[key][side] = (_wrfbdy_side_table(dataset[f"{base}_B{suffix}"], record, side),
                                     _wrfbdy_side_table(dataset[f"{base}_BT{suffix}"], record, side))
    return tables


@pytest.fixture
def pair(tmp_path):
    start = write_wrf_state(tmp_path / "wrfinput_d01")
    return start, write_wrfbdy(tmp_path / "wrfbdy_d01", start, width=WIDTH)


def test_the_fixture_is_a_pair_the_door_accepts(pair):
    start, bdy = pair
    _check_initial_boundary_pair(_restored(start), _tables(bdy), WIDTH)


def test_an_analysed_start_is_refused_against_the_old_boundary_and_accepted_after_the_rebase(tmp_path, pair):
    _start, bdy = pair
    analysis = write_wrf_state(tmp_path / "analysis.nc", theta_offset=0.75, qv_scale=1.02, mu_offset=40.0)
    with pytest.raises(ValueError, match="does not match initial"):
        _check_initial_boundary_pair(_restored(analysis), _tables(bdy), WIDTH)
    out = tmp_path / "wrfbdy.rebased"
    receipt = wrfbdy_update.rebase_first_record(bdy, analysis, out)
    _check_initial_boundary_pair(_restored(analysis), _tables(out), WIDTH)
    assert receipt["first_interval_seconds"] == 3600.0
    assert receipt["max_abs_first_record_change"]["T"] > 0.0
    with netCDF4.Dataset(bdy) as old, netCDF4.Dataset(out) as new:
        for base in KEYS:
            for suffix in SIDES:
                b0 = np.asarray(old[f"{base}_B{suffix}"][0], np.float64)
                bt0 = np.asarray(old[f"{base}_BT{suffix}"][0], np.float64)
                b1 = np.asarray(new[f"{base}_B{suffix}"][0], np.float64)
                bt1 = np.asarray(new[f"{base}_BT{suffix}"][0], np.float64)
                # the end of the first interval is the boundary's own (da_update_bc)
                np.testing.assert_allclose(b1 + 3600.0 * bt1, b0 + 3600.0 * bt0,
                                           rtol=2e-6, atol=1e-3, err_msg=f"{base}_B{suffix}")
                # every later record is the original's, word for word
                for name in (f"{base}_B{suffix}", f"{base}_BT{suffix}"):
                    assert np.array_equal(np.asarray(old[name][1]).view(np.uint32),
                                          np.asarray(new[name][1]).view(np.uint32)), name


def test_rebasing_onto_the_boundarys_own_start_moves_no_first_record_value(tmp_path, pair):
    start, bdy = pair
    receipt = wrfbdy_update.rebase_first_record(bdy, start, tmp_path / "same.rebased")
    assert all(value == 0.0 for value in receipt["max_abs_first_record_change"].values())
    assert wrfbdy_update.same_start(start, start)
    assert not wrfbdy_update.same_start(start, write_wrf_state(tmp_path / "other.nc", theta_offset=0.1))


def test_a_start_at_another_time_is_refused(tmp_path, pair):
    _start, bdy = pair
    later = write_wrf_state(tmp_path / "later.nc", valid="2022-06-13_23:15:00")
    with pytest.raises(wrfbdy_update.BoundaryUpdateRefused, match="G8b"):
        wrfbdy_update.rebase_first_record(bdy, later, tmp_path / "x")


def test_a_table_with_no_coupling_rule_is_kept_as_the_boundarys_own(tmp_path, pair):
    start, bdy = pair
    odd = tmp_path / "odd"
    with netCDF4.Dataset(bdy) as source, netCDF4.Dataset(odd, "w", format="NETCDF3_64BIT_OFFSET") as out:
        for name, dim in source.dimensions.items():
            out.createDimension(name, None if dim.isunlimited() else len(dim))
        for name, variable in source.variables.items():
            out.createVariable(name, variable.dtype, variable.dimensions)[...] = variable[...]
        for suffix in SIDES:
            template = source["T_B" + suffix]
            out.createVariable(f"QKE_B{suffix}", "f4", template.dimensions)[...] = template[...]
        out.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
    analysis = write_wrf_state(tmp_path / "analysis.nc", theta_offset=0.5)
    receipt = wrfbdy_update.rebase_first_record(odd, analysis, tmp_path / "y")
    assert "QKE" in receipt["kept"] and "QKE" not in receipt["fields"]
    with netCDF4.Dataset(odd) as old, netCDF4.Dataset(tmp_path / "y") as new:
        for suffix in SIDES:
            name = f"QKE_B{suffix}"
            assert np.array_equal(np.asarray(old[name][...]), np.asarray(new[name][...])), name
