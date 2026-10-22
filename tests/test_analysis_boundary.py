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


def test_a_start_outside_the_first_interval_is_refused(tmp_path, pair):
    _start, bdy = pair
    for valid in ("2022-06-13_22:45:00", "2022-06-14_00:00:00", "2022-06-14_00:15:00"):
        other = write_wrf_state(tmp_path / f"other-{valid[11:13]}{valid[14:16]}.nc", valid=valid)
        with pytest.raises(wrfbdy_update.BoundaryUpdateRefused, match="cropped to the record"):
            wrfbdy_update.rebase_first_record(bdy, other, tmp_path / "x")


def _stamp(dataset, name, record):
    return b"".join(bytes(c) for c in np.asarray(dataset[name][record])).decode()


def test_a_start_inside_the_first_interval_is_restamped_and_keeps_the_end_and_every_later_record(tmp_path, pair):
    """G8b: a cold start at 23:15 inside the 23:00-00:00 record.  The first record is the start's own
    coupling stamped 23:15 (the door's pair check passes), its tendency ends the interval where the
    boundary's own did, and every later record is the original's word for word."""
    _start, bdy = pair
    later = write_wrf_state(tmp_path / "later.nc", valid="2022-06-13_23:15:00", theta_offset=0.4, mu_offset=25.0)
    out = tmp_path / "wrfbdy.lead"
    receipt = wrfbdy_update.rebase_first_record(bdy, later, out)
    assert receipt["start_inside_interval"] == {"record_time": "2022-06-13_23:00:00",
                                                "start_time": "2022-06-13_23:15:00",
                                                "offset_seconds": 900.0, "lead_interval_seconds": 2700.0}
    _check_initial_boundary_pair(_restored(later), _tables(out), WIDTH)
    with netCDF4.Dataset(bdy) as old, netCDF4.Dataset(out) as new:
        assert _stamp(new, "Times", 0) == "2022-06-13_23:15:00"
        assert _stamp(new, "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_", 0) == "2022-06-13_23:15:00"
        for name in ("Times", "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_",
                     "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"):
            assert _stamp(new, name, 1) == _stamp(old, name, 1), name
        assert (_stamp(new, "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_", 0)
                == _stamp(old, "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_", 0) == "2022-06-14_00:00:00")
        for base in KEYS:
            for suffix in SIDES:
                b0 = np.asarray(old[f"{base}_B{suffix}"][0], np.float64)
                bt0 = np.asarray(old[f"{base}_BT{suffix}"][0], np.float64)
                b1 = np.asarray(new[f"{base}_B{suffix}"][0], np.float64)
                bt1 = np.asarray(new[f"{base}_BT{suffix}"][0], np.float64)
                # the record at 00:00 is reached from the start as the original reached it from 23:00
                np.testing.assert_allclose(b1 + 2700.0 * bt1, b0 + 3600.0 * bt0,
                                           rtol=2e-6, atol=1e-3, err_msg=f"{base}_B{suffix}")
                for name in (f"{base}_B{suffix}", f"{base}_BT{suffix}"):
                    assert np.array_equal(np.asarray(old[name][1]).view(np.uint32),
                                          np.asarray(new[name][1]).view(np.uint32)), name


def test_an_on_record_start_writes_the_same_bytes_as_before(tmp_path, pair):
    """The on-record path is untouched by the G8b branch: a rebase onto an analysis at the first
    record's own time is byte for byte what the rule has always written."""
    import hashlib

    _start, bdy = pair
    analysis = write_wrf_state(tmp_path / "analysis.nc", theta_offset=0.75, qv_scale=1.02, mu_offset=40.0)
    one, two = tmp_path / "one", tmp_path / "two"
    first = wrfbdy_update.rebase_first_record(bdy, analysis, one)
    second = wrfbdy_update.rebase_first_record(bdy, analysis, two)
    assert "start_inside_interval" not in first
    assert hashlib.sha256(one.read_bytes()).digest() == hashlib.sha256(two.read_bytes()).digest()
    with netCDF4.Dataset(bdy) as old, netCDF4.Dataset(one) as new:
        assert _stamp(new, "Times", 0) == _stamp(old, "Times", 0)


def test_the_lead_interval_is_read_as_a_start_inside_an_interval(tmp_path, pair):
    from datetime import datetime

    from gpuwm.ingest.wrfinput import lead_interval_seconds, wrfbdy_coverage

    _start, bdy = pair
    later = write_wrf_state(tmp_path / "later.nc", valid="2022-06-13_23:15:00", theta_offset=0.4)
    out = tmp_path / "wrfbdy.lead"
    wrfbdy_update.rebase_first_record(bdy, later, out)
    coverage = wrfbdy_coverage(out)
    assert coverage.forcing_interval_seconds == 3600
    assert coverage.lead_interval_seconds == 2700
    assert coverage.start == datetime(2022, 6, 13, 23, 15)
    assert coverage.end == datetime(2022, 6, 14, 1, 0)
    plain = wrfbdy_coverage(bdy)
    assert (plain.forcing_interval_seconds, plain.lead_interval_seconds) == (3600, None)
    t = [datetime(2022, 6, 13, 23, 15), datetime(2022, 6, 14, 0), datetime(2022, 6, 14, 1)]
    n = [datetime(2022, 6, 14, 0), datetime(2022, 6, 14, 1), datetime(2022, 6, 14, 2)]
    assert lead_interval_seconds(t, n) == (3600, 2700)
    assert lead_interval_seconds(t, n[-1:]) == (3600, 2700)
    with pytest.raises(ValueError, match="nextbdytime declares"):
        lead_interval_seconds(t, ())            # a short first interval nothing declares
    with pytest.raises(ValueError, match="uniform"):
        lead_interval_seconds(t + [datetime(2022, 6, 14, 3)], n + [datetime(2022, 6, 14, 4)])
    longer = [datetime(2022, 6, 13, 22, 0), datetime(2022, 6, 14, 0), datetime(2022, 6, 14, 1)]
    with pytest.raises(ValueError, match="uniform"):
        lead_interval_seconds(longer, [datetime(2022, 6, 14, 0)] + n[1:])


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
