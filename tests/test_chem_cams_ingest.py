"""CAMS carried onto the model grid: decode, convert, blend, remap, fill.

The fixture is a CAMS-shaped GRIB2 pair written by ecCodes 2.48 -- ECMWF's own
encoder -- for exactly the parameters the ``cams-global`` source row selects
(``tests/data/chem_cams/``, built by
``tools/chem_cams_fixtures/make_synthetic_cams.py``).  The VALUES are
synthetic plausible profiles; the BYTES are ECMWF's encoding: the WMO
chemical template 4.40 for the gases, ECMWF-local 192/210 records for the
aerosols, the L137 coefficients in every Section 4.

What is asserted:

* every stage runs on the Rust operators (a library without them is a
  refusal, never a NumPy stand-in);
* the chain equals a float64 NumPy transcription of the same stages
  (conversion, time blend, source pressure, bilinear, WRF real's vertical
  interpolation from :func:`gpuwm.verify.npref.np_wrf_real_vert_interp`) to
  float32 rounding;
* the physics the rows declare: ppmv from kg/kg per moist air with
  ``q/(1-q)`` and ``mwdry/M``; a field constant in the vertical stays
  constant; the time blend weights are the valid time's;
* initialization finds the files through the content-addressed cache and
  refuses, naming the fetch, when nothing covers the frame or the area.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import chem_table
from gpuwm.chem_boundary_remap import (
    MWDRY_G_MOL, BoundaryRemapError, TargetGrid, boundary_fields_for_grid,
    boundary_rows, decode_source_files,
)

DATA = Path(__file__).resolve().parent / "data" / "chem_cams"
CYCLE = datetime(2025, 7, 30, 0, tzinfo=timezone.utc)


def _backend():
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    try:
        backend = CpuPreprocessBackend()
    except Exception as error:  # the library is not built on this box
        pytest.skip(f"CPU preprocessing bridge unavailable: {error}")
    for name in ("weighted_combination", "hybrid_full_pressure"):
        if not callable(getattr(backend, name, None)):
            pytest.skip(f"bridge predates {name}")
    return backend


def _fixture_files():
    ml = sorted(DATA.glob("SYNTHETIC-cams-ml-*.grib2"))
    sl = sorted(DATA.glob("SYNTHETIC-cams-sl-*.grib2"))
    if not ml or not sl:
        pytest.skip("synthetic CAMS fixture absent")
    return ml[0], sl[0]


def _table():
    return chem_table.load_sets(["cams_aq"], ["cams-global"])


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    _backend()
    try:
        from gpuwm import grib2_stack  # noqa: F401
    except ImportError:
        pytest.skip("grib2_stack front absent")
    table = _table()
    source = table.sources["cams-global"]
    from gpuwm.chem_boundary_remap import _needed_variables
    variables = _needed_variables(source, boundary_rows(table, source.name))
    ml, sl = _fixture_files()
    work = tmp_path_factory.mktemp("stack")
    return decode_source_files(source, {"model_level": ml, "single_level": sl},
                               variables, work)


def _target(frames, nz=12):
    lat_axis = next(iter(next(iter(frames.frames.values())).values())).latitude
    lon_axis = next(iter(next(iter(frames.frames.values())).values())).longitude
    lat = np.linspace(min(lat_axis) + 0.1, max(lat_axis) - 0.1, 3)
    lon = np.linspace(min(lon_axis) + 0.1, max(lon_axis) - 0.1, 4)
    lon2, lat2 = np.meshgrid(lon, lat)
    # A plausible total-pressure column, bottom-up, 1000 hPa to 150 hPa.
    column = np.linspace(99000.0, 15000.0, nz)
    pressure = np.broadcast_to(column[:, None, None], (nz, *lat2.shape)).copy()
    return TargetGrid(lat2, lon2, pressure.astype(np.float32))


def test_the_fixture_decodes_every_selected_variable_at_both_frames(frames):
    table = _table()
    source = table.sources["cams-global"]
    assert frames.times == (CYCLE, CYCLE + timedelta(hours=3))
    for moment in frames.times:
        frame = frames.frames[moment]
        for row in boundary_rows(table, source.name):
            for key in row.boundary[0]["fields"]:
                assert frame[key].array.shape[0] == 137, key
        assert len(frame["specific_humidity"].coordinate_values) == 276


def test_the_chain_equals_a_float64_transcription(frames):
    from gpuwm.verify.npref import np_wrf_real_vert_interp
    backend = _backend()
    table = _table()
    source = table.sources["cams-global"]
    target = _target(frames)
    when = CYCLE + timedelta(hours=1)
    fields, receipt = boundary_fields_for_grid(table, source, frames, target,
                                               when, backend=backend)
    assert receipt["frames"] == [CYCLE.isoformat(),
                                 (CYCLE + timedelta(hours=3)).isoformat()]
    assert receipt["weights"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    f0 = frames.frames[CYCLE]
    f1 = frames.frames[CYCLE + timedelta(hours=3)]
    coords = f0["specific_humidity"].coordinate_values
    a = np.asarray(coords[:138], dtype=np.float64)
    b = np.asarray(coords[138:], dtype=np.float64)
    lat_axis = f0["ozone"].latitude
    lon_axis = f0["ozone"].longitude

    def bilinear(stack):
        stack = np.asarray(stack, dtype=np.float64)
        y = (target.latitude - lat_axis[0]) / (lat_axis[1] - lat_axis[0])
        x = (target.longitude - lon_axis[0]) / (lon_axis[1] - lon_axis[0])
        iy = np.minimum(np.floor(y).astype(int), lat_axis.size - 2)
        ix = np.minimum(np.floor(x).astype(int), lon_axis.size - 2)
        fy, fx = y - iy, x - ix
        s = stack.reshape(-1, lat_axis.size, lon_axis.size)
        out = ((1 - fy) * ((1 - fx) * s[:, iy, ix] + fx * s[:, iy, ix + 1])
               + fy * ((1 - fx) * s[:, iy + 1, ix] + fx * s[:, iy + 1, ix + 1]))
        return out.reshape(stack.shape[:-2] + target.latitude.shape)

    ps = (2 / 3) * np.asarray(f0["surface_pressure"].array[0], np.float64) + \
        (1 / 3) * np.asarray(f1["surface_pressure"].array[0], np.float64)
    ph = a[:, None, None] + b[:, None, None] * ps[None]
    pf = 0.5 * (ph[:-1] + ph[1:])
    for row in boundary_rows(table, source.name):
        entry = row.boundary[0]
        (key,) = entry["fields"]
        scale = MWDRY_G_MOL / row.molar_mass_g_mol * 1e6
        conv = [scale * np.asarray(f[key].array, np.float64)
                / (1 - np.asarray(f["specific_humidity"].array, np.float64))
                for f in (f0, f1)]
        src = (2 / 3) * conv[0] + (1 / 3) * conv[1]
        field_h = bilinear(src)
        # The reference takes the column bottom-up (descending pressure);
        # model level 137 is the lowest, so the stacks are reversed.
        expected = np_wrf_real_vert_interp(
            field_h[::-1], field_h[-1], bilinear(pf)[::-1], bilinear(ps),
            np.asarray(target.pressure, np.float64), interp_in_logp=True,
            extrap="constant", force_sfc_in_vinterp=1,
            zap_close_levels=500.0, vboundb=target.pressure.shape[0])
        got = np.asarray(fields[row.name], np.float64)
        assert got.shape == target.pressure.shape
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=0,
                                   err_msg=row.name)


def test_a_vertically_constant_field_stays_constant(frames):
    backend = _backend()
    table = _table()
    source = table.sources["cams-global"]
    target = _target(frames)
    constant = {}
    for moment, frame in frames.frames.items():
        slot = dict(frame)
        for key in ("ozone", "nitrogen_dioxide", "carbon_monoxide",
                    "sulphur_dioxide"):
            like = slot[key]
            values = np.full(like.array.shape, 1.0e-8, np.float32)
            slot[key] = SimpleNamespace(
                key=key, valid_time=like.valid_time, levels=like.levels,
                latitude=like.latitude, longitude=like.longitude,
                coordinate_values=like.coordinate_values, array=values)
        constant[moment] = slot
    from gpuwm.chem_boundary_remap import SourceFrames
    fields, _ = boundary_fields_for_grid(
        table, source, SourceFrames(source.name, constant), target,
        CYCLE, backend=backend)
    # 1e-8 kg/kg of ozone per moist air, as ppmv of dry air, at every level:
    # the only variation left is the dry-air divisor 1/(1-q).
    q = np.asarray(frames.frames[CYCLE]["specific_humidity"].array, np.float64)
    lo = 1e-8 * MWDRY_G_MOL / 47.997 * 1e6 / (1 - q.min())
    hi = 1e-8 * MWDRY_G_MOL / 47.997 * 1e6 / (1 - q.max())
    got = np.asarray(fields["o3"], np.float64)
    assert np.all(got >= lo * (1 - 1e-6)) and np.all(got <= hi * (1 + 1e-6))


def test_a_valid_time_outside_the_frames_is_refused(frames):
    backend = _backend()
    table = _table()
    with pytest.raises(BoundaryRemapError, match="never extrapolated"):
        boundary_fields_for_grid(table, table.sources["cams-global"], frames,
                                 _target(frames), CYCLE + timedelta(hours=4),
                                 backend=backend)


def _imported_cache(tmp_path, area):
    from gpuwm import data_store_fetch as dsf
    ml, sl = _fixture_files()
    source = chem_table.catalog().sources["cams-global"]
    plan = dsf.resolve_acquisition(
        source, area=area, start=CYCLE, end=CYCLE + timedelta(hours=1),
        now=CYCLE + timedelta(days=30))
    groups = {r.group: r.sha256 for r in plan.requests}
    root = tmp_path / "cache"
    dsf.fetch(plan, cache_root=root, progress=lambda *_: None,
              files={groups["model_level"]: ml, groups["single_level"]: sl},
              files_note="SYNTHETIC test fixture")
    return root


def test_initialization_fills_a_state_from_the_cache_and_names_its_files(
        frames, tmp_path):
    from gpuwm.chem_source_init import fill_boundary_sources
    manifest = json.loads(next(DATA.glob("SYNTHETIC-cams-*.manifest.json")).read_text())
    north, west, south, east = manifest["area_nwse"]
    target = _target(frames)
    area = f"{south + 0.05},{west + 0.05},{north - 0.05},{east - 0.05}"
    root = _imported_cache(tmp_path, area)
    table = _table()
    nz, ny, nx = target.pressure.shape
    state = SimpleNamespace(chem=SimpleNamespace(table=table))
    for row in table.rows:
        setattr(state, row.state_attr, np.zeros((nz, ny, nx), np.float32))
    cfg = SimpleNamespace(chem_sources="cams-global")
    receipts = fill_boundary_sources(
        state, cfg, valid_time=CYCLE + timedelta(hours=1),
        latlon=(target.latitude, target.longitude), pressure=target.pressure,
        cache_root=root, backend=_backend())
    assert len(receipts) == 1 and len(receipts[0]["files"]) == 2
    for row in table.rows:
        field = getattr(state, row.state_attr)
        assert np.all(np.isfinite(field)) and np.all(field > 0), row.name
    # The frame records the rows its sealed lateral forcing carries.
    assert state._external_scalar_boundary_fields == tuple(
        row.state_attr for row in table.transported)


def test_initialization_refuses_without_a_covering_file(frames, tmp_path):
    from gpuwm.chem_source_init import ChemSourceUnavailable, fill_boundary_sources
    table = _table()
    target = _target(frames)
    state = SimpleNamespace(chem=SimpleNamespace(table=table))
    for row in table.rows:
        setattr(state, row.state_attr,
                np.zeros(target.pressure.shape, np.float32))
    cfg = SimpleNamespace(chem_sources="cams-global")
    with pytest.raises(ChemSourceUnavailable,
                       match="python -m gpuwm.data_store_fetch fetch"):
        fill_boundary_sources(
            state, cfg, valid_time=CYCLE, latlon=(target.latitude,
                                                  target.longitude),
            pressure=target.pressure, cache_root=tmp_path / "empty",
            backend=_backend())


def test_a_chem_off_state_is_untouched():
    from gpuwm.chem_source_init import fill_boundary_sources
    state = SimpleNamespace()
    assert fill_boundary_sources(state, SimpleNamespace(), valid_time=None,
                                 latlon=None, pressure=None) == []
    assert vars(state) == {}
