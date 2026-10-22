"""wrf-analysis-bundle/1: the format, its refusals, and the producer's helpers, on real files.

Every NetCDF byte goes through the Rust decoder and the Rust classic writer, as on the door.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from analysis_bundle_fixtures import bundle_identity, write_wrf_state
from gpuwm.io import analysis_bundle as ab


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path, name: str) -> np.ndarray:
    with netCDF4.Dataset(path) as dataset:
        return np.asarray(dataset.variables[name][0])


def test_resident_increment_bundle_is_byte_identical(pair, tmp_path):
    background, analysis = pair
    inc = tmp_path / "source-increments.nc"
    ab.write_increment(analysis, background, inc, fields="all")
    arrays = ab.read_increment(inc)
    file_bundle, _ = ab.make_bundle(tmp_path / "from-file", background=background, increment=inc,
                                   iau_window_s=600.0, **bundle_identity())
    resident, _ = ab.make_bundle(tmp_path / "resident", background=background, increment_arrays=arrays,
                                iau_window_s=600.0, **bundle_identity())
    for name in ("analysis.nc", "increment.nc", "manifest.json"):
        assert _sha(tmp_path / "from-file" / name) == _sha(tmp_path / "resident" / name)


def test_resident_increment_refuses_broadcast_shape(pair, tmp_path):
    background, analysis = pair
    inc = tmp_path / "source-increments.nc"
    ab.write_increment(analysis, background, inc, fields="all")
    arrays = ab.read_increment(inc)
    arrays["T"] = arrays["T"][:1]
    with pytest.raises(ValueError, match="shape"):
        ab.make_bundle(tmp_path / "resident", background=background, increment_arrays=arrays,
                       iau_window_s=600.0, **bundle_identity())


@pytest.fixture
def pair(tmp_path):
    background = write_wrf_state(tmp_path / "background.nc")
    analysis = write_wrf_state(tmp_path / "analysis.nc", theta_offset=0.0)
    with netCDF4.Dataset(analysis, "r+") as dataset:
        t = dataset.variables["T"]
        values = np.asarray(t[0])
        values[1, 2, 3] += 1.5
        t[0] = values
        u = dataset.variables["U"]
        values = np.asarray(u[0])
        values[:, 3, 4] += 2.0
        u[0] = values
        q = dataset.variables["QRAIN"]
        values = np.asarray(q[0])
        values[2, 3, 3] = 2.0e-4
        q[0] = values
    return background, analysis


def _make(tmp_path, pair, **kwargs):
    background, analysis = pair
    options = dict(bundle_identity(), iau_window_s=600.0)
    options.update(kwargs)
    return ab.make_bundle(tmp_path / "bundle", background=background, analysis=analysis, **options)


def test_make_writes_a_bundle_the_reader_accepts(tmp_path, pair):
    bundle, record = _make(tmp_path, pair)
    assert bundle.manifest["schema"] == ab.SCHEMA
    assert bundle.valid_time == "2022-06-13_23:00:00"
    assert bundle.manifest["valid_time"] == "2022-06-13T23:00:00Z"
    assert bundle.manifest["grid"]["nx"] == 7 and bundle.manifest["grid"]["nz"] == 4
    assert bundle.manifest["iau"] == {"window_s": 600.0}
    assert sorted(bundle.files) == ["analysis.nc", "increment.nc"]
    names = sorted(ab.increment_variables(bundle.increment))
    # the core and vapour always, every other prognostic only where it differs
    assert names == sorted(set(ab.REQUIRED_INCREMENT) | {"QRAIN"})
    assert record["increment"]["exact_background_all"] is True
    again = ab.load_bundle(bundle.root / "manifest.json")
    assert again.manifest_sha256 == bundle.manifest_sha256
    assert ab.main(["verify", str(bundle.root)]) == 0


def test_the_background_comes_back_bit_for_bit(tmp_path, pair):
    background, _analysis = pair
    bundle, _record = _make(tmp_path, pair, ph="analysis")
    rebuilt = tmp_path / "rebuilt.nc"
    receipt = ab.write_background(bundle.analysis, bundle.increment, rebuilt)
    assert "T" in receipt["changed"] and not receipt["clipped_negative"]
    for name in ab.increment_variables(bundle.increment):
        a, b = _read(rebuilt, name), _read(background, name)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), name


def test_a_zero_increment_rebuilds_the_analysis_file_byte_for_byte(tmp_path):
    state = write_wrf_state(tmp_path / "state.nc")
    bundle, record = ab.make_bundle(tmp_path / "bundle", background=state, analysis=state,
                                    iau_window_s=600.0, fields="all", **bundle_identity())
    increments = ab.read_increment(bundle.increment)
    assert sorted(increments) == sorted(set(ab.REQUIRED_INCREMENT) | {"QRAIN"})
    assert all(not np.any(value) for value in increments.values())
    assert record["balance"]["max_abs_dphp_m2s2"] == 0.0
    rebuilt = tmp_path / "background.nc"
    ab.write_background(bundle.analysis, bundle.increment, rebuilt)
    assert _sha(rebuilt) == _sha(bundle.analysis) == _sha(state)


def test_an_increment_file_makes_the_analysis(tmp_path, pair):
    background, analysis = pair
    first, _ = _make(tmp_path, pair, ph="analysis")
    second, _ = ab.make_bundle(tmp_path / "from-increment", background=background,
                               increment=first.increment, iau_window_s=600.0, ph="analysis",
                               **bundle_identity())
    assert _sha(second.analysis) == _sha(first.analysis)


def test_an_increment_that_takes_moisture_below_zero_makes_a_non_negative_analysis(tmp_path):
    background = write_wrf_state(tmp_path / "background.nc")
    qr = _read(background, "QRAIN").astype(np.float64)
    increment = {name: np.zeros_like(_read(background, name), dtype=np.float64) for name in ab.REQUIRED_INCREMENT}
    delta = np.zeros_like(qr)
    delta[1, 2, 3] = -qr[1, 2, 3] - 4.0e-14       # an ensemble mean's rounding: just below zero
    increment["QRAIN"] = delta
    record = ab.write_analysis(background, increment, tmp_path / "analysis.nc")
    analysis = _read(tmp_path / "analysis.nc", "QRAIN")
    assert analysis.min() >= 0.0 and analysis[1, 2, 3] == 0.0
    assert record["clipped_negative"]["QRAIN"]["values"] == 1
    bundle, made = ab.make_bundle(tmp_path / "bundle", background=background, analysis=tmp_path / "analysis.nc",
                                  iau_window_s=600.0, ph="analysis", **bundle_identity())
    rebuilt = tmp_path / "rebuilt.nc"
    ab.write_background(bundle.analysis, bundle.increment, rebuilt)
    assert np.array_equal(_read(rebuilt, "QRAIN"), _read(background, "QRAIN"))


def test_ph_is_the_engines_and_moves_only_where_the_column_changed(tmp_path, pair):
    background, analysis = pair
    bundle, record = _make(tmp_path, pair)
    dph = _read(bundle.analysis, "PH").astype(np.float64) - _read(background, "PH")
    # T moved at (j=2, i=3) and rain at (j=3, i=3); U moves no column
    moved = np.argwhere(np.any(dph != 0.0, axis=0))
    assert sorted(map(tuple, moved)) == [(2, 3), (3, 3)]
    assert np.all(dph[0] == 0.0)
    assert record["balance"]["applied"] is True and record["balance"]["mass_moved"] is False
    # a warm layer thickens the column above it
    assert dph[2, 2, 3] > 0.0


def test_a_surface_pressure_increment_is_balanced_at_its_own_mass(tmp_path):
    background = write_wrf_state(tmp_path / "background.nc")
    analysis = write_wrf_state(tmp_path / "analysis.nc")
    with netCDF4.Dataset(analysis, "r+") as dataset:
        mu = np.asarray(dataset.variables["MU"][0])
        mu[1, 1] += 150.0
        dataset.variables["MU"][0] = mu
    bundle, record = ab.make_bundle(tmp_path / "bundle", background=background, analysis=analysis,
                                    iau_window_s=600.0, **bundle_identity())
    dph = _read(bundle.analysis, "PH").astype(np.float64) - _read(background, "PH")
    assert record["balance"]["mass_moved"] is True
    assert sorted(map(tuple, np.argwhere(np.any(dph != 0.0, axis=0)))) == [(1, 1)]
    assert np.all(np.isfinite(dph)) and np.all(dph[0] == 0.0)
    assert ab.read_increment(bundle.increment)["MU"][1, 1] == pytest.approx(150.0)


def _wrfout_without_increment(manifest):
    manifest["analysis"]["form"] = "wrfout"
    manifest["files"].pop("increment.nc")
    manifest.pop("iau")


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m.update(extra=1), "does not know"),
    (lambda m: m["grid"].update(id="s3://bucket/grid"), "path or address"),
    (lambda m: m["static"].update(sha256="ABC"), "SHA-256"),
    (lambda m: m["analysis"].update(kind="best"), "analysis.kind"),
    (lambda m: m.pop("iau"), "go together"),
    (lambda m: m.update(forecast={"args": ["--outdir", "x"]}), "never a host binding"),
    (lambda m: _wrfout_without_increment(m), "cycling restart"),
    (lambda m: m["physics"].update(mode="fast"), "physics.mode"),
])
def test_manifest_refusals(tmp_path, pair, mutate, message):
    bundle, _ = _make(tmp_path, pair)
    manifest = json.loads((bundle.root / "manifest.json").read_text())
    mutate(manifest)
    with pytest.raises(ab.BundleRefusal, match=message):
        ab.validate_manifest(manifest)


def test_a_changed_file_and_an_unlisted_file_are_refused(tmp_path, pair):
    bundle, _ = _make(tmp_path, pair)
    (bundle.root / "noda.nc").write_bytes(b"not listed")
    with pytest.raises(ab.BundleRefusal, match="not listed"):
        ab.load_bundle(bundle.root)
    (bundle.root / "noda.nc").unlink()
    with open(bundle.increment, "r+b") as stream:
        stream.seek(-8, 2)
        tail = stream.read(8)
        stream.seek(-8, 2)
        stream.write(bytes(byte ^ 0xFF for byte in tail))
    with pytest.raises(ab.BundleRefusal, match="SHA-256"):
        ab.load_bundle(bundle.root)


@pytest.mark.parametrize("other, message", [
    (dict(xlat_offset=0.5), "horizontal_sha256"),
    (dict(znw_top=0.01), "vertical coordinate"),
    (dict(hgt_offset=25.0), "static fields"),
])
def test_a_run_on_another_grid_is_refused(tmp_path, pair, other, message):
    bundle, _ = _make(tmp_path, pair)
    run = write_wrf_state(tmp_path / "run.nc", **other)
    with pytest.raises(ab.BundleRefusal, match=message):
        ab.check_run(bundle, run)
    assert ab.check_run(bundle, pair[0])["static_sha256"] == bundle.manifest["static"]["sha256"]


def test_a_netcdf4_file_is_refused_by_name(tmp_path, pair):
    background, _ = pair
    analysis = write_wrf_state(tmp_path / "nc4.nc", fmt="NETCDF4")
    with pytest.raises(ab.BundleRefusal, match="classic"):
        ab.make_bundle(tmp_path / "bundle", background=background, analysis=analysis,
                       iau_window_s=600.0, **bundle_identity())


def test_an_increment_of_a_field_the_forecast_never_integrates_is_refused(tmp_path, pair):
    bundle, _ = _make(tmp_path, pair)
    odd = tmp_path / "odd-increment.nc"
    with netCDF4.Dataset(bundle.increment) as source, netCDF4.Dataset(odd, "w", format="NETCDF3_64BIT_OFFSET") as out:
        for name, dim in source.dimensions.items():
            out.createDimension(name, None if dim.isunlimited() else len(dim))
        for name, variable in source.variables.items():
            copy = out.createVariable(name, variable.dtype, variable.dimensions)
            copy[...] = variable[...]
        extra = out.createVariable("TSK", "f8", ("Time", "south_north", "west_east"))
        extra[0] = np.zeros((6, 7))
    with pytest.raises(ab.BundleRefusal, match="TSK"):
        ab.increment_variables(odd)


def test_bundle_prefetch_keeps_unmasked_scaled_values(tmp_path, monkeypatch):
    """Reading prefetched bundle fields must not restart one decoder per field."""
    from gpuwm import netcdf_bridge
    path = write_wrf_state(tmp_path / 'state.nc')
    with netCDF4.Dataset(path, 'r+') as ds:
        ds['T'].missing_value = np.float32(-999.0)
        ds['T'][0, 0, 0, 0] = -999.0
    expected = netcdf_bridge.open_dataset(path)
    expected.set_auto_mask(False)
    names = ['T', 'U', 'QVAPOR']
    reference = {name: np.asarray(expected.variables[name][...]) for name in names}
    runs = []
    original = netcdf_bridge._run
    def counted(arguments, **kwargs):
        runs.append(arguments[1])
        return original(arguments, **kwargs)
    monkeypatch.setattr(netcdf_bridge, '_run', counted)
    actual = ab._open(path)
    runs.clear()
    # 2.9.4 batches through _prefetch_unmasked (the unmasked policy is set per variable before the batch).
    ab._prefetch_unmasked(actual, names)
    for name in names:
        np.testing.assert_array_equal(ab._first_record(actual.variables[name]), reference[name][0])
    assert runs == ['dump']
    assert ab._first_record(actual.variables['T'])[0, 0, 0] == -999.0
