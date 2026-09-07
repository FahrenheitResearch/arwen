"""Actual overlay arithmetic, lazy source lifetime and prepared policy controls."""
from dataclasses import replace
from datetime import datetime, timedelta
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.hrrr import HrrrNativeSnapshot, hrrr_source_grid
from gpuwm.ingest.horiz import HorizontalSnapshot
from gpuwm.ingest.source_metadata import SourceSnapshotMetadata, snapshot_metadata
from gpuwm.ingest.water_overlay import (WaterOverlayError, apply_water_temperature_overlay,
    load_bound_water_overlay, overlay_snapshot_sequence, verify_overlay_sequence)
from gpuwm.ingest.water_temperature import (WaterTemperatureStatics,
    assemble_horizontal_water_temperature)
from test_water_overlay import make_snapshot, write_fine_overlay, write_overlay, _analytic


def test_native_cropped_projection_overlay_matches_geographic_analysis(tmp_path):
    # An offset native crop has no fake degree axes. Its adapter supplies the
    # same reversible source geometry used by horizontal interpolation.
    source = HrrrNativeSnapshot(datetime(2026, 7, 20), 0, 710, 415, 5, 6,
        {"LANDSEA": np.zeros((5, 6)), "SKINTEMP": np.full((5, 6), 275.),
         "TT": np.arange(60.).reshape(2, 5, 6)})
    source.fields["LANDSEA"][0, :] = 1.
    rows, cols = np.indices((5, 6))
    lat, lon = hrrr_source_grid().ij_to_latlon(cols + 711., rows + 416.)
    yy = np.linspace(lat.min()-.2, lat.max()+.2, 11)
    xx = np.linspace(lon.min()-.2, lon.max()+.2, 13)
    path = write_overlay(tmp_path / "native-water.nc", yy, xx,
                         _analytic(yy[:, None], xx[None, :]))
    overlay, binding = load_bound_water_overlay(path)
    updated, receipt = apply_water_temperature_overlay(source, overlay)
    assert type(updated) is HrrrNativeSnapshot
    assert (updated.i_start, updated.j_start) == (710, 415)
    assert updated.fields["TT"] is source.fields["TT"]
    assert updated.fields["LANDSEA"] is source.fields["LANDSEA"]
    np.testing.assert_array_equal(updated.fields["SKINTEMP"][0], 275.)
    np.testing.assert_allclose(updated.fields["SKINTEMP"][1:], _analytic(lat, lon)[1:], rtol=0, atol=1e-12)
    assert receipt["replaced_cells"] == 24
    assert binding["bytes"] == path.stat().st_size


def test_lazy_overlay_metadata_does_not_materialize_or_retain_weather(tmp_path):
    path = write_fine_overlay(tmp_path / "water.nc")
    overlay, binding = load_bound_water_overlay(path)
    sample = make_snapshot()
    seen = []
    refs = []

    class Lazy:
        valid_times = tuple(sample.valid_time + timedelta(hours=i) for i in range(3))
        def __len__(self): return 3
        def snapshot_metadata(self, index):
            return SourceSnapshotMetadata(Era5Snapshot, sample.latitude, sample.longitude)
        def __getitem__(self, index):
            if not 0 <= index < 3: raise IndexError(index)
            seen.append(index)
            result = make_snapshot(self.valid_times[index])
            refs.append(weakref.ref(result.fields["TT"]))
            return result

    raw = Lazy()
    assert overlay_snapshot_sequence(raw, None) is raw
    seq = overlay_snapshot_sequence(raw, overlay, binding=binding)
    assert seq.valid_times == raw.valid_times
    assert snapshot_metadata(seq, 0).snapshot_type is Era5Snapshot
    assert seen == []
    with pytest.raises(WaterOverlayError, match="every prepared forcing time"):
        verify_overlay_sequence(seq)
    for index in (2, 0, 1):
        current = seq[index]
        assert seen[-1] == index
        assert len(seq._receipts) <= 3
        del current
        gc.collect()
        assert all(ref() is None for ref in refs)
    receipt = verify_overlay_sequence(seq)
    assert receipt["snapshots"] == 3
    assert all(row["replaced_cells"] == 35 for row in receipt["per_snapshot"])
    # Same-size mutation preserves paths and metadata but must not reuse a seal.
    before = path.read_bytes()
    path.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
    with pytest.raises(WaterOverlayError, match="bytes changed"):
        verify_overlay_sequence(seq)


def test_overlay_changed_after_decode_is_refused_before_forcing(tmp_path):
    path = write_fine_overlay(tmp_path / "water.nc")
    overlay, binding = load_bound_water_overlay(path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(WaterOverlayError, match="changed after it was loaded"):
        overlay_snapshot_sequence((make_snapshot(),), overlay, binding=binding)


def test_common_water_assembly_honors_selected_policy_without_changing_meteorology():
    skin = np.full((4, 5), 285.)
    sst = np.full((4, 5), 294.)
    sst[:, 0] = 0.
    met = HorizontalSnapshot(datetime(2026, 7, 20), np.array([1000.]),
                             {"SKINTEMP": skin, "SST": sst, "TT": skin[None]})
    def assemble(policy):
        statics = WaterTemperatureStatics.for_route(route="generic prepared input",
            policy=policy, landmask=np.zeros_like(skin), lu_index=np.full_like(skin, 21),
            landuse_attrs={"ISLAKE": 21})
        return assemble_horizontal_water_temperature(met, statics)
    coherent, compat = assemble("era5_class_coherent"), assemble("wrf_compat")
    np.testing.assert_array_equal(coherent.water_temperature, 285.)
    np.testing.assert_array_equal(compat.water_temperature[:, 1:], 294.)
    np.testing.assert_array_equal(compat.water_temperature[:, 0], 285.)
    for result, policy in ((coherent, "era5_class_coherent"), (compat, "wrf_compat")):
        assert result.fields["SST"] is sst
        assert result.fields["TT"] is met.fields["TT"]
        assert result.water_temperature_receipt["policy"] == policy
    # The source without a separate SST keeps every original skin value.
    no_sst = replace(met, fields={"SKINTEMP": skin})
    statics = WaterTemperatureStatics.for_route(route="generic skin only", policy=None,
        landmask=np.zeros_like(skin), lu_index=np.full_like(skin, 21), landuse_attrs={"ISLAKE": 21})
    np.testing.assert_array_equal(assemble_horizontal_water_temperature(no_sst, statics).water_temperature, skin)


@pytest.mark.parametrize("chosen", [True, False])
def test_native_spawn_worker_preserves_pressure_operand(monkeypatch, chosen):
    import tools.hrrr_single_domain_benchmark as runner
    seen = []
    def initialize(*args, **kwargs):
        seen.append(kwargs["sfcp_to_sfcp"])
        return {}, {}, {}, {}
    monkeypatch.setattr(runner, "_initialize_boundary_sides", initialize)
    monkeypatch.setattr(runner, "_PREPARE_WORKER_CONTEXT", None)
    runner._prepare_worker_init(object(), {}, [1., 0.], 5000., 5, "cpu", None, chosen)
    runner._prepare_boundary_hour(1, {}, 2, 0)
    assert seen == [chosen]


def test_true_and_false_native_stock_controls_must_agree(tmp_path):
    from gpuwm.hrrr_hierarchy_direct import _require_raw_stock_delta
    from test_hrrr_hierarchy_direct import _raw_runtime_namelist
    native, stock = tmp_path / "native", tmp_path / "stock"
    left = _raw_runtime_namelist(2, longwave=0, theta_m=0)
    right = _raw_runtime_namelist(2, longwave=1, theta_m=1, ghg_input=0, do_radar_ref=1)
    for setting, expected in ((".true.", True), (".false.", False)):
        native.write_text(left.replace("sfcp_to_sfcp = .true.", "sfcp_to_sfcp = " + setting))
        stock.write_text(right.replace("sfcp_to_sfcp = .true.", "sfcp_to_sfcp = " + setting))
        assert _require_raw_stock_delta(native, stock)["certified_native_runtime"]["domains.sfcp_to_sfcp"] == [expected]
    stock.write_text(right)
    with pytest.raises(ValueError, match="same boolean sfcp_to_sfcp"):
        _require_raw_stock_delta(native, stock)


def test_native_solved_surface_consumes_finished_water_once_on_fresh_and_restore():
    from gpuwm.ingest.hrrr_physics import resolve_prepared_noah_surface
    from gpuwm.ingest.prepared_cache import select_prepared_met_fields
    from test_prepared_surface_restore import _native_met, _cfg, _static
    raw = _native_met()
    fields = dict(raw.fields)
    fields["LANDSEA"] = np.zeros_like(fields["LANDSEA"])
    fields["SKINTEMP"] = np.full_like(fields["SKINTEMP"], 285.)
    baseline = resolve_prepared_noah_surface(SimpleNamespace(fields=fields), _cfg(), _static())
    water = np.full_like(fields["SKINTEMP"], 301.)
    providers = np.full(water.shape, 2, dtype=np.uint8)
    receipt = {"policy": "external_overlay", "counts": {"water": 4}}
    met = SimpleNamespace(fields=fields, water_temperature=water,
                          water_temperature_source=providers,
                          water_temperature_receipt=receipt)
    # The native preparation path releases full mapped meteorology before
    # solving Noah. Its retained subset must carry the completed water field.
    selected = select_prepared_met_fields(met)
    solved = resolve_prepared_noah_surface(selected, _cfg(), _static())
    assert not np.array_equal(solved.fields["TSLB"], baseline.fields["TSLB"])
    np.testing.assert_array_equal(solved.fields["TSLB"], 301.)
    assert selected.water_temperature_receipt == receipt
    np.testing.assert_array_equal(selected.water_temperature_source, providers)
    water[:] = 299.
    providers[:] = 9
    receipt["counts"]["water"] = -1
    np.testing.assert_array_equal(selected.water_temperature, 301.)
    np.testing.assert_array_equal(selected.water_temperature_source, 2)
    assert selected.water_temperature_receipt["counts"]["water"] == 4
    # The cache stores the solved water/soil surface. Restore must consume it,
    # even after all native soil inputs and assembly temporaries are gone.
    assert resolve_prepared_noah_surface(SimpleNamespace(fields={}), _cfg(), _static(),
                                        surface=solved) is solved
