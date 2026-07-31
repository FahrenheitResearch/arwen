"""Phase-5 Task 2 gates: generic runtime purity, discovery, and wiring.

Covers the plan's grep-able constant-absence gate, the resolved-config
print (G2 completion), the coverage-derived run ceiling, the
config-driven vertical grid (G4 completion), and the frozen-profile
delegation surface.
"""

from __future__ import annotations

import os

import ast
import re
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from gpuwm import runtime
from gpuwm.case_data import load_experiment_case
from gpuwm.experiment import VerticalConfig

from test_case_data import make_case_toml

REPO = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
MAY99 = Path(os.environ.get("GPUWM_TEST_MAY99_DATA",
                    "gpuwm-fixture-unset/may99-data"))
MAY_FILES = (MAY99 / "era5_may1999_pl.grib",
             MAY99 / "era5_may1999_sl.grib")
requires_may99 = pytest.mark.skipif(
    not all(path.is_file() for path in MAY_FILES)
    or not (BUNDLE / "era5_grib/Vtable.ERA5_CDO").is_file(),
    reason="staged May-1999 ERA5 or reference Vtable is absent",
)
requires_real74 = pytest.mark.skipif(
    not (BUNDLE / "era5_grib/era5_19740403.grb").is_file()
    or not (BUNDLE / "era5_grib/Vtable.ERA5_CDO").is_file(),
    reason="read-only 1974 reference bundle is absent",
)


def test_generic_ingest_and_runtime_have_no_met_em_reader():
    sources = [REPO / "gpuwm/runtime.py"]
    sources.extend((REPO / "gpuwm/ingest").glob("*.py"))
    assert all("met_em" not in path.read_text(encoding="utf-8").lower()
               for path in sources)


def _fixture_pair(tmp_path):
    return load_experiment_case(make_case_toml(tmp_path))


def test_transition_receipt_is_atomic_hash_bound_and_public(
        tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    class Coupler:
        complete = True

        def transition_receipt(self):
            return {"source_domain": 1, "target_domain": 2,
                    "requested_policy": "same-scheme-only",
                    "effective_policy": "same-scheme-only",
                    "process_start_parent_ticks": 0,
                    "process_force_count": 1,
                    "parent_interval_ticks": 3,
                    "final_parent_ticks": 3,
                    "expected_cumulative_force_count": 1,
                    "current_process_coverage_complete": self.complete,
                    "first_parent_ticks": 3,
                    "last_parent_ticks": 3}

    model = SimpleNamespace(
        nodes_by_grid_id={
            1: SimpleNamespace(coupler=None),
            2: SimpleNamespace(coupler=Coupler()),
        },
        experiment_fingerprint="a" * 64,
        root=SimpleNamespace(clock=SimpleNamespace(elapsed_seconds=60.0)),
    )
    monkeypatch.setenv("GPUWM_RUN_ID", "run-123")
    monkeypatch.setenv("GPUWM_CONFIG_DIGEST", "b" * 64)
    path, digest, transitions = \
        runtime._write_microphysics_transition_receipt(
            tmp_path, model, SimpleNamespace(name="mixed"), resumed=False)
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    assert path.name == runtime.MICROPHYSICS_TRANSITION_RECEIPT_NAME
    assert digest == hashlib.sha256(encoded).hexdigest()
    assert payload["schema"] == "gpuwm.microphysics-transitions/v1"
    assert payload["status"] == "PASS"
    assert payload["experiment_fingerprint"] == "a" * 64
    assert payload["run_id"] == "run-123"
    assert payload["config_digest"] == "b" * 64
    assert payload["transitions"] == list(transitions)
    assert not list(tmp_path.glob("*.partial-*"))

    Coupler.complete = False
    with pytest.raises(RuntimeError, match="force coverage is incomplete"):
        runtime._write_microphysics_transition_receipt(
            tmp_path, model, SimpleNamespace(name="mixed"), resumed=False)


# ---------------------------------------------------------------------------
# Gate: grep-able constant absence + no frozen-case import in the runtime.
# ---------------------------------------------------------------------------

def test_runtime_path_carries_no_frozen_case_constants():
    """The formerly implicit constants left the runtime path (G1/G4):
    the frozen profile's pinned names and the fixed forcing-coverage
    ceiling literal must not appear in gpuwm/runtime.py or
    gpuwm/case_data.py, and the runtime path never imports the frozen
    verification cases."""
    for name in ("runtime.py", "case_data.py"):
        text = (REPO / "gpuwm" / name).read_text(encoding="utf-8")
        for token in ("ETA_LEVELS", "START_TIME", "BUNDLE", "43200"):
            assert token not in text, (name, token)
        assert re.search(r"(from|import)\s+gpuwm\.verify", text) is None, name
        assert "verify.cases" not in text, name


def test_normal_output_path_has_no_case_literals_or_inline_assertions():
    paths = [REPO / "gpuwm" / "runtime.py",
             *(REPO / "gpuwm" / "io").glob("*.py")]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for token in ("real74", "1974"):
            assert token not in lowered, (path, token)
        inline_assertions = [node for node in ast.walk(ast.parse(text))
                             if isinstance(node, ast.Assert)]
        assert not inline_assertions, path


def test_frozen_profile_keeps_its_ceiling_and_delegates_the_machinery():
    """The pinned run ceiling stays in the frozen profile; the moved
    machinery is shared by identity (aliases), so the two paths cannot
    fork silently."""
    from gpuwm.verify.cases import real74_d01

    assert real74_d01.RealCaseRunSummary is runtime.RealCaseRunSummary
    assert real74_d01._refl_10cm_due is runtime.refl_10cm_due
    assert real74_d01._whole_step_count is runtime.whole_step_count

    cfg = replace(real74_d01.phase3_config(), run_seconds=86400.0)
    with pytest.raises(ValueError, match="43200"):
        real74_d01._configured_run_schedule(cfg)
    # The generic schedule has no fixed ceiling: the same duration is a
    # legal whole-step schedule (its ceiling derives from forcing
    # coverage in forcing_schedule).
    assert runtime.configured_run_schedule(cfg) == (1440, 60)


# ---------------------------------------------------------------------------
# Gate: coverage-derived run ceiling + forcing discovery.
# ---------------------------------------------------------------------------

def test_run_seconds_beyond_forcing_coverage_rejected_with_coverage(
        tmp_path):
    exp, data = _fixture_pair(tmp_path)
    start = exp.start_time
    times = [start, start + timedelta(hours=6), start + timedelta(hours=12)]
    long = replace(exp, run_seconds=12.5 * 3600.0)
    with pytest.raises(ValueError) as err:
        runtime.forcing_schedule(long, data, times)
    message = str(err.value)
    assert "run_seconds" in message
    assert "coverage" in message
    assert f"{12 * 3600.0:g}" in message      # the coverage, stated
    # Exactly at coverage is accepted.
    exact = replace(exp, run_seconds=12 * 3600.0)
    assert runtime.forcing_schedule(exact, data, times) == tuple(times)


def test_forcing_discovery_validates_start_interval_and_count(tmp_path):
    exp, data = _fixture_pair(tmp_path)
    start = exp.start_time
    with pytest.raises(ValueError, match="no snapshot at the experiment"):
        runtime.forcing_schedule(
            exp, data, [start + timedelta(hours=1),
                        start + timedelta(hours=7)])
    with pytest.raises(ValueError, match="at least one interval"):
        runtime.forcing_schedule(exp, data, [start])
    # Declared interval policy enforced against the discovered schedule.
    with pytest.raises(ValueError, match="forcing_interval_s"):
        runtime.forcing_schedule(
            exp, data, [start, start + timedelta(hours=3),
                        start + timedelta(hours=9)])
    # Snapshots before start_time are ignored by discovery.
    times = [start - timedelta(hours=6), start,
             start + timedelta(hours=6), start + timedelta(hours=12)]
    assert runtime.forcing_schedule(exp, data, times) == tuple(times[1:])


# ---------------------------------------------------------------------------
# Gate: two distinct legal eta_levels lists -> two distinct vertical
# grids from config (G4 completion).
# ---------------------------------------------------------------------------

def test_two_distinct_eta_level_lists_build_two_distinct_grids():
    nz = 4
    eta_a = (1.0, 0.8, 0.55, 0.3, 0.0)
    eta_b = (1.0, 0.9, 0.7, 0.4, 0.0)
    coord_a = runtime.vertical_coord_for(
        VerticalConfig(eta_levels=eta_a, p_top=10000.0, hybrid_opt=2,
                       etac=0.2), nz)
    coord_b = runtime.vertical_coord_for(
        VerticalConfig(eta_levels=eta_b, p_top=10000.0, hybrid_opt=2,
                       etac=0.2), nz)
    np.testing.assert_array_equal(coord_a.znw, np.asarray(eta_a))
    np.testing.assert_array_equal(coord_b.znw, np.asarray(eta_b))
    assert not np.array_equal(coord_a.znw, coord_b.znw)
    assert not np.array_equal(coord_a.dnw, coord_b.dnw)
    # The idealized empty declaration has no real-data vertical grid.
    with pytest.raises(ValueError, match="eta_levels"):
        runtime.vertical_coord_for(
            VerticalConfig(eta_levels=(), p_top=0.0, hybrid_opt=2,
                           etac=0.2), nz)


# ---------------------------------------------------------------------------
# Gate: resolved-config print enumerates every path/time/policy (G2).
# ---------------------------------------------------------------------------

def test_resolved_config_report_enumerates_every_path_time_policy(
        tmp_path):
    from types import SimpleNamespace

    exp, data = _fixture_pair(tmp_path)
    start = exp.start_time
    times = (start, start + timedelta(hours=6), start + timedelta(hours=12))
    exclusions = (start - timedelta(hours=12),
                  start + timedelta(hours=18))
    catalog = SimpleNamespace(
        valid_times=times, excluded_valid_times=exclusions)
    report = runtime.resolved_config_report(
        exp, data, forcing_times=times, input_catalog=catalog)

    # Every declared input path appears verbatim.
    for record in data.resolved_inputs():
        assert str(record.path) in report, record
    # Every formerly implicit path/time/policy is named.
    for key in (
            "input.forcing", "input.vtable", "input.wps_namelist",
            "input.geog_root", "input.source_orography", "variable=SOILHGT",
            "time.start_time", "time.run_seconds", "time.dt",
            "time.history_interval_s", "time.restart_interval_s",
            "radiation.column_chunk",
            "time.forcing_interval_s", "time.forcing_times",
            "time.forcing_times_consumed",
            "time.forcing_times_excluded_by_catalog",
            "time.forcing_coverage_s",
            "vertical.nz", "vertical.eta_levels", "vertical.p_top",
            "vertical.hybrid_opt", "vertical.etac",
            "policy.sfcp_to_sfcp", "policy.co2_vmr",
            "policy.climatology_date",
            "output.domain_id", "output.title", "output.filename_pattern",
            "grid.nx_ny_dx"):
        assert key in report, key
    assert start.isoformat() in report
    assert f"{12 * 3600.0:g}" in report          # the stated coverage
    assert all(value.isoformat() in report for value in exclusions)
    assert "fixture title" in report
    assert "wrfout_d01_" in report
    # Without discovered times the declared interval policy still prints.
    partial = runtime.resolved_config_report(exp, data)
    assert "time.forcing_interval_s" in partial
    assert "time.forcing_times" not in partial


# ---------------------------------------------------------------------------
# Single-domain scope, keyed decode cache, trace-gas hook surface, CLI.
# ---------------------------------------------------------------------------

def test_multi_domain_experiment_is_rejected_with_task14_pointer():
    exp = load_experiment_case(REPO / "configs" / "real74_4dom.toml")[0]
    with pytest.raises(NotImplementedError, match="Task 14"):
        runtime.single_domain(exp)


def test_cached_era5_snapshots_is_keyed_by_resolved_inputs(
        tmp_path, monkeypatch):
    """Different spellings of one resolved input pair share one decode;
    a different pair gets its own (the argument-less single-slot cache
    pattern is retired)."""
    from gpuwm.ingest import grib

    calls = []
    monkeypatch.setattr(
        grib, "decode_era5_grib",
        lambda grib_path, vtable_path, *, bridge=None: calls.append(
            (grib_path, vtable_path)) or ())
    grib._decode_era5_grib_resolved.cache_clear()
    try:
        grib_a = tmp_path / "a.grb"
        vtable = tmp_path / "Vtable"
        spelled = tmp_path / "sub" / ".." / "a.grb"
        assert grib.cached_era5_snapshots(grib_a, vtable) == ()
        assert grib.cached_era5_snapshots(spelled, vtable) == ()
        assert len(calls) == 1
        grib_b = tmp_path / "b.grb"
        assert grib.cached_era5_snapshots(grib_b, vtable) == ()
        assert len(calls) == 2
    finally:
        grib._decode_era5_grib_resolved.cache_clear()


def _assert_snapshots_value_identical(actual, expected):
    assert tuple(snapshot.valid_time for snapshot in actual) == tuple(
        snapshot.valid_time for snapshot in expected)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left.levels_hpa, right.levels_hpa)
        np.testing.assert_array_equal(left.latitude, right.latitude)
        np.testing.assert_array_equal(left.longitude, right.longitude)
        assert tuple(left.fields) == tuple(right.fields)
        for name in left.fields:
            np.testing.assert_array_equal(left.fields[name], right.fields[name])


def test_runtime_decode_filters_synthetic_per_date_extra_from_catalog_selection(
        tmp_path, monkeypatch):
    """A surface-only CDS cross-product record is outside runtime authority."""
    from types import SimpleNamespace
    from gpuwm.ingest import grib

    start = datetime(1999, 5, 3, 12)
    selected = (start, start + timedelta(hours=6),
                start + timedelta(hours=12))
    extra = start - timedelta(hours=12)
    lat = np.asarray([1.0, 0.0], dtype=np.float64)
    lon = np.asarray([10.0, 11.0], dtype=np.float64)
    pressure = tmp_path / "pressure.grib"
    surface = tmp_path / "surface.grib"
    vtable = tmp_path / "Vtable.ERA5"
    executable = tmp_path / "grib1_bridge.exe"
    for path in (pressure, surface, vtable, executable):
        path.write_bytes(b"fixture")

    def partial(path, valid_time, *, pressure_level):
        if pressure_level:
            return grib._PartialSnapshot(
                path, valid_time, (100, 1000), lat, lon,
                {"T": np.full((2, 2, 2), 275.0, dtype=np.float64)})
        return grib._PartialSnapshot(
            path, valid_time, (), lat, lon,
            {"T2": np.full((2, 2), 285.0, dtype=np.float64)})

    pressure_partials = tuple(
        partial(pressure, value, pressure_level=True) for value in selected)
    surface_partials = tuple(
        partial(surface, value, pressure_level=False)
        for value in (extra, *selected))
    all_partials = (*pressure_partials, *surface_partials)

    monkeypatch.setattr(grib, "parse_vtable", lambda path: ())
    monkeypatch.setattr(grib, "build_rust_bridge",
                        lambda release=True: executable)
    monkeypatch.setattr(
        grib, "_decode_bridge_partials",
        lambda path, entries, bridge: (
            pressure_partials if Path(path) == pressure.resolve()
            else surface_partials))
    grib._decode_era5_gribs_resolved.cache_clear()
    grib._decode_era5_forcing_partials_resolved.cache_clear()
    try:
        catalog = SimpleNamespace(
            valid_times=selected, excluded_valid_times=(extra,), files=())
        data = SimpleNamespace(
            forcing=(pressure, surface), vtable=vtable)
        snapshots = runtime.forcing_snapshots(data, catalog)
        assert tuple(snapshots) == selected
        assert extra not in snapshots

        # If an incomplete time is selected, the completeness diagnostic also
        # names the catalog exclusions that explain the decode boundary.
        older = extra - timedelta(hours=6)
        with pytest.raises(ValueError) as caught:
            grib._merge_partials(
                all_partials, valid_times=(*selected, extra),
                excluded_valid_times=(older,))
        message = str(caught.value)
        assert "no pressure-level fields" in message
        assert "catalog exclusions" in message
        assert older.isoformat() in message
    finally:
        grib._decode_era5_gribs_resolved.cache_clear()
        grib._decode_era5_forcing_partials_resolved.cache_clear()


@requires_may99
def test_real_may1999_ingest_decode_consumes_exact_catalog_times():
    """Read-only staged regression for the failure that opened p5t6fix."""
    from gpuwm.ingest.preflight import build_input_catalog

    base_exp, base_data = load_experiment_case(
        REPO / "configs/real74_d01_exp.toml")
    exp = replace(base_exp, start_time=datetime(1999, 5, 3, 12))
    data = replace(
        base_data, forcing=MAY_FILES, source_orography=None, co2_vmr=None)
    catalog = build_input_catalog(data)
    decoded = runtime.forcing_snapshots(data, catalog)

    expected = (
        datetime(1999, 5, 3, 12), datetime(1999, 5, 3, 18),
        datetime(1999, 5, 4, 0))
    assert catalog.valid_times == expected
    assert catalog.excluded_valid_times == (
        datetime(1999, 5, 3, 0), datetime(1999, 5, 4, 12),
        datetime(1999, 5, 4, 18))
    assert runtime.forcing_schedule(exp, data, decoded) == expected
    _assert_snapshots_value_identical(
        tuple(decoded.values()), catalog.snapshots)


@requires_real74
def test_real74_full_catalog_selection_is_value_and_order_identical():
    """The all-three-times frozen forcing shape is bit-inert at this seam."""
    from gpuwm.ingest.preflight import build_input_catalog

    _exp, data = load_experiment_case(
        REPO / "configs/real74_d01_exp.toml")
    catalog = build_input_catalog(data)
    assert catalog.excluded_valid_times == ()
    decoded = runtime.forcing_snapshots(data, catalog)
    _assert_snapshots_value_identical(
        tuple(decoded.values()), catalog.snapshots)


def test_rrtmgp_trace_gas_policy_hook_is_dated_and_profile_declared(
        monkeypatch):
    """None selects the dated policy; real74 declares its 330 ppm override."""
    import dataclasses
    from types import SimpleNamespace
    from gpuwm.core import rrtmgp

    fields = {field.name: field
              for field in dataclasses.fields(rrtmgp.RRTMGPRadiation)}
    assert "trace_gas_overrides" in fields
    assert fields["trace_gas_overrides"].default is None
    assert rrtmgp.trace_gases(datetime(1999, 5, 3))["co2"] == pytest.approx(
        367.80e-6)
    from gpuwm.verify.cases import real74_d01

    captured = {}

    def fake_prepare(cfg, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            cfg=cfg, grid=kwargs["grid"], static_fields={},
            initial_result=object(), final_analysis=object(),
            initial_snow_water_kgm2=np.empty((0, 0)))

    monkeypatch.setattr(real74_d01.runtime, "prepare_real_case", fake_prepare)
    monkeypatch.setattr(real74_d01, "grids_from_wps_namelist",
                        lambda path: (object(),))
    real74_d01.prepare_phase3_case()
    assert captured["trace_gas_overrides"] == {"co2": 330.0e-6}
    assert real74_d01.SOURCE_OROGRAPHY.variable == "SOILHGT"
    assert real74_d01.SOURCE_OROGRAPHY.path.name.startswith("met_em.d01.")


def test_experiment_runtime_threads_catalog_providers_when_declarations_omitted(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gpuwm.ingest import preflight

    exp, declared = _fixture_pair(tmp_path)
    data = replace(declared, source_orography=None, co2_vmr=None)
    times = (exp.start_time, exp.start_time + timedelta(hours=6),
             exp.start_time + timedelta(hours=12))
    snapshots = tuple(SimpleNamespace(valid_time=value) for value in times)
    catalog = SimpleNamespace(
        snapshots=snapshots, inventory=("SOILGEO",),
        units={"SOILGEO": "m2 s-2"}, valid_times=times,
        excluded_valid_times=())
    monkeypatch.setattr(preflight, "build_input_catalog", lambda value: catalog)
    monkeypatch.setattr(
        runtime, "forcing_snapshots",
        lambda case_data, input_catalog: {
            snapshot.valid_time: snapshot for snapshot in snapshots})
    monkeypatch.setattr(runtime, "experiment_grid",
                        lambda experiment, case_data: object())
    monkeypatch.setattr(
        runtime.GeogSelection, "from_case_data",
        lambda case_data, domain_id: None)
    sentinel = object()
    captured = {}

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(runtime, "prepare_real_case", fake_prepare)
    assert runtime.prepare_experiment_case(exp, data) is sentinel
    assert captured["forcing_catalog"] is catalog
    assert captured["source_orography_path"] is None
    assert captured["source_orography_variable"] is None
    assert captured["trace_gas_overrides"] is None


def test_single_domain_implicit_trace_gases_keep_experiment_column_chunk(
        tmp_path, monkeypatch):
    """The dated trace-gas policy must not drop the configured RRTMGP chunk."""
    import sys
    from types import SimpleNamespace

    from gpuwm.core import diagnostics, physics, rrtmgp

    exp, declared = _fixture_pair(tmp_path)
    dc = replace(exp.root, run=replace(exp.root.run, ra_physics=4))
    exp = replace(exp, domains=(dc,), column_chunk=47)
    data = replace(declared, source_orography=None, co2_vmr=None)
    assert len(exp.domains) == 1
    assert data.co2_vmr is None

    ny, nx = dc.run.ny, dc.run.nx
    zeros = np.zeros((ny, nx), dtype=np.float32)
    ones = np.ones((ny, nx), dtype=np.float32)

    class Grid:
        e_we = nx + 1
        e_sn = ny + 1
        dx = dc.run.dx
        dy = dc.run.dy

        def coriolis_m(self):
            return zeros, zeros

        def rotation_m(self):
            return zeros, ones

        def mapfac_m(self):
            return ones

        def mapfac_u(self):
            return np.ones((ny, nx + 1), dtype=np.float32)

        def mapfac_v(self):
            return np.ones((ny + 1, nx), dtype=np.float32)

        def latlon_mass(self):
            return zeros, zeros

    class State:
        physics = None

        def set_map_coriolis(self, *args, **kwargs):
            pass

    met_fields = {
        "T2": zeros.copy(),
        "U10": np.zeros((ny, nx + 1), dtype=np.float32),
        "V10": np.zeros((ny + 1, nx), dtype=np.float32),
    }
    lu_index = ones.copy()
    lu_index[0, 0] = np.float32(21.0)
    static = {
        "HGT_M": zeros, "SCT_DOM": ones, "TMN": zeros,
        "GREENFRAC": np.zeros((12, ny, nx), dtype=np.float32),
        "ALBEDO12M": np.zeros((12, ny, nx), dtype=np.float32),
        "LAI12M": np.zeros((12, ny, nx), dtype=np.float32),
        "LANDMASK": ones, "LU_INDEX": lu_index,
        "SNOALB": zeros,
    }
    soil = SimpleNamespace(
        tsk=zeros, soil_temperature=zeros, soil_moisture=zeros,
        liquid_moisture=zeros, deep_soil_temperature=zeros, xice=zeros,
        snow_water=zeros, snow_depth=zeros)

    monkeypatch.setattr(runtime, "experiment_grid", lambda *args: Grid())
    selection = SimpleNamespace(landuse_global_attrs=lambda: {
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
        "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13})
    monkeypatch.setattr(
        runtime.GeogSelection, "from_case_data",
        lambda *args, **kwargs: selection)
    monkeypatch.setattr(runtime, "_cached_static_build", lambda *args, **kw: static)
    monkeypatch.setattr(
        runtime, "interpolate_era5_to_lambert",
        lambda *args, **kwargs: SimpleNamespace(fields=met_fields))
    monkeypatch.setattr(runtime, "vertical_coord_for", lambda *args: object())

    def fake_initialize_real(*args, **kwargs):
        return SimpleNamespace(
            state=State(), surface_pressure=ones, surface_qv=zeros)

    monkeypatch.setattr(runtime, "initialize_real", fake_initialize_real)
    monkeypatch.setattr(
        runtime, "build_state_lateral_boundaries", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "attach_lateral_boundaries", lambda *args: None)
    soil_kwargs = {}

    def preprocess(*args, **kwargs):
        soil_kwargs.update(kwargs)
        return soil

    monkeypatch.setattr(runtime, "preprocess_land_surface_soil", preprocess)
    monkeypatch.setattr(
        runtime, "monthly_interp_to_date", lambda values, when: values[0])
    monkeypatch.setattr(diagnostics, "update_diagnostics", lambda *args: None)
    monkeypatch.setattr(rrtmgp.RRTMGPRadiation, "__post_init__", lambda self: None)

    physics_kwargs = {}

    def fake_initialize_physics(state, cfg, *, radiation, **kwargs):
        physics_kwargs.update(kwargs)
        fields = {
            name: np.zeros((ny, nx), dtype=np.float32)
            for name in ("snoalb", "albbck", "lai", "shdmin", "shdmax",
                         "psfc", "t2", "q2", "th2", "u10", "v10")
        }
        # ``noah_params`` is part of what the real driver returns, and
        # ``gpuwm/runtime.py`` reads it to initialize SNOALB (the knob
        # lane's ported ``rdmaxalb`` branch, WRF
        # module_sf_noahdrv.F:1902-1903).  ``None`` is the production
        # constructor's own default and is what this fixture's
        # ``rdmaxalb=True`` path uses -- that branch keeps the supplied
        # geogrid percentage and never reads the VEGPARM table.
        state.physics = SimpleNamespace(
            radiation_callable=radiation, fields=fields,
            noah_params=None)
        return state.physics

    monkeypatch.setattr(physics, "initialize_physics", fake_initialize_physics)
    monkeypatch.setitem(sys.modules, "cupy", np)

    times = (exp.start_time, exp.start_time + timedelta(hours=6))
    catalog = SimpleNamespace(
        valid_times=times, excluded_valid_times=(), inventory=())
    snapshots = {valid_time: object() for valid_time in times}
    prepared = runtime.prepare_experiment_case(
        exp, data, input_catalog=catalog, forcing_by_time=snapshots)

    # No lake skin override on the ERA5 lane: metgrid's masked=both
    # SKINTEMP chain with static-landmask targets already yields the
    # water-source skin at lake cells (real.exe without TAVGSFC keeps it).
    assert soil_kwargs.get("lake_mask") is None
    assert soil_kwargs.get("lake_skin_temperature") is None
    assert "landmask" in soil_kwargs

    radiation = prepared.initial_result.state.physics.radiation_callable
    assert isinstance(radiation, rrtmgp.RRTMGPRadiation)
    assert radiation.trace_gas_overrides is None
    assert radiation.column_chunk == exp.column_chunk
    landuse = physics_kwargs["landuse"]
    expected_ivgtyp = np.ones((ny, nx), dtype=np.int32)
    expected_ivgtyp[0, 0] = 17
    expected_xland = np.ones((ny, nx), dtype=np.float32)
    expected_xland[0, 0] = 2.0
    np.testing.assert_array_equal(landuse.ivgtyp, expected_ivgtyp)
    np.testing.assert_array_equal(landuse.xland, expected_xland)
    expected_lake = np.zeros((ny, nx), dtype=bool)
    expected_lake[0, 0] = True
    np.testing.assert_array_equal(landuse.lakemask, expected_lake)
    np.testing.assert_array_equal(landuse.pblh, 0.0)
    np.testing.assert_array_equal(landuse.ust, np.float32(1.0e-4))
    assert "landmask" not in physics_kwargs


def test_prepare_real_case_fails_before_io_when_both_orography_sources_exist(
        tmp_path):
    from types import SimpleNamespace

    exp, data = _fixture_pair(tmp_path)
    cfg = runtime.single_domain(exp).run

    class Grid:
        e_we = cfg.nx + 1
        e_sn = cfg.ny + 1
        dx = cfg.dx
        dy = cfg.dy

    with pytest.raises(ValueError) as caught:
        runtime.prepare_real_case(
            cfg, grid=Grid(), geog_root=data.geog_root,
            source_orography_path=data.source_orography.path,
            source_orography_variable=data.source_orography.variable,
            vertical=exp.vertical, sfcp_to_sfcp=True,
            snapshot_for=lambda value: None,
            forcing_times=(exp.start_time,
                           exp.start_time + timedelta(hours=6)),
            start_time=exp.start_time,
            forcing_catalog=SimpleNamespace(inventory=("SOILGEO",)))
    message = str(caught.value)
    assert str(data.source_orography.path) in message
    assert "variable=SOILHGT" in message
    assert "SOILGEO via era5_z_invariant" in message


def test_run_experiment_threads_experiment_timing_authority(
        tmp_path, monkeypatch):
    """History/run/restart cadence comes from Experiment/DomainConfig."""
    exp, data = _fixture_pair(tmp_path)
    dc = exp.domains[0]
    compatibility_copy = replace(
        dc.run, run_seconds=600.0, output_interval_s=60.0,
        restart_interval_s=120.0)
    authoritative_domain = replace(
        dc, history_interval_s=1800.0, run=compatibility_copy)
    authoritative = replace(
        exp, run_seconds=3600.0, restart_interval_s=0.0,
        domains=(authoritative_domain,))
    prepared = object()
    captured = {}
    from types import SimpleNamespace
    from gpuwm.ingest import preflight

    catalog = SimpleNamespace(
        valid_times=(authoritative.start_time,
                     authoritative.start_time + timedelta(hours=1)),
        excluded_valid_times=())

    monkeypatch.setattr(preflight, "build_input_catalog", lambda data: catalog)
    monkeypatch.setattr(
        runtime, "forcing_snapshots", lambda data, input_catalog: {})
    monkeypatch.setattr(
        runtime, "forcing_schedule",
        lambda exp, data, snapshots: (
            exp.start_time, exp.start_time + timedelta(hours=1)))
    monkeypatch.setattr(
        runtime, "prepare_experiment_case",
        lambda exp, data, **kwargs: prepared)

    def integrate(*args, **kwargs):
        captured.update(kwargs)
        return "summary"

    monkeypatch.setattr(runtime, "integrate_prepared_case", integrate)
    assert runtime.run_experiment(
        authoritative, data, tmp_path / "out") == "summary"
    assert captured["run_seconds"] == authoritative.run_seconds
    assert (captured["history_interval_s"]
            == authoritative_domain.history_interval_s)
    assert (captured["restart_interval_s"]
            == authoritative.restart_interval_s)


def test_prepare_real_case_rejects_trace_overrides_without_rrtmgp(
        tmp_path):
    exp, data = _fixture_pair(tmp_path)
    dc = runtime.single_domain(exp)
    assert dc.run.ra_physics == 0

    class _Grid:
        e_we = dc.run.nx + 1
        e_sn = dc.run.ny + 1
        dx = dc.run.dx
        dy = dc.run.dy

    with pytest.raises(ValueError, match="ra_physics = 4"):
        runtime.prepare_real_case(
            dc.run, grid=_Grid(), geog_root=data.geog_root,
            source_orography_path=data.source_orography.path,
            source_orography_variable=data.source_orography.variable,
            vertical=exp.vertical, sfcp_to_sfcp=data.sfcp_to_sfcp,
            snapshot_for=lambda t: None,
            forcing_times=(exp.start_time,
                           exp.start_time + timedelta(hours=6)),
            start_time=exp.start_time,
            trace_gas_overrides={"co2": data.co2_vmr})


def test_cli_experiment_commands_route_through_runtime(
        tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    import gpuwm.cli as cli
    import gpuwm.case_data as case_data_module
    import gpuwm.supervisor as supervisor

    exp_sentinel = SimpleNamespace(name="stub")
    data_sentinel = SimpleNamespace()
    calls = []
    monkeypatch.setattr(case_data_module, "load_experiment_case",
                        lambda path: (exp_sentinel, data_sentinel))
    monkeypatch.setattr(
        runtime, "write_static",
        lambda exp, data, output: calls.append(("static", exp, data))
        or Path(output))
    monkeypatch.setattr(
        runtime, "write_ingest",
        lambda exp, data, output: calls.append(("ingest", exp, data))
        or Path(output))

    class _Summary:
        wrfout_paths = (Path("a"), Path("b"))
        completed_seconds = 3600.0
        nan_free = True

    monkeypatch.setattr(
        runtime, "run_experiment",
        lambda exp, data, outdir, restart=None, health_debug=False:
        calls.append(("run", exp, data, Path(outdir), restart,
                      health_debug)) or _Summary())

    def supervised(args):
        summary = runtime.run_experiment(
            exp_sentinel, data_sentinel, args.outdir, restart=args.restart,
            health_debug=args.health_debug)
        print({"wrfout_count": len(summary.wrfout_paths),
               "nan_free": summary.nan_free})
        return 0

    monkeypatch.setattr(supervisor, "supervise_from_cli", supervised)

    path = tmp_path / "exp.toml"
    path.write_text('[experiment]\nname = "stub"\n', encoding="utf-8")
    assert cli.main(["static", str(path),
                     "--output", str(tmp_path / "s.npz")]) == 0
    assert cli.main(["ingest", str(path),
                     "--output", str(tmp_path / "i.npz")]) == 0
    assert cli.main(["run", str(path),
                     "--outdir", str(tmp_path / "out")]) == 0
    kinds = [entry[0] for entry in calls]
    assert kinds == ["static", "ingest", "run"]
    assert all(entry[1] is exp_sentinel and entry[2] is data_sentinel
               for entry in calls)
    run_call = calls[-1]
    assert run_call[3] == tmp_path / "out" and run_call[4] is None
    assert run_call[5] is False
    out = capsys.readouterr().out
    assert "'wrfout_count': 2" in out
    assert "'nan_free': True" in out
