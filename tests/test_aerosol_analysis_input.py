"""Analyzed aerosol preserves donor pressure and reaches specified boundaries."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import ctypes
import os
import shutil
import subprocess

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_aerosol_source_options
from gpuwm.ingest.analyzed_numbers import CANONICAL_NUMBER_FIELDS
from gpuwm.ingest.horiz import regular_horizontal_method
from gpuwm.ingest.real import initialize_real


def test_analyzed_selection_is_strict_and_independent_of_climatology_pair():
    from test_mp28_runnable import _cfg
    for changes in ({"use_rap_aero_icbc": True}, {"mp28_aerosol_source": "analysis"},
                    {"use_rap_aero_icbc": True, "aer_init_opt": 1, "wif_input_opt": 1}):
        validate_aerosol_source_options(_cfg(mp_physics=28, **changes))
    with pytest.raises(ValueError, match="cannot also select"):
        validate_aerosol_source_options(_cfg(
            mp_physics=28, use_rap_aero_icbc=True, mp28_aerosol_source="synthetic"))


def test_number_rows_use_operational_nearest_operator():
    # A bilinear/parabolic replacement smooths sharp aerosol gradients before
    # initialization. Operational METGRID.TBL QN* rows start nearest_neighbor.
    for name in CANONICAL_NUMBER_FIELDS.values():
        assert regular_horizontal_method(name, 3) == "nearest"


def test_missing_nearest_aerosol_uses_neighbors_before_zero_fill():
    from test_metgrid_number_initialization import _require_cpu_bridge
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    _require_cpu_bridge()
    values, counts = CpuPreprocessBackend().wps_masked_chain(
        np.array([[[np.nan, 20.], [40., 60.]]]), np.ones((2, 2), dtype=bool),
        None, np.array([[.1, .8, 0.]]), np.array([[.1, .8, 0.]]),
        np.ones((1, 3), dtype=bool), ("nearest_neighbor", "four_pt", "average_4pt"),
        mode="plain", fill_value=0.0)
    np.testing.assert_array_equal(values, [[40., 60., 0.]])
    assert counts[0, 3] == 1


def test_missing_analyzed_pair_fails_before_state_allocation():
    from test_metem_differential import _synthetic
    snapshot, cfg, coord, terrain, orography = _synthetic(2500., 2500.)
    cfg = replace(cfg, mp_physics=28, mp28_aerosol_source="analysis")
    with pytest.raises(ValueError, match="missing QNIFA, QNWFA"):
        initialize_real(snapshot, cfg, coord, terrain, source_orography=orography,
                        preprocess_backend="cpu", state_backend="cpu", analyzed_species=())


def test_surface_emission_dataset_guard_does_not_replace_analyzed_icbc(monkeypatch):
    from test_metgrid_number_initialization import _case, _no_wif_dataset
    from gpuwm.config import mp28_aerosol_lateral_forcing_precondition
    _no_wif_dataset(monkeypatch)
    cfg = replace(_case(28)[1], mp28_aerosol_source="analysis")
    assert mp28_aerosol_lateral_forcing_precondition(cfg) is None
    reason = mp28_aerosol_lateral_forcing_precondition(replace(cfg, use_rap_aero_icbc=True))
    assert "two-dimensional surface emission" in reason
    assert "analyzed three-dimensional aerosol" in reason


def test_donor_uses_its_own_pressure_and_forces_both_boundary_numbers():
    from test_metgrid_number_initialization import _case, _require_cpu_bridge
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    from gpuwm.ingest.horiz import HorizontalSnapshot
    _require_cpu_bridge()
    snapshot, cfg, coord, terrain, orography = _case(28)
    cfg = replace(cfg, mp28_aerosol_source="analysis", specified=True)
    fields = {k: v for k, v in snapshot.fields.items()
              if k not in ("QNWFA", "QNIFA", "QNWFA_SFC", "QNIFA_SFC")}
    snapshot = replace(snapshot, fields=fields)
    # Distinct donor ladder and zero donor moisture expose accidental reuse
    # of the meteorological analysis's moisture-adjusted pressure coordinate.
    levels = np.array([760., 580., 390., 220., 100., 35.], dtype=np.float64)
    shape = snapshot.fields["TT"].shape
    pressure = np.broadcast_to(levels[:, None, None] * 100., shape).copy()
    donor_fields = {
        "PRES": pressure, "SPFH": np.zeros(shape), "Q2": np.zeros(terrain.shape),
        "PSFC": np.full(terrain.shape, 77000.), "T2": snapshot.fields["T2"],
        "TT": snapshot.fields["TT"], "GHT": snapshot.fields["GHT"],
        "SOURCE_OROGRAPHY": orography,
        "QNWFA": (pressure * .73 + 600.).astype(np.float32),
        "QNIFA": (pressure * .031 + 30.).astype(np.float32),
    }
    for name in ("QNWFA", "QNIFA"):
        donor_fields[name + "_SFC"] = donor_fields[name][0].copy()
    donor = HorizontalSnapshot(snapshot.valid_time, levels, donor_fields)
    result = initialize_real(snapshot, cfg, coord, terrain,
        source_orography=orography, p_top=5000., preprocess_backend="cpu",
        state_backend="cpu", analyzed_species=(), aerosol_snapshot=donor)
    native = CpuPreprocessBackend()
    for name, target in (("QNWFA", "nwfa"), ("QNIFA", "nifa")):
        expected = native.wrf_vertical_interpolate(
            donor_fields[name], donor_fields[name + "_SFC"], pressure,
            donor_fields["PSFC"], result.dry_pressure,
            interp_in_logp=True, extrap="constant", vboundb=cfg.nz + 1)
        np.testing.assert_array_equal(getattr(result.state, target), expected)
    assert result.state._external_scalar_boundary_fields == ("qv", "nwfa", "nifa")
    assert result.hydrometeor_initialization["number_moments"]["separate_aerosol_donor"]
    assert result.aerosol_initialization["aerosol_source"] == "metgrid-analyzed"
    with pytest.raises(ValueError, match="same valid time"):
        initialize_real(snapshot, cfg, coord, terrain,
            source_orography=orography, aerosol_snapshot=replace(
                donor, valid_time=donor.valid_time + timedelta(hours=1)))


def test_operational_namelist_selects_analyzed_pair(tmp_path):
    from test_mp28_runnable import _namelist_pair, _CONSTANT_GLW_ACK
    from gpuwm.namelist_import import import_namelists
    wps, inp = _namelist_pair(tmp_path, mp=28,
        physics_extra=" use_aero_icbc = .true., use_rap_aero_icbc = .true.,\n")
    toml, _ = import_namelists(wps, inp, name="analyzed-aerosol",
                              acknowledgements=_CONSTANT_GLW_ACK)
    assert "use_rap_aero_icbc = true" in toml
    assert 'mp28_aerosol_source = "analysis"' in toml


def test_operational_analysis_keeps_monthly_surface_emission_separate():
    from test_metgrid_number_initialization import _case, _require_cpu_bridge
    _require_cpu_bridge()
    staged = os.environ.get("GPUWM_WIF_CLIMATOLOGY")
    if not staged or not Path(staged).is_file():
        pytest.skip("the WRF monthly aerosol dataset is needed for the real emission ingest")
    snapshot, cfg, coord, terrain, orography = _case(28)
    cfg = replace(cfg, mp28_aerosol_source="analysis", use_rap_aero_icbc=True,
                  wif_climatology_path=staged)
    kwargs = dict(source_orography=orography, p_top=5000.,
                  preprocess_backend="cpu", state_backend="cpu", analyzed_species=(),
                  wif_grid_latlon=(np.full(terrain.shape, 35.), np.full(terrain.shape, -97.)))
    operational = initialize_real(snapshot, cfg, coord, terrain, **kwargs)
    generic = initialize_real(snapshot, replace(cfg, use_rap_aero_icbc=False,
                               wif_climatology_path=""), coord, terrain, **kwargs)
    for name in ("nwfa", "nifa"):
        np.testing.assert_array_equal(getattr(operational.state, name), getattr(generic.state, name))
    assert np.all(np.isfinite(operational.state.nwfa2d))
    assert np.all(operational.state.nwfa2d > 0)
    assert not np.array_equal(operational.state.nwfa2d, generic.state.nwfa2d)
    receipt = operational.aerosol_initialization
    assert receipt["aerosol_source"] == "metgrid-analyzed"
    assert receipt["surface_emission"]["source"] == "monthly-climatology"


def test_surface_emission_matches_operational_fortran_bits(tmp_path):
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is required for the operational surface-emission oracle")
    source = Path(__file__).with_name("oracles") / "aerosol_surface_mass.f90"
    library = tmp_path / "aerosol_surface.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-O0", "-ffp-contract=off",
                    str(source), "-o", str(library)], check=True)
    oracle = ctypes.CDLL(str(library)).aerosol_surface_oracle
    oracle.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] + [ctypes.c_float] * 3
    count = 2048
    shape = (32, 64)
    number = np.geomspace(1., 1.e10, count).astype(np.float32).reshape(shape)
    lower = np.linspace(0., 30000., count, dtype=np.float32).reshape(shape)
    upper = lower + np.linspace(90., 1600., count, dtype=np.float32).reshape(shape)
    alt = np.linspace(.5, 1.8, count, dtype=np.float32).reshape(shape)
    expected = np.empty(shape, dtype=np.float32)
    native = CpuPreprocessBackend()
    for dx, dy in ((3000., 3000.), (13000., 9000.), (1000., 1000.)):
        oracle(number.ctypes.data, lower.ctypes.data, upper.ctypes.data,
               alt.ctypes.data, expected.ctypes.data, count, 9.81, dx, dy)
        actual = native.aerosol_surface_mass(number, np.stack((lower, upper)), alt, dx, dy)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
