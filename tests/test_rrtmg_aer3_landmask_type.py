"""aer_opt = 3 land/water aerosol type (rrtmg_aer3_type_source = "landmask").

THE BREAKAGE THIS PREVENTS
--------------------------
NOAA's RRFS v2 runs the MPAS fork RRFSx/MPAS-Model 1d6e370e, whose
shortwave driver gives every water column the maritime Shettle-Fenn optics
(taer_type 3 where xland == 2, mpas_atmphys_driver_radiation_sw.F:583-586).
The engine's aer_opt = 3 optics were the HRRR WRF fork's, rural type 1
everywhere, so an RRFS-shaped hex run put continental aerosol optics over
every ocean, lake and coast: on the oracle deck below that is up to several
times the band optical depth at low humidity and a different single-
scattering albedo and asymmetry in every band.

The oracle is that fork's own Fortran (gt_aod, calc_aerosol_rrtmg_sw and
RSLF compiled verbatim at -O0 without contraction, called the way the
driver calls them; tools/rrfs_mpas_aer3_oracle), stored in
tests/data/aer3_mpas_oracle/aer3_mpas_oracle.npz: 40 profiles, each once
over land and once over water, 30 layers, 14 bands.  The deck reaches every
gt_aod branch, both RH clamps, every temperature index and the number caps.

CPU only: this module imports no device (tests/conftest.py would mark and
skip the whole module on a CPU runner otherwise).  The kernel's word-for-
word check is tests/test_rrtmg_aer3_landmask_type_gpu.py.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import rrtmg_aerosol_optics as ao

F = np.float32
REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "aer3_mpas_oracle" / "aer3_mpas_oracle.npz"
FIXTURE_SHA256 = REPO / "tests" / "data" / "aer3_mpas_oracle" / "SHA256"
KERNEL = REPO / "gpuwm" / "core" / "kernels" / "rrtmg_aer3.cu"
OUTS = ("tauaer", "ssaaer", "asyaer")
INPUTS = ("p", "t", "qv", "dz8w", "nwfa", "nifa")

#: The rural identity every WRF and HRRR aer_opt = 3 run has carried since
#: the port; the land/water type must not move it.
RURAL_TABLES_SHA256 = (
    "9962257443d94d4e71341d29165dbb40ece68b61f21b92c8b83c520cf03ae2b8")


def bits_equal(name, got, want):
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    differ = got.view(np.uint32) != want.view(np.uint32)
    assert not differ.any(), (
        f"{name}: {int(differ.sum())} of {got.size} words differ; first "
        f"got {got[differ][:4]} want {want[differ][:4]}")
    return got.size


@pytest.fixture(scope="module")
def oracle():
    want = FIXTURE_SHA256.read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert got == want, f"aer3 MPAS oracle fixture sha256 {got} != pinned {want}"
    data = np.load(FIXTURE)
    receipt = json.loads(str(data["receipt"]))
    assert receipt["repository"] == "RRFSx/MPAS-Model"
    assert receipt["commit"] == "1d6e370e7b43a332074809f170532a216c2212fb"
    assert receipt["sources"]["physics_wrf/module_ra_rrtmg_sw_aerosols.F"] == (
        "4985e49f8225fdca6a40ae253cfe4ed2dd86c3fc1a46469cb3369fd1fb972cfd")
    assert receipt["sources"]["physics_wrf/module_mp_thompson_aerosols.F"] == (
        "ffdd347c5396d7fdcf2784c6c3fd7c453008afd31ab297414c18e5a402b4d248")
    assert receipt["sources"]["mpas_atmphys_driver_radiation_sw.F"] == (
        "cf27cfa597771c3184c0b46fd4b7d7edcedcedbd966a3ed2dd82bc326ee5aa56")
    assert "-O0" in receipt["flags"] and "-ffp-contract=off" in receipt["flags"]
    return {k: data[k] for k in data.files if k != "receipt"}


def _inputs(o):
    return [o[f"in/{k}"] for k in INPUTS]


# --- tables ---------------------------------------------------------------
def test_the_rural_identity_is_unchanged():
    assert ao.tables_sha256() == ao.AER3_TABLES_SHA256 == RURAL_TABLES_SHA256


def test_the_maritime_hash_is_pinned():
    assert ao.maritime_tables_sha256() == ao.AER3_MARITIME_TABLES_SHA256


def test_both_types_are_the_words_of_the_pinned_noaa_source(oracle):
    o = oracle
    for name, rural, maritime in (
            ("raod_lut", ao.RAOD_RURAL, ao.RAOD_MARITIME),
            ("ssa_lut", ao.SSA_RURAL, ao.SSA_MARITIME),
            ("asy_lut", ao.ASY_RURAL, ao.ASY_MARITIME)):
        bits_equal(f"{name} rural", rural, o[f"src/{name}/rural"])
        bits_equal(f"{name} maritime", maritime, o[f"src/{name}/maritime"])


def _cuda_table(name):
    text = KERNEL.read_text(encoding="utf-8")
    match = re.search(
        rf"__device__ const float {name}\[AER3_NRH\]\[AER3_NB\] = \{{(.*?)\}};",
        text, re.S)
    assert match, name
    values = [F(v.rstrip("f")) for v in re.findall(r"[0-9.]+f", match.group(1))]
    return np.array(values, F).reshape(8, 14)


def test_the_kernel_declares_the_same_maritime_and_rural_words():
    for name, table in (("AER3_RAOD_MAR", ao.RAOD_MARITIME),
                        ("AER3_SSA_MAR", ao.SSA_MARITIME),
                        ("AER3_ASY_MAR", ao.ASY_MARITIME),
                        ("AER3_RAOD", ao.RAOD_RURAL),
                        ("AER3_SSA", ao.SSA_RURAL),
                        ("AER3_ASY", ao.ASY_RURAL)):
        bits_equal(name, _cuda_table(name), table)


# --- the driver's type ----------------------------------------------------
def test_the_type_is_the_noaa_drivers(oracle):
    o = oracle
    got = ao.aer_type_from_xland(o["in/xland"])
    assert got.dtype == np.int32
    assert np.array_equal(got, o["out/aer_type"])
    assert set(np.unique(got)) == {1, 3}


@pytest.mark.parametrize("xland", [0.0, 1.5, 3.0, np.nan])
def test_an_xland_the_driver_assigns_no_type_is_refused(xland):
    with pytest.raises(ValueError, match="assigns no aerosol type"):
        ao.aer_type_from_xland(np.array([1.0, xland, 2.0], F))


def test_a_type_other_than_rural_or_maritime_is_refused(oracle):
    o = oracle
    ncol = o["in/p"].shape[0]
    with pytest.raises(ValueError, match="1 \\(rural\\) and 3"):
        ao.aer3_sw_optics(*_inputs(o), aer_type=np.full(ncol, 2, np.int32))
    with pytest.raises(ValueError, match="shape"):
        ao.aer3_sw_optics(*_inputs(o), aer_type=np.ones(ncol + 1, np.int32))


def test_smoke_with_the_land_water_type_is_refused(oracle):
    o = oracle
    ncol = o["in/p"].shape[0]
    with pytest.raises(ValueError, match="no smoke coupling"):
        ao.aer3_sw_optics(*_inputs(o), aer_type=np.ones(ncol, np.int32),
                          smoke_aod=np.zeros_like(o["in/p"]),
                          smoke_feedback=True)


# --- the twin against the NOAA fork --------------------------------------
def test_twin_matches_the_noaa_mpas_fork_word_for_word(oracle):
    o = oracle
    aer_type = ao.aer_type_from_xland(o["in/xland"])
    tau, ssa, asy, taod = ao.aer3_sw_optics(*_inputs(o), aer_type=aer_type)
    words = 0
    for name, got in zip(OUTS, (tau, ssa, asy)):
        words += bits_equal(name, got, o[f"out/{name}"])
    words += bits_equal("taod5503d", taod, o["out/taod5503d"])
    col = np.zeros(taod.shape[0], np.float32)
    for k in range(taod.shape[1]):
        col = (col + taod[:, k]).astype(np.float32)
    words += bits_equal("taod5502d", col, o["out/taod5502d"])
    print(f"aer3 landmask twin vs NOAA MPAS fork: {words} words, max ULP 0")


def test_land_columns_are_the_rural_optics_word_for_word(oracle):
    """A type-1 column is the HRRR fork's rural optics: the land half of the
    NOAA deck equals the untyped call, so every rural configuration runs the
    words it always ran."""
    o = oracle
    aer_type = ao.aer_type_from_xland(o["in/xland"])
    typed = ao.aer3_sw_optics(*_inputs(o), aer_type=aer_type)
    rural = ao.aer3_sw_optics(*_inputs(o))
    land = aer_type == 1
    for name, got, want in zip(OUTS + ("taod5503d",), typed, rural):
        bits_equal(f"land {name}", got[land], want[land])
    bits_equal("all-rural typed", ao.aer3_sw_optics(
        *_inputs(o), aer_type=np.ones_like(aer_type))[0], rural[0])


def test_water_columns_move_off_the_rural_optics(oracle):
    """The defect this closes is visible on the deck: the same air over water
    gets different optics in every band."""
    o = oracle
    m = o["in/p"].shape[0] // 2
    tau = o["out/tauaer"]
    ssa = o["out/ssaaer"]
    assert np.array_equal(o["in/p"][:m], o["in/p"][m:])
    assert np.all(tau[m:] != tau[:m]) or np.mean(tau[m:] != tau[:m]) > 0.99
    assert np.all(ssa[m:] != ssa[:m]) or np.mean(ssa[m:] != ssa[:m]) > 0.99
    # gt_aod and the column AOD do not depend on the type
    bits_equal("taod5503d land vs water", o["out/taod5503d"][m:],
               o["out/taod5503d"][:m])


def test_the_maritime_optics_keep_the_sw_division_numerator_normal():
    """The rural range argument (tests/test_rrtmg_aerosol_optics.py) holds a
    fortiori: every maritime Lagrange factor floor sits above the rural one
    over the whole clamped RH range."""
    rh = np.concatenate([
        np.linspace(0.0, 99.0, 99001), ao.SPEC_RHS,
        np.nextafter(ao.SPEC_RHS, F(200.0)),
        np.nextafter(ao.SPEC_RHS, F(-1.0))]).astype(np.float32)
    rh = np.clip(rh, F(0.0), F(99.0))
    raod = float(ao._lagrange(rh, ao.RAOD_MARITIME).min())
    ssa = float(ao._lagrange(rh, ao.SSA_MARITIME).min())
    asy = float(ao._lagrange(rh, ao.ASY_MARITIME).min())
    assert raod >= 0.27 and ssa >= 0.69 and asy >= 0.66, (raod, ssa, asy)
    assert raod >= 0.058 and ssa >= 0.55 and asy >= 0.61


# --- configuration --------------------------------------------------------
def _cfg(**kw):
    from gpuwm.config import RunConfig
    base = dict(nx=6, ny=4, nz=5, dx=3000.0, dy=3000.0, ztop=20000.0,
                dt=20.0, run_seconds=0.0)
    base.update(kw)
    return RunConfig(**base)


_LEGACY = dict(mp_physics=28, ra_physics=4, ra_rrtmg_variant="rrtmg_legacy")


def test_the_default_is_rural_and_passes():
    from gpuwm.config import validate_radiation_driver_options
    cfg = _cfg()
    assert cfg.rrtmg_aer3_type_source == "rural"
    validate_radiation_driver_options(cfg)
    validate_radiation_driver_options(
        _cfg(aer_opt=3, rrtmg_aer3_type_source="landmask", **_LEGACY))


@pytest.mark.parametrize("overrides,match", [
    (dict(rrtmg_aer3_type_source="urban"), "must be 'rural'"),
    (dict(rrtmg_aer3_type_source="landmask"), "never read"),
    (dict(rrtmg_aer3_type_source="landmask", aer_opt=0, **_LEGACY), "never read"),
    (dict(rrtmg_aer3_type_source="landmask", aer_opt=3, mp_physics=28,
          ra_physics=4, ra_rrtmg_variant="rte-rrtmgp"), "legacy"),
])
def test_refusals_name_the_breakage(overrides, match):
    from gpuwm.config import validate_radiation_driver_options
    with pytest.raises(ValueError, match=match):
        validate_radiation_driver_options(_cfg(**overrides))


def test_the_column_suite_admits_it_only_on_legacy_rrtmg():
    from gpuwm.core import mpas_column_suite as suite
    row = suite.GENERATION_SELECTORS["rrtmg_aer3_type_source"]
    assert row["values"] == ("rural", "landmask")
    assert row["requires"] == ("radiation_scheme", ("rrtmg_legacy",))


def test_the_restart_echo_binds_only_the_moved_value():
    from gpuwm.io import restart
    assert "rrtmg_aer3_type_source" not in restart.configuration_echo(
        _cfg(aer_opt=3, **_LEGACY))
    echo = restart.configuration_echo(
        _cfg(aer_opt=3, rrtmg_aer3_type_source="landmask", **_LEGACY))
    assert echo["rrtmg_aer3_type_source"] == "landmask"


@pytest.mark.parametrize("kwargs,match", [
    (dict(aer_opt=0, aer3_type_source="landmask"), "never be read"),
    (dict(aer_opt=3, aer3_type_source="urban"), "must be one of"),
])
def test_the_adapter_refuses_a_type_it_would_not_read(kwargs, match):
    """Constructor validation runs before any device work."""
    from datetime import datetime
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    lat = np.zeros((1, 2), np.float32)
    with pytest.raises(ValueError, match=match):
        RRTMGLegacyRadiation(datetime(2026, 10, 3, 12), lat, lat, **kwargs)
