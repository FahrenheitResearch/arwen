"""Cloud-radiation coupling seams: micron radii contract + WRF snow discount.

Seam 1: every microphysics writer must emit radiation-facing effective
radii in MICRONS (state.py, thompson_contract.py).  The Thompson kernel
historically wrote metres, so every cloudy Thompson column radiated at the
RRTMGP clip floors (re_liq 2.5 um, ice diameter 10 um).  These tests pin
the writer to the contract and gate the contract at the radiation boundary
with scheme-agnostic physical-plausibility bands.

Seam 2: WRF v4.6.1's option-4 explicit-snow-radius coupling discounts snow
mass by MIN(0.99, (130/re_s)^2) with re_s floored at 10 um and capped at
130 um, and takes the ice path from cloud ice only
(module_ra_rrtmg_lw.F:12242,12500-12532; module_ra_rrtmg_sw.F:10824,
11040-11067).  The FP32 expression is pinned bitwise by a fixture built
from the unmodified WRF statements (tools/wrf_rrtmg_snow_probe), and its
selection is bound to the wrf_rrtmg_compatibility receipt token.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

FIXTURE = Path(__file__).parent / "data" / "wrf_rrtmg_snow_discount_fixture.csv"

f32 = np.float32


def _from_bits(hex_word):
    return np.uint32(int(hex_word, 16)).view(f32)


def _wrf_snow_discount_replica(re_s_um):
    """FP32 replica of the WRF coupling in WRF's operation order.

    Input is the micron state value (WRF's ``re_snow*1.E6``).  Returns
    (snow_mass_factor, capped_resnow_um) exactly as
    module_ra_rrtmg_lw.F:12242,12515-12528 compute them.
    """
    floored = max(f32(10.0), f32(re_s_um))
    if floored > f32(130.0):
        quotient = f32(f32(130.0) / floored)
        return min(f32(0.99), f32(quotient * quotient)), f32(130.0)
    return f32(0.99), floored


def _fixture_rows():
    with FIXTURE.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------------------
# Seam 2: WRF-side fixture, bitwise (max_ulp 0).


def test_wrf_snow_discount_fixture_provenance_and_coverage():
    rows = _fixture_rows()
    assert len(rows) >= 2000
    resnow = np.array([_from_bits(row["resnow_floored_um"]) for row in rows])
    factor = np.array([_from_bits(row["snow_mass_factor"]) for row in rows])
    # Both predicate branches are exercised, including the MIN crossover
    # window 130 < re_s < 130/sqrt(0.99) where (130/re_s)^2 > 0.99.
    assert np.any(resnow <= 130.0)
    assert np.any(resnow > 130.0)
    assert np.any((resnow > 130.0) & (factor == f32(0.99)))
    assert np.any(factor < f32(0.99))
    # Physical range endpoints of the Thompson snow radius are present.
    assert resnow.min() == f32(10.0)
    assert resnow.max() > 900.0
    assert factor.min() == pytest.approx((130.0 / 999.0) ** 2, rel=1e-5)


def test_wrf_snow_discount_expression_matches_fixture_bitwise():
    mismatches = 0
    for row in _fixture_rows():
        re_m = _from_bits(row["re_snow_m"])
        micron_state = f32(re_m * f32(1.0e6))  # the writer's conversion
        floored = max(f32(10.0), micron_state)
        factor, capped = _wrf_snow_discount_replica(micron_state)
        for mine, name in ((floored, "resnow_floored_um"),
                           (factor, "snow_mass_factor"),
                           (capped, "resnow_final_um")):
            if mine.view(np.uint32) != _from_bits(row[name]).view(np.uint32):
                mismatches += 1
    assert mismatches == 0  # max_ulp 0 against the compiled WRF statements


def test_wrf_snow_discount_fixture_documents_dead_ice_increment():
    """WRF adds 1% of snow to a scalar it never stores (dead in WRF too).

    module_ra_rrtmg_lw.F:12518 / _sw.F:11058 increment ``gicewp`` after its
    last consumer (cicewp, _lw.F:12503 / _sw.F:11043).  The fixture records
    the increment to document that gpuwm intentionally does not reproduce
    it; this test pins the recorded value to WRF's left-to-right FP32
    arithmetic so the fixture row provenance stays auditable.
    """
    # The fixture stores outputs only; reconstruct the increment for the
    # three fixed (qs, pdel) layer states the probe deck cycles through
    # (tools/wrf_rrtmg_snow_probe/make_inputs.py, outer combo loop).
    layer_states = ((f32(2.5e-3), f32(15.0)), (f32(1.0e-4), f32(5.0)),
                    (f32(8.0e-3), f32(30.0)))
    rows = _fixture_rows()
    per_state = len(rows) // len(layer_states)
    assert per_state * len(layer_states) == len(rows)
    gravmks = f32(9.81)
    checked = 0
    for index, row in enumerate(rows):
        qs, pdel = layer_states[index // per_state]
        one_minus = f32(f32(1.0) - f32(0.99))
        expected = f32(f32(f32(f32(f32(qs * one_minus) * pdel) * f32(100.0))
                           / gravmks) * f32(1000.0))
        observed = _from_bits(row["gicewp_dead_increment"])
        assert observed.view(np.uint32) == expected.view(np.uint32)
        checked += 1
    assert checked == len(rows)


# ---------------------------------------------------------------------------
# Seam 2: adapter behavior and token binding.


def test_snow_treatment_token_mapping_is_versioned_and_fail_closed():
    from gpuwm.core.rrtmgp import (
        SNOW_TREATMENT_FULL_MASS, SNOW_TREATMENT_WRF_DISCOUNT,
        snow_treatment_for_compatibility)
    from gpuwm.physics_compat import (
        WRF_RRTMG_COMPATIBILITY_TOKENS,
        WRF_RRTMG_LEGACY,
        WRF_RRTMG_TO_RTE_RRTMGP,
        WRF_RRTMG_TO_RTE_RRTMGP_V1)

    # The receipt token was bumped for the snow-coupling change; -v1 runs
    # are never relabeled and stay selectable.  The assembled lineage also
    # honors the legacy-RRTMG receipt (a different algorithm, no -v2), so
    # the honored-token tuple is the three-token union.
    assert WRF_RRTMG_TO_RTE_RRTMGP == "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
    assert WRF_RRTMG_TO_RTE_RRTMGP_V1 == "wrf-rrtmg-4-4-to-rte-rrtmgp-v1"
    assert WRF_RRTMG_LEGACY == "wrf-rrtmg-4-4-legacy-v1"
    assert WRF_RRTMG_COMPATIBILITY_TOKENS == (
        WRF_RRTMG_TO_RTE_RRTMGP_V1, WRF_RRTMG_TO_RTE_RRTMGP,
        WRF_RRTMG_LEGACY)

    assert snow_treatment_for_compatibility("none") \
        == SNOW_TREATMENT_FULL_MASS
    assert snow_treatment_for_compatibility(WRF_RRTMG_TO_RTE_RRTMGP_V1) \
        == SNOW_TREATMENT_FULL_MASS
    assert snow_treatment_for_compatibility(WRF_RRTMG_TO_RTE_RRTMGP) \
        == SNOW_TREATMENT_WRF_DISCOUNT
    with pytest.raises(ValueError, match="unknown wrf_rrtmg_compatibility"):
        snow_treatment_for_compatibility("wrf-rrtmg-4-4-to-rte-rrtmgp-v3")
    with pytest.raises(ValueError, match="unknown wrf_rrtmg_compatibility"):
        snow_treatment_for_compatibility("")


def test_run_config_accepts_both_tokens_and_fails_closed_on_unknown():
    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.physics_compat import (
        WRF_RRTMG_TO_RTE_RRTMGP, WRF_RRTMG_TO_RTE_RRTMGP_V1)

    def cfg(**overrides):
        values = dict(nx=4, ny=3, nz=4, dx=1000.0, dy=1000.0,
                      ztop=10000.0, dt=5.0, run_seconds=10.0)
        values.update(overrides)
        return RunConfig(**values)

    native = validate_run_config(cfg(ra_physics=4))
    assert native.wrf_rrtmg_compatibility == "none"
    for token in (WRF_RRTMG_TO_RTE_RRTMGP_V1, WRF_RRTMG_TO_RTE_RRTMGP):
        accepted = validate_run_config(
            replace(native, wrf_rrtmg_compatibility=token))
        assert accepted.wrf_rrtmg_compatibility == token
    with pytest.raises(ValueError, match="wrf_rrtmg_compatibility must"):
        validate_run_config(replace(
            native, wrf_rrtmg_compatibility="wrf-rrtmg-4-4-to-rte-rrtmgp-v3"))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("scheme", ["wsm6", "thompson"])
@pytest.mark.parametrize("with_cldfra", [False, True])
def test_hydrometeor_paths_wrf_snow_discount(scheme, with_cldfra):
    import cupy as cp
    from gpuwm.core.rrtmgp import (
        SNOW_TREATMENT_FULL_MASS, SNOW_TREATMENT_WRF_DISCOUNT,
        hydrometeor_paths)

    plev = cp.asarray([[100000.0, 90000.0, 80000.0, 70000.0, 60000.0]] * 2,
                      dtype=cp.float32)
    qc = cp.asarray([[1.0e-3, 0.0, 0.0, 2.0e-4],
                     [0.0, 0.0, 0.0, 0.0]], dtype=cp.float32)
    qr = cp.zeros_like(qc)
    qi = cp.asarray([[0.0, 2.0e-4, 1.0e-4, 0.0],
                     [5.0e-5, 0.0, 3.0e-4, 0.0]], dtype=cp.float32)
    qs = cp.asarray([[0.0, 1.0e-3, 0.0, 5.0e-4],
                     [2.0e-3, 0.0, 0.0, 0.0]], dtype=cp.float32)
    effc = cp.full_like(qc, 12.0)
    effi = cp.asarray([[4.99, 40.0, 25.0, 4.99],
                       [60.0, 4.99, 30.0, 4.99]], dtype=cp.float32)
    # Snow radii straddle the floor, the cap, and the MIN crossover.
    effs = cp.asarray([[9.99, 500.0, 9.99, 130.2],
                       [999.0, 9.99, 9.99, 9.99]], dtype=cp.float32)
    cldfra = (cp.asarray([[1.0, 0.6, 0.004, 0.9],
                          [0.5, 0.0, 1.0, 0.0]], dtype=cp.float32)
              if with_cldfra else None)

    common = dict(microphysics=scheme, effc=effc, effi=effi, effs=effs,
                  cldfra=cldfra)
    legacy = hydrometeor_paths(plev, qc, qr, qi, qs,
                               snow_treatment=SNOW_TREATMENT_FULL_MASS,
                               **common)
    wrf = hydrometeor_paths(plev, qc, qr, qi, qs,
                            snow_treatment=SNOW_TREATMENT_WRF_DISCOUNT,
                            **common)

    # NumPy FP32 references in the exact operation order of the adapter.
    np_plev = cp.asnumpy(plev)
    np_qc, np_qi, np_qs = (cp.asnumpy(a) for a in (qc, qi, qs))
    np_effi, np_effs = cp.asnumpy(effi), cp.asnumpy(effs)
    mass_path = np.abs(np.diff(np_plev, axis=1)).astype(f32) \
        * f32(1000.0 / 9.80665)
    incloud = np.maximum(f32(0.01), cp.asnumpy(cldfra)) if with_cldfra \
        else None

    def in_cloud(path):
        return (path / incloud).astype(f32) if with_cldfra else path

    # Legacy: full snow mass merged at native radius (unchanged behavior).
    legacy_ciwp = in_cloud(((np_qi + np_qs) * mass_path).astype(f32))
    np.testing.assert_array_equal(cp.asnumpy(legacy.ciwp), legacy_ciwp)
    frozen = np_qi + np_qs
    legacy_reice = np.where(
        frozen > f32(1.0e-20),
        (np_qi * np_effi + np_qs * np_effs)
        / np.maximum(frozen, f32(1.0e-20)), f32(25.0)).astype(f32)
    np.testing.assert_array_equal(
        cp.asnumpy(legacy.dgice),
        np.clip(f32(2.0) * legacy_reice, f32(10.0), f32(180.0)))

    # WRF discount: floor 10, cap 130, MIN(0.99, (130/re_s)^2), qi-only
    # ice path plus the discounted snow path.
    re_s0 = np.maximum(f32(10.0), np_effs)
    quotient = (f32(130.0) / re_s0).astype(f32)
    factor = np.where(re_s0 > f32(130.0),
                      np.minimum(f32(0.99), (quotient * quotient).astype(f32)),
                      f32(0.99)).astype(f32)
    fixture_factor = np.vectorize(
        lambda value: _wrf_snow_discount_replica(value)[0],
        otypes=[f32])(np_effs)
    np.testing.assert_array_equal(factor, fixture_factor)
    qs_eff = (np_qs * factor).astype(f32)
    wrf_ciwp = in_cloud(((np_qi + qs_eff) * mass_path).astype(f32))
    np.testing.assert_array_equal(cp.asnumpy(wrf.ciwp), wrf_ciwp)
    re_s_eff = np.minimum(re_s0, f32(130.0))
    frozen_eff = np_qi + qs_eff
    wrf_reice = np.where(
        frozen_eff > f32(1.0e-20),
        (np_qi * np_effi + qs_eff * re_s_eff)
        / np.maximum(frozen_eff, f32(1.0e-20)), f32(25.0)).astype(f32)
    np.testing.assert_array_equal(
        cp.asnumpy(wrf.dgice),
        np.clip(f32(2.0) * wrf_reice, f32(10.0), f32(180.0)))

    # Liquid path and radius are identical across treatments.
    np.testing.assert_array_equal(cp.asnumpy(legacy.clwp),
                                  cp.asnumpy(wrf.clwp))
    np.testing.assert_array_equal(cp.asnumpy(legacy.reliq),
                                  cp.asnumpy(wrf.reliq))
    # The discount only removes snow mass: never adds.
    assert np.all(cp.asnumpy(wrf.ciwp) <= cp.asnumpy(legacy.ciwp))

    with pytest.raises(ValueError, match="snow_treatment must be one of"):
        hydrometeor_paths(plev, qc, qr, qi, qs,
                          snow_treatment="wrf-discount-typo", **common)


@requires_gpu
@pytest.mark.gpu
def test_snow_treatment_does_not_alter_morrison_or_kessler():
    import cupy as cp
    from gpuwm.core.rrtmgp import (
        SNOW_TREATMENT_FULL_MASS, SNOW_TREATMENT_WRF_DISCOUNT,
        hydrometeor_paths)

    plev = cp.asarray([[100000.0, 90000.0]], dtype=cp.float32)
    qc = cp.asarray([[1.0e-3]], dtype=cp.float32)
    qi = cp.asarray([[4.0e-4]], dtype=cp.float32)
    qs = cp.asarray([[1.0e-3]], dtype=cp.float32)
    zero = cp.zeros_like(qc)
    for treatment in (SNOW_TREATMENT_FULL_MASS, SNOW_TREATMENT_WRF_DISCOUNT):
        kessler = hydrometeor_paths(plev, qc, zero, qi, qs,
                                    microphysics="kessler",
                                    snow_treatment=treatment)
        # WRF merges snow into ice for schemes without an explicit snow
        # radius (module_ra_rrtmg_lw.F:12488-12493): full mass either way.
        np.testing.assert_allclose(
            float(kessler.ciwp[0, 0]),
            (4.0e-4 + 1.0e-3) * 10000.0 / 9.80665 * 1000.0, rtol=2e-6)
    morrison = {}
    for treatment in (SNOW_TREATMENT_FULL_MASS, SNOW_TREATMENT_WRF_DISCOUNT):
        paths = hydrometeor_paths(
            plev, qc, zero, qi, qs, microphysics="morrison",
            play=cp.asarray([[95000.0]], dtype=cp.float32),
            tlay=cp.asarray([[260.0]], dtype=cp.float32),
            nc=cp.asarray([[8.0e7]], dtype=cp.float32),
            nr=zero, ni=cp.asarray([[5.0e5]], dtype=cp.float32),
            ns=cp.asarray([[1.0e4]], dtype=cp.float32),
            snow_treatment=treatment)
        morrison[treatment] = (cp.asnumpy(paths.ciwp),
                               cp.asnumpy(paths.dgice))
    np.testing.assert_array_equal(
        morrison[SNOW_TREATMENT_FULL_MASS][0],
        morrison[SNOW_TREATMENT_WRF_DISCOUNT][0])
    np.testing.assert_array_equal(
        morrison[SNOW_TREATMENT_FULL_MASS][1],
        morrison[SNOW_TREATMENT_WRF_DISCOUNT][1])


# ---------------------------------------------------------------------------
# Seam 1: writers honor the micron contract; the boundary gate enforces it.


@requires_gpu
@pytest.mark.gpu
def test_thompson_effective_radius_kernel_writes_microns():
    import cupy as cp
    from gpuwm.core.thompson import launch_effective_radius

    n = 6
    temperature = cp.asarray(
        [285.0, 260.0, 250.0, 240.0, 285.0, 230.0], dtype=cp.float32)
    pressure = cp.asarray(
        [90000.0, 60000.0, 50000.0, 40000.0, 90000.0, 30000.0],
        dtype=cp.float32)
    qv = cp.full((n,), 2.0e-3, dtype=cp.float32)
    qc = cp.asarray([1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=cp.float32)
    qi = cp.asarray([0.0, 2.0e-4, 1.0e-4, 0.0, 0.0, 0.0], dtype=cp.float32)
    ni = cp.asarray([0.0, 5.0e5, 1.0e4, 0.0, 0.0, 0.0], dtype=cp.float32)
    qs = cp.asarray([0.0, 0.0, 1.0e-3, 2.0e-3, 0.0, 0.0], dtype=cp.float32)
    effc = cp.zeros((n,), dtype=cp.float32)
    effi = cp.zeros((n,), dtype=cp.float32)
    effs = cp.zeros((n,), dtype=cp.float32)
    launch_effective_radius(
        temperature, pressure, qv, qc, qi, ni, qs, effc, effi, effs)
    out = {name: cp.asnumpy(value)
           for name, value in (("effc", effc), ("effi", effi),
                               ("effs", effs))}

    # Background cells carry the micron background values: exactly the
    # FP32 metre constants times the FP32 1e6 conversion.
    background = {name: f32(f32(metres) * f32(1.0e6)) for name, metres in
                  (("effc", 2.49e-6), ("effi", 4.99e-6), ("effs", 9.99e-6))}
    for name in ("effc", "effi", "effs"):
        assert out[name][4] == background[name]
        assert out[name][5] == background[name]
    # Diagnosed values live inside WRF's clamp ranges, in microns.  The
    # bounds are the FP32 metre clamps times the FP32 conversion (e.g.
    # fl(999e-6)*fl(1e6) lands one ULP above decimal 999.0).
    def band(lo_m, hi_m):
        return (f32(f32(lo_m) * f32(1.0e6)), f32(f32(hi_m) * f32(1.0e6)))

    lo, hi = band(2.49e-6, 50.0e-6)
    assert lo <= out["effc"][0] <= hi
    lo, hi = band(4.99e-6, 125.0e-6)
    assert all(lo <= value <= hi for value in out["effi"][1:3])
    lo, hi = band(9.99e-6, 999.0e-6)
    assert all(lo <= value <= hi for value in out["effs"][2:4])
    # And the cloudy diagnostics moved off the background (physical PSDs).
    assert out["effc"][0] > 2.51
    assert out["effs"][3] > 25.0


@requires_gpu
@pytest.mark.gpu
def test_hydrometeor_paths_gate_rejects_out_of_unit_radii():
    import cupy as cp
    from gpuwm.core.rrtmgp import hydrometeor_paths

    plev = cp.asarray([[100000.0, 90000.0]], dtype=cp.float32)
    qc = cp.asarray([[1.0e-3]], dtype=cp.float32)
    zero = cp.zeros_like(qc)
    good = dict(effc=cp.asarray([[12.0]], dtype=cp.float32),
                effi=cp.asarray([[30.0]], dtype=cp.float32),
                effs=cp.asarray([[80.0]], dtype=cp.float32))
    paths = hydrometeor_paths(plev, qc, zero, zero, zero,
                              microphysics="thompson", **good)
    assert float(paths.reliq[0, 0]) == 12.0

    for name, bad in (("effc", 2.49e-6), ("effi", 4.99e-6),
                      ("effs", 9.99e-6), ("effs", 999000.0)):
        radii = dict(good)
        radii[name] = cp.asarray([[bad]], dtype=cp.float32)
        with pytest.raises(ValueError, match="physical-plausibility band"):
            hydrometeor_paths(plev, qc, zero, zero, zero,
                              microphysics="thompson", **radii)


def test_plausibility_bands_trap_every_metric_prefix():
    from gpuwm.core.rrtmgp import EFFECTIVE_RADIUS_PLAUSIBLE_UM

    # Writer extreme ranges (microns) across scheme families: WSM6/
    # Thompson clamps and Morrison lambda bounds (cloud minimum is
    # (pgam+3)/(2(pgam+1)) = 0.59 um at pgam=10).
    clamp = {"effc": (0.59, 50.0), "effi": (1.5, 525.0),
             "effs": (9.99, 3000.0)}
    # Background fills used by the writers for clear cells
    # (module_model_constants metre constants converted, Morrison's 25).
    backgrounds = {"effc": (2.49, 2.5, 25.0), "effi": (4.99, 5.0, 25.0),
                   "effs": (9.99, 10.0, 25.0)}
    for name, (lower, upper) in EFFECTIVE_RADIUS_PLAUSIBLE_UM.items():
        lo, hi = clamp[name]
        assert lower <= lo and hi <= upper  # legitimate writers pass
        # Every field is background-filled in clear cells, and any
        # metric-prefix error (factor >= 1000 either way) pushes every
        # background out of its band -- so a wrong-unit writer trips the
        # gate on the first radiation call regardless of cloud state.
        for background in backgrounds[name]:
            assert background / 1000.0 < lower   # metres/millimetres trip
            assert background * 1000.0 > upper   # nanometre-style trips
        # Metre-scale emissions of even the LARGEST legitimate value trip
        # the lower bound outright.
        assert hi / 1.0e6 < lower
