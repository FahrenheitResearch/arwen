"""The vectorized RUC cold start equals the per-column transcription it replaced.

Named breakage: ``ruc_initialize_cold_start`` evaluated ruclsminit one column
and one soil level at a time with a Python ``logf`` per level; on a 1 km
HRRR-physics start (1300x1300, box G, 2026-10-05) that was about 75 s of
main-thread time per pass, before the first model step.  The array form must
keep every word: this file carries the replaced loop verbatim as the oracle
and compares bit patterns on fields that exercise every branch (sea ice,
water soil, frozen and thawed levels, availability floors and caps), and pins
the native glibc 2.39 ``logf`` array entry against the scalar transcription.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core import noahmp_libm
from gpuwm.core.ruc import load_ruc_parameters, ruc_initialize_cold_start


def _reference(tslb, smois, isltyp, ivgtyp, xice,
               mminlu="MODIFIED_IGBP_MODIS_NOAH"):
    """The per-column loop as it stood at 534c6dea3 (gpuwm/core/ruc.py)."""
    _glibc_logf = noahmp_libm.logf
    _glibc_powf = noahmp_libm.powf
    bundle = load_ruc_parameters()
    vegetation = bundle.vegetation_for(mminlu)
    temperature = np.asarray(tslb, dtype=np.float32)
    total_water = np.asarray(smois, dtype=np.float32)
    nzs = temperature.shape[0]
    horizontal_shape = temperature.shape[1:]
    liquid = np.empty_like(total_water)
    frozen = np.empty_like(total_water)
    availability = np.empty(horizontal_shape, dtype=np.float32)
    roughness = np.empty(horizontal_shape, dtype=np.float32)
    t_flat = temperature.reshape(nzs, -1)
    w_flat = total_water.reshape(nzs, -1)
    liquid_flat = liquid.reshape(nzs, -1)
    frozen_flat = frozen.reshape(nzs, -1)
    soil_flat = np.asarray(isltyp).reshape(-1)
    vegetation_flat = np.asarray(ivgtyp).reshape(-1)
    ice_flat = np.asarray(xice, dtype=np.float32).reshape(-1)
    availability_flat = availability.reshape(-1)
    roughness_flat = roughness.reshape(-1)
    riw = np.float32(np.float32(900.0) * np.float32(1.0e-3))
    xlmelt = np.float32(3.35e5)
    t_freeze = np.float32(273.15)
    gravity = np.float32(9.81)
    one = np.float32(1.0)
    minimum_availability = np.float32(0.00001)

    for column in range(soil_flat.size):
        soil = bundle.soil.rows[int(soil_flat[column]) - 1]
        veg = vegetation.rows[int(vegetation_flat[column]) - 1]
        bb = np.float32(soil.values[0])
        drysmc = np.float32(soil.values[1])
        maxsmc = np.float32(soil.values[3])
        refsmc = np.float32(soil.values[4])
        satpsi = np.float32(soil.values[5])
        dqm = np.float32(maxsmc - drysmc)
        psis = np.float32(-satpsi)
        roughness_flat[column] = np.float32(veg.z0)

        if ice_flat[column] > np.float32(0.0):
            frozen_flat[:, column] = one
            liquid_flat[:, column] = np.float32(0.0)
            availability_flat[column] = one
            continue
        if soil_flat[column] == 14:
            frozen_flat[:, column] = np.float32(0.0)
            liquid_flat[:, column] = one
            availability_flat[column] = one
            continue

        numerator = np.float32(w_flat[0, column] - drysmc)
        denominator = np.float32(refsmc - drysmc)
        raw_availability = np.float32(numerator / denominator)
        availability_flat[column] = np.float32(
            max(minimum_availability, min(one, raw_availability))
        )
        for level in range(nzs):
            soil_temperature = t_flat[level, column]
            # LOG and ** on REAL are glibc's logf and powf in WRF's build,
            # and are taken from their transcriptions here: NumPy's float32
            # log and power are the host's (NumPy 2.5's AVX-512 loops on an
            # AVX-512 Linux machine), so SH2O and SMFR3D at cold start moved
            # with the CPU that constructed the driver.
            tln = _glibc_logf(np.float32(soil_temperature / t_freeze))
            if tln < np.float32(0.0):
                base = np.float32(xlmelt * np.float32(soil_temperature - t_freeze))
                base = np.float32(base / soil_temperature)
                base = np.float32(base / gravity)
                base = np.float32(base / psis)
                exponent = np.float32(np.float32(-one) / bb)
                equilibrium = np.float32(
                    np.float32(dqm + drysmc)
                    * _glibc_powf(base, exponent)
                )
                equilibrium = np.float32(max(np.float32(0.0), equilibrium))
                equilibrium = np.float32(
                    min(equilibrium, w_flat[level, column])
                )
                liquid_flat[level, column] = equilibrium
                frozen_flat[level, column] = np.float32(
                    np.float32(w_flat[level, column] - equilibrium) / riw
                )
            else:
                frozen_flat[level, column] = np.float32(0.0)
                liquid_flat[level, column] = w_flat[level, column]


    return liquid, frozen, availability, roughness


def _fields(seed, nzs=9, ny=23, nx=31):
    rng = np.random.default_rng(seed)
    bundle = load_ruc_parameters()
    nsoil = len(bundle.soil.rows)
    nveg = len(bundle.vegetation_for("MODIFIED_IGBP_MODIS_NOAH").rows)
    tslb = rng.uniform(240.0, 300.0, (nzs, ny, nx)).astype(np.float32)
    tslb[:, ::5, :] = np.float32(273.15)              # the freezing point itself
    tslb[:, 1::7, ::3] = np.nextafter(np.float32(273.15), np.float32(0.0))
    smois = rng.uniform(0.0, 0.5, (nzs, ny, nx)).astype(np.float32)
    smois[0, ::4, ::4] = np.float32(0.0)              # availability floor
    smois[0, 2::6, 1::5] = np.float32(1.0)            # availability cap
    isltyp = rng.integers(1, nsoil + 1, (ny, nx)).astype(np.int32)
    isltyp[::3, ::4] = 14                             # water soil
    ivgtyp = rng.integers(1, nveg + 1, (ny, nx)).astype(np.int32)
    xice = np.zeros((ny, nx), np.float32)
    xice[::5, ::6] = rng.uniform(0.1, 1.0, xice[::5, ::6].shape)
    xice[0, 0] = np.float32(1.0)
    return tslb, smois, isltyp, ivgtyp, xice


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def test_vectorized_cold_start_matches_the_column_loop_bit_for_bit():
    for seed in (20261005, 7, 11):
        fields = _fields(seed)
        got = ruc_initialize_cold_start(*fields)
        want = _reference(*fields)
        for name, a, b in zip(("sh2o", "smfr3d", "mavail", "znt"),
                              (got.sh2o, got.smfr3d, got.mavail, got.znt),
                              want):
            assert a.shape == b.shape, name
            mismatch = np.flatnonzero(_words(a) != _words(b))
            assert mismatch.size == 0, (
                f"{name}: {mismatch.size} words differ from the column loop "
                f"(seed {seed}, first flat index {mismatch[:3]})")
        # the fixture really reaches the frozen branch
        assert np.any(got.smfr3d[:, fields[4] == 0] > 0), seed


def test_native_logf_array_matches_the_scalar_transcription():
    rng = np.random.default_rng(3)
    probes = [
        rng.uniform(0.6, 1.2, 200_000),
        rng.uniform(1e-6, 1e6, 50_000),
        np.array([1.0, 2.0, 0.5, 0.0, -0.0, -1.0, np.inf, np.nan,
                  1e-45, 1.1754942e-38, 3.4028235e38]),
    ]
    values = np.concatenate(probes).astype(np.float32)
    got = noahmp_libm.logf_array(values)
    want = np.array([noahmp_libm.logf(v) for v in values], dtype=np.float32)
    nan = np.isnan(want)
    assert np.array_equal(np.isnan(got), nan)
    assert np.array_equal(_words(got)[~nan], _words(want)[~nan])
