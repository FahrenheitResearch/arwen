"""Kain--Fritsch cumulus physics (WRF ``cu_physics=1``).

The CPU mirror is the numerical authority.  These tests pin the KF lookup
table, trigger/no-trigger behavior, 90 % CAPE closure, finite bounded
tendencies, precipitation, and the one-thread-per-column CUDA port.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


_REAL74_COLUMNS = Path(__file__).parent / "data" / "kf_real74_12z_columns.npz"


def _sounding(*, unstable: bool, nz: int = 49):
    """A hydrostatic Great-Plains-like pressure-level column."""
    z_ifc = 16000.0 * np.linspace(0.0, 1.0, nz + 1) ** 1.18
    z = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    dz = np.diff(z_ifc)
    pressure = 98000.0 * np.exp(-z / 8100.0)
    exner = (pressure / 100000.0) ** (287.0 / 1004.5)
    if unstable:
        # Deep but non-limiter branch: the 1.6-km moisture transition forces
        # mixed-phase convection while retaining exact WRF water closure.
        temperature = np.maximum(293.0 - 0.0070 * z, 205.0)
        qv = (0.014 * np.exp(-z / 3500.0)
              / (1.0 + np.exp((z - 1600.0) / 150.0))
              + 0.001 * np.exp(-z / 5000.0))
        u = 8.0 + 0.0020 * z
        w = np.where(z < 2500.0, 0.01, 0.0)
    else:
        temperature = np.maximum(276.0 - 0.0040 * z, 215.0)
        temperature += 5.0 * np.exp(-((z - 900.0) / 650.0) ** 2)
        qv = 0.0032 * np.exp(-z / 1500.0)
        u = 12.0 + 0.0008 * z
        w = np.full(nz, -0.03)
    return dict(
        u=u, v=np.zeros(nz), temperature=temperature, qv=qv,
        qc=np.zeros(nz), pressure=pressure, exner=exner, dz=dz,
        w=w, dx=12000.0, dt=60.0, cudt=300.0,
    )


def _real74_sounding(label: str):
    with np.load(_REAL74_COLUMNS, allow_pickle=False) as data:
        result = {name: data[f"{label}_{name}"].astype(np.float64)
                  for name in ("u", "v", "temperature", "qv", "qc",
                               "pressure", "exner", "dz", "w")}
    return {**result, "dx": 12000.0, "dt": 60.0, "cudt": 300.0}


def _shallow_sounding():
    """Deterministic failed-deep column whose deepest fallback is level 5."""
    sounding = _sounding(unstable=True)
    z = np.cumsum(sounding["dz"]) - 0.5 * sounding["dz"]
    sounding["temperature"] = np.maximum(288.0 - 0.004 * z, 205.0)
    sounding["qv"] = 1.25 * (
        0.014 * np.exp(-z / 3500.0)
        / (1.0 + np.exp((z - 1600.0) / 150.0))
        + 0.001 * np.exp(-z / 5000.0))
    return sounding


def _guarded_sounding():
    """Buoyant trigger rejected because its cloud violates WRF top bounds."""
    sounding = _sounding(unstable=True)
    z = np.cumsum(sounding["dz"]) - 0.5 * sounding["dz"]
    sounding["temperature"] = np.maximum(276.0 - 0.002 * z, 205.0)
    sounding["qv"] = 0.4 * (
        0.014 * np.exp(-z / 3500.0)
        / (1.0 + np.exp((z - 1600.0) / 150.0))
        + 0.001 * np.exp(-z / 5000.0))
    sounding["w"] = np.zeros_like(z)
    return sounding


def _noitr_revert_sounding():
    """Deep column that exercises WRF's pre-revert PPTFLX behavior."""
    sounding = _sounding(unstable=True)
    z = np.cumsum(sounding["dz"]) - 0.5 * sounding["dz"]
    sounding["temperature"] = np.maximum(
        298.4098288149002 - 0.009160072858501657 * z
        - 1.5910521383350194 * np.sin(z / 2714.0159533469146), 195.0)
    sounding["qv"] = (
        0.01577952015645559 * np.exp(-z / 4008.7493855872376)
        / (1.0 + np.exp((z - 1975.874432519607) / 679.6869532216963))
        + 0.0016032688726005835 * np.exp(-z / 5952.823988933326))
    sounding["u"] = 13.868313809395254 + 0.0053811191228315515 * z
    sounding["v"] = 6.891164372559324 + 0.00043676849781746274 * z
    sounding["w"] = np.where(z < 1808.1232877308598,
                              0.03092984493021217, 0.0)
    sounding.update(dx=9000.0, dt=300.0, cudt=300.0)
    return sounding


def _tder_suppression_sounding():
    """Deep column with 0 < trial TDER < 1, so WRF removes the downdraft."""
    from gpuwm.verify.npref import _kf_saturation_mixing_ratio

    sounding = _sounding(unstable=True)
    sounding["qv"] = _kf_saturation_mixing_ratio(
        sounding["temperature"], sounding["pressure"])
    sounding.update(dx=300.0, dt=10.0, cudt=300.0)
    return sounding


def _independent_updraft_cape(sounding, source_bottom):
    """Re-derive WRF ABE without calling ``np_kf_column`` or its outputs."""
    from gpuwm.core.kf import load_kf_table
    from gpuwm.verify.npref import (
        _kf_condload, _kf_dtfrznew, _kf_environment_thetae,
        _kf_mixed_virtual_temperature, _kf_prof5,
        _kf_saturation_mixing_ratio, _kf_tpmix2,
    )

    table = load_kf_table()
    temperature = np.asarray(sounding["temperature"], dtype=np.float64)
    pressure = np.asarray(sounding["pressure"], dtype=np.float64)
    qv = np.asarray(sounding["qv"], dtype=np.float64)
    dz = np.asarray(sounding["dz"], dtype=np.float64)
    w = np.asarray(sounding["w"], dtype=np.float64)
    dx = float(sounding["dx"])
    nz = temperature.size
    z = np.cumsum(dz) - 0.5 * dz
    qsat = _kf_saturation_mixing_ratio(temperature, pressure)
    qenv = np.clip(np.minimum(qv, qsat), 1.0e-6, None)
    tv_env = temperature * (1.0 + 0.608 * qenv)
    rho = pressure / (287.0 * tv_env)
    dp = rho * 9.81 * dz

    source_top = source_bottom
    source_dp = 0.0
    while source_top < nz and source_dp <= 5000.0:
        source_dp += dp[source_top]
        source_top += 1
    source = slice(source_bottom, source_top)
    tmix = float(np.sum(dp[source] * temperature[source]) / source_dp)
    qmix = float(np.sum(dp[source] * qenv[source]) / source_dp)
    zmix = float(np.sum(dp[source] * z[source]) / source_dp)
    pmix = float(np.sum(dp[source] * pressure[source]) / source_dp)
    emix = max(qmix * pmix / (0.622 + qmix), 0.6112)
    a1 = max(emix / 611.2, 0.001)
    position = (a1 - 0.001) / 0.075
    index = int(np.trunc(position))
    base = 0.001 + 0.075 * index
    fraction = (a1 - base) / 0.075
    tlog = ((1.0 - fraction) * float(table.log_ratio[index])
            + fraction * float(table.log_ratio[index + 1]))
    dewpoint = (17.67 * 273.15 - 29.65 * tlog) / (17.67 - tlog)
    tlcl = (dewpoint
            - (0.212 + 1.571e-3 * (dewpoint - 273.16)
               - 4.36e-4 * (tmix - 273.16)) * (tmix - dewpoint))
    tlcl = min(tlcl, tmix)
    zlcl = zmix + (tlcl - tmix) / (-9.81 / 1004.5)
    klcl = int(np.searchsorted(z, zlcl, side="left"))
    below = klcl - 1
    lcl_fraction = (zlcl - z[below]) / (z[klcl] - z[below])
    tenv_lcl = ((1.0 - lcl_fraction) * temperature[below]
                + lcl_fraction * temperature[klcl])
    qenv_lcl = ((1.0 - lcl_fraction) * qenv[below]
                + lcl_fraction * qenv[klcl])
    tven_lcl = tenv_lcl * (1.0 + 0.608 * qenv_lcl)
    w_lcl = ((1.0 - lcl_fraction) * w[below] + lcl_fraction * w[klcl])
    w_scaled = w_lcl * dx / 25000.0 - 0.02 * min(zlcl / 2000.0, 1.0)
    dt_lcl = 0.0 if w_scaled < 1.0e-4 else 4.64 * w_scaled ** 0.33
    plcl = ((1.0 - lcl_fraction) * pressure[below]
            + lcl_fraction * pressure[klcl])
    thetae = _kf_environment_thetae(
        pmix, tmix, qmix, table.log_ratio)
    radius = (1000.0 if w_scaled < 0.0 else
              2000.0 if w_scaled > 0.1 else
              1000.0 + 1000.0 * w_scaled / 0.1)
    tv_lcl = tlcl * (1.0 + 0.608 * qmix)
    base_mass_flux = plcl / (287.0 * tv_lcl) * 0.01 * dx * dx
    kbase = klcl - 1
    parcel_t = np.zeros(nz)
    parcel_q = np.zeros(nz)
    thetaeu = np.zeros(nz)
    qliq = np.zeros(nz)
    qice = np.zeros(nz)
    updraft = np.zeros(nz)
    updraft[kbase] = base_mass_flux
    parcel_t[kbase] = tlcl
    parcel_q[kbase] = qmix
    thetaeu[kbase] = thetae
    wlcl = (1.0 if dt_lcl <= 1.0e-4 else
            min(1.0 + 0.5 * np.sqrt(
                2.0 * 9.81 * dt_lcl * 500.0 / tven_lcl), 3.0))
    w2 = wlcl * wlcl
    ee1, ud1, rei = 1.0, 0.0, 0.0
    upold = upnew = base_mass_flux
    ttemp = 268.16
    cape = 0.0
    for nk in range(kbase, nz - 1):
        nk1 = nk + 1
        parcel_t[nk1] = temperature[nk1]
        thetaeu[nk1] = thetaeu[nk]
        parcel_q[nk1] = parcel_q[nk]
        qliq[nk1] = qliq[nk]
        qice[nk1] = qice[nk]
        (parcel_t[nk1], parcel_q[nk1], qliq[nk1], qice[nk1],
         qnewlq, qnewice) = _kf_tpmix2(
             pressure[nk1], thetaeu[nk1], parcel_t[nk1], parcel_q[nk1],
             qliq[nk1], qice[nk1], table)
        if parcel_t[nk1] <= 268.16:
            if parcel_t[nk1] > 248.16:
                ttemp = min(ttemp, 268.16)
                frozen_fraction = ((ttemp - parcel_t[nk1])
                                   / (ttemp - 248.16))
            else:
                frozen_fraction = 1.0
            ttemp = parcel_t[nk1]
            frozen = (qliq[nk1] + qnewlq) * frozen_fraction
            qnewice += qnewlq * frozen_fraction
            qnewlq *= 1.0 - frozen_fraction
            qice[nk1] += qliq[nk1] * frozen_fraction
            qliq[nk1] *= 1.0 - frozen_fraction
            (parcel_t[nk1], thetaeu[nk1], parcel_q[nk1],
             qice[nk1]) = _kf_dtfrznew(
                 parcel_t[nk1], pressure[nk1], thetaeu[nk1],
                 parcel_q[nk1], frozen, qice[nk1])
        tvu = parcel_t[nk1] * (1.0 + 0.608 * parcel_q[nk1])
        if nk == kbase:
            layer_depth = z[nk1] - zlcl
            be = (tv_lcl + tvu) / (tven_lcl + tv_env[nk1]) - 1.0
        else:
            layer_depth = z[nk1] - z[nk]
            tvu_below = parcel_t[nk] * (1.0 + 0.608 * parcel_q[nk])
            be = (tvu_below + tvu) / (tv_env[nk] + tv_env[nk1]) - 1.0
        (qliq[nk1], qice[nk1], w2, qnewlq, qnewice,
         _qlqout, _qicout) = _kf_condload(
             qliq[nk1], qice[nk1], w2, layer_depth,
             2.0 * layer_depth * 9.81 * be / 1.5,
             2.0 * rei * w2 / upold, qnewlq, qnewice)
        if w2 < 1.0e-3:
            break
        environment_thetae = _kf_environment_thetae(
            pressure[nk1], temperature[nk1], qenv[nk1], table.log_ratio)
        rei = base_mass_flux * dp[nk1] * 0.03 / radius
        tvqu = parcel_t[nk1] * (
            1.0 + 0.608 * parcel_q[nk1] - qliq[nk1] - qice[nk1])
        if nk == kbase:
            dilbe = ((tv_lcl + tvqu) / (tven_lcl + tv_env[nk1]) - 1.0) \
                * layer_depth
        else:
            tvqu_below = parcel_t[nk] * (
                1.0 + 0.608 * parcel_q[nk] - qliq[nk] - qice[nk])
            dilbe = ((tvqu_below + tvqu) / (tv_env[nk] + tv_env[nk1])
                     - 1.0) * layer_depth
        if dilbe > 0.0:
            cape += dilbe * 9.81
        if tvqu <= tv_env[nk1]:
            ee2, ud2 = 0.5, 1.0
        else:
            f1, f2 = 0.95, 0.05
            mixed_virtual = _kf_mixed_virtual_temperature(
                pressure[nk1], f1 * environment_thetae + f2 * thetaeu[nk1],
                f1 * qenv[nk1] + f2 * parcel_q[nk1],
                f2 * qliq[nk1], f2 * qice[nk1], table)
            if mixed_virtual > tv_env[nk1]:
                ee2, ud2 = 1.0, 0.0
            else:
                f1, f2 = 0.10, 0.90
                mixed_virtual = _kf_mixed_virtual_temperature(
                    pressure[nk1],
                    f1 * environment_thetae + f2 * thetaeu[nk1],
                    f1 * qenv[nk1] + f2 * parcel_q[nk1],
                    f2 * qliq[nk1], f2 * qice[nk1], table)
                equilibrium = np.clip(
                    (tv_env[nk1] - tvqu) * f1 / (mixed_virtual - tvqu),
                    0.0, 1.0)
                if equilibrium == 1.0:
                    ee2, ud2 = 1.0, 0.0
                elif equilibrium == 0.0:
                    ee2, ud2 = 0.0, 1.0
                else:
                    ee2, ud2 = _kf_prof5(equilibrium)
        ee2 = max(ee2, 0.5)
        ud2 *= 1.5
        entrainment = 0.5 * rei * (ee1 + ee2)
        detrainment = 0.5 * rei * (ud1 + ud2)
        if updraft[nk] - detrainment < 10.0:
            if dilbe > 0.0:
                cape -= dilbe * 9.81
            break
        ee1, ud1 = ee2, ud2
        upold = updraft[nk] - detrainment
        upnew = upold + entrainment
        updraft[nk1] = upnew
        parcel_q[nk1] = (upold * parcel_q[nk1]
                         + entrainment * qenv[nk1]) / upnew
        thetaeu[nk1] = (upold * thetaeu[nk1]
                        + entrainment * environment_thetae) / upnew
        qliq[nk1] *= upold / upnew
        qice[nk1] *= upold / upnew
    return float(cape)


def _column_budget_residuals(sounding, result):
    """Return applied total-water/MSE residuals and their numeric scales."""
    temperature = np.asarray(sounding["temperature"], dtype=np.float64)
    pressure = np.asarray(sounding["pressure"], dtype=np.float64)
    qv = np.asarray(sounding["qv"], dtype=np.float64)
    dz = np.asarray(sounding["dz"], dtype=np.float64)
    es = 611.2 * np.exp((17.67 * temperature - 17.67 * 273.15)
                        / (temperature - 29.65))
    qsat = 0.622 * es / (pressure - es)
    qenv = np.clip(np.minimum(qv, qsat), 1.0e-6, None)
    mass = pressure / (287.0 * temperature * (1.0 + 0.608 * qenv)) * dz
    rain_rate = float(result["rainc"]) / float(sounding["cudt"])
    water_layers = mass * (
        result["rqvcuten"] + result["rqccuten"] + result["rqicuten"]
        + result["rqrcuten"] + result["rqscuten"])
    water_residual = float(np.sum(water_layers) + rain_rate)
    water_scale = float(np.sum(np.abs(water_layers)) + abs(rain_rate))
    sensible_layers = (mass * 1004.5 * result["rthcuten"]
                       * sounding["exner"])
    latent_layers = (mass * (3.15e6 - 2370.0 * temperature)
                     * result["rqvcuten"])
    mse_residual = float(np.sum(sensible_layers + latent_layers))
    mse_scale = float(np.sum(np.abs(sensible_layers)
                             + np.abs(latent_layers)))
    return water_residual, water_scale, mse_residual, mse_scale


def test_kf_lookup_table_is_shipped_data_with_provenance():
    from gpuwm.core.kf import load_kf_table

    table = load_kf_table()
    assert table.temperature.shape == (250, 220)
    assert table.qsat.shape == (250, 220)
    assert table.thetae_base.shape == (220,)
    assert table.log_ratio.shape == (200,)
    assert table.pressure_top == 5000.0
    assert table.pressure_reciprocal > 0.0
    assert table.thetae_reciprocal == 1.0
    assert np.isfinite(table.temperature).all()
    assert np.isfinite(table.qsat).all()
    assert np.all(np.diff(table.log_ratio) > 0.0)
    provenance = (Path(__file__).parents[1] / "gpuwm" / "data" /
                  "kf_lutab" / "PROVENANCE.md")
    text = provenance.read_text(encoding="utf-8")
    assert "module_cu_kfeta.F:3174-3301" in text
    assert "https://github.com/wrf-model/WRF.git" in text
    assert "v4.6.1" in text
    assert "d66e442fccc04111067e29274c9f9eaccc3cef28" in text
    # The parent-file pin must be the LF digest -- the bytes a Linux checkout
    # and the gfortran oracle build see -- because that is where the pin gets
    # verified.  The CRLF rendering stays recorded so a Windows checkout that
    # computes it does not read a mismatch as a moved source, but it is
    # labelled, not offered as the authority.
    lf_digest = "e6376c2d85c45470f49d545b25d513b5ec111bf36b87beebc740bf42825c6e5f"
    crlf_digest = "b2ee225b2148d54afa464f941d967cb197a8a73e23d6fd3450086c6f3a705895"
    lowered = text.lower()
    assert lf_digest in lowered
    assert crlf_digest in lowered
    assert lowered.index(lf_digest) < lowered.index(crlf_digest)
    assert "CRLF" in text
    # The FABE erratum: the published relative move was the absolute one.
    assert "4.46e-5" in text
    assert "1.32e-6" in text
    assert "default-`REAL` arithmetic" in text
    assert "Deliberate runtime deviations from WRF v4.6.1" in text
    for documented_guard in ("0.99*p", "dewpoint/`ALU`", "zero `TMA`",
                             "top two mass levels", "return zero outputs",
                             "`DPTHMX > 5000 Pa`"):
        assert documented_guard in text
    assert "SHA-256" in text
    license_file = provenance.with_name("LICENSE.txt")
    license_text = license_file.read_text(encoding="utf-8")
    assert "National Center for Atmospheric Research" in license_text
    assert "public domain" in license_text
    assert (hashlib.sha256((provenance.parent / "kf_lutab.npz").read_bytes())
            .hexdigest()) in text


#: SHA-256 of the raw binary32 streams `tools/kf_lutab_oracle.F90` writes --
#: WRF v4.6.1 `module_cu_kfeta.F:3174-3301` verbatim, gfortran 13.3.0 against
#: glibc 2.39 on x86-64, identical at `-O0 -ffp-contract=off` and at
#: `-O2 -ftree-vectorize`.  Recorded in `gpuwm/data/kf_lutab/PROVENANCE.md`.
#: Fortran writes column-major, hence `order="F"` on the two 2-D members.
_KF_LUTAB_ORACLE_SHA256 = {
    "ttab.bin":
        "04c8e3cd4138e440d486b5899575063fde81b4beba5d82c8ffe6a2d766fd5ce0",
    "qstab.bin":
        "39ae8b543154f035cc47512929335c15c7880c30f33f41cd8a4518c3e8975d7a",
    "the0k.bin":
        "a6e207a3f0f9b3631aea74ce89b070b57eed6c298fb40f2488dca4627391874d",
    "alu.bin":
        "6836d80708f02642d1dc53b547817dbab623545cfee778671290d7d825eeaf4d",
    "scalars.bin":
        "206e7022af0e9ef62d5203083a83efac175607e8263c873b19b7708d6bbe352c",
}

#: NumPy ufuncs the generator must never reach for.  WRF's `KF_LUTAB` is
#: default `REAL` -- binary32 -- and gfortran lowers its `exp`, `**` and
#: `alog` to the scalar glibc `expf`, `powf` and `logf`.  NumPy's float32
#: transcendentals are a different function *and* move between releases, so
#: one appearing here is both a fidelity bug and a reproducibility bug.
_FORBIDDEN_NUMPY_TRANSCENDENTALS = (
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "power",
    "float_power", "sqrt", "cbrt",
)


def _oracle_stream_digests(table):
    """Digest a table the way `kf_lutab_oracle.F90` writes it out."""
    def as_le(array):
        return np.asarray(array, dtype="<f4")

    return {
        "ttab.bin": hashlib.sha256(
            as_le(table["temperature"]).tobytes(order="F")).hexdigest(),
        "qstab.bin": hashlib.sha256(
            as_le(table["qsat"]).tobytes(order="F")).hexdigest(),
        "the0k.bin": hashlib.sha256(
            as_le(table["thetae_base"]).tobytes()).hexdigest(),
        "alu.bin": hashlib.sha256(
            as_le(table["log_ratio"]).tobytes()).hexdigest(),
        "scalars.bin": hashlib.sha256(as_le([
            table["pressure_reciprocal"], table["thetae_reciprocal"],
            table["pressure_top"]]).tobytes()).hexdigest(),
    }


def test_kf_lutab_generator_preserves_wrf_single_precision_semantics():
    """Reproduce `KF_LUTAB` bit-for-bit, against WRF and not against NumPy.

    `module_cu_kfeta.F:3174-3301` is default `REAL`, and WRF v4.6.1 does not
    promote it (`arch/preamble:63` sets `NATIVE_RWORDSIZE = 4`; every
    `PROMOTION` line in `arch/configure.defaults` leaves `-fdefault-real-8`
    commented out).  So "WRF single precision semantics" means binary32
    arithmetic *plus* the single-precision libm gfortran calls -- glibc's
    `expf`, `powf`, `logf`.  This pins all 110,420 cells against the recorded
    output of that Fortran, and guards the generator against drifting back
    onto a NumPy transcendental, which is neither WRF's function nor stable
    between NumPy releases.
    """
    from gpuwm.core import noahmp_libm
    from gpuwm.core.kf import load_kf_table
    from tools import generate_kf_lutab

    # The libm the pinned digests were produced against.  A host that moves
    # off glibc 2.39 has not invalidated them -- the transcriptions are
    # self-contained -- but the claim they reproduce is version-specific.
    assert noahmp_libm.GLIBC_VERSION == "2.39"

    generated = generate_kf_lutab.generate()
    assert all(value.dtype == np.float32 for value in generated.values())

    # All 110,420 cells, against gfortran 13.3.0 + glibc 2.39.
    assert _oracle_stream_digests(generated) == _KF_LUTAB_ORACLE_SHA256

    # Anchors, so a digest mismatch reports a number and not just a hash.
    # Every value below is what the Fortran prints, not what NumPy computes;
    # (88, 211) and (148, 204) are the worst cells of the superseded
    # NumPy-built table (65 and 452 ULP off WRF respectively).
    anchors = ((0, 0), (10, 50), (100, 100), (249, 219), (88, 211),
               (148, 204))
    np.testing.assert_array_equal(
        [generated["temperature"][index] for index in anchors],
        np.asarray([150.0, 157.02183532714844, 232.8066864013672,
                    307.37762451171875, 238.96221923828125,
                    281.1182556152344], dtype=np.float32))
    np.testing.assert_array_equal(
        [generated["qsat"][index] for index in anchors],
        np.asarray([1.0677771999922925e-9, 1.3225719408538339e-9,
                    2.1497125271707773e-4, 3.207545354962349e-2,
                    1.9985366088803858e-4, 6.541135720908642e-3],
                   dtype=np.float32))
    np.testing.assert_array_equal(
        generated["thetae_base"][[0, 50, 100, 219]],
        np.asarray([352.6997985839844, 213.61912536621094,
                    179.8502655029297, 145.97471618652344],
                   dtype=np.float32))
    np.testing.assert_array_equal(
        generated["log_ratio"][[0, 1, 50, 100, 199]],
        np.asarray([-6.9077534675598145, -2.577021837234497,
                    1.3220229148864746, 2.015035390853882,
                    2.7031028270721436], dtype=np.float32))

    shipped = load_kf_table()
    for name in ("temperature", "qsat", "thetae_base", "log_ratio"):
        np.testing.assert_array_equal(generated[name], getattr(shipped, name))
    for name in ("pressure_top", "pressure_reciprocal",
                 "thetae_reciprocal"):
        assert float(generated[name]) == getattr(shipped, name)


def test_kf_lutab_generator_calls_no_numpy_transcendental(monkeypatch):
    """Negative control for the table's reproducibility.

    The digests above cannot tell you *why* the table reproduces, so this
    breaks every NumPy elementary function under the generator and reruns it.
    If a future edit reaches for `np.exp` instead of
    `gpuwm.core.noahmp_libm.expf`, this fails immediately rather than five
    NumPy releases later.  `+ - * /` are left alone: those are correctly
    rounded in binary32 and are the same function everywhere.
    """
    from tools import generate_kf_lutab

    called: list[str] = []

    def forbid(name):
        def trap(*args, **kwargs):
            called.append(name)
            raise AssertionError(
                f"generate_kf_lutab called np.{name}; WRF's KF_LUTAB is "
                f"binary32 and gfortran lowers it to glibc's expf/powf/logf. "
                f"Use gpuwm.core.noahmp_libm.")
        return trap

    for name in _FORBIDDEN_NUMPY_TRANSCENDENTALS:
        monkeypatch.setattr(np, name, forbid(name), raising=True)

    # The trap is live: prove it fires before trusting that it did not.
    with pytest.raises(AssertionError, match=r"called np\.exp"):
        np.exp(np.float32(1.0))
    called.clear()

    table = generate_kf_lutab.generate()
    assert called == []
    assert _oracle_stream_digests(table) == _KF_LUTAB_ORACLE_SHA256


def test_kf_mirror_strongly_unstable_closes_cape_and_rains():
    from gpuwm.verify.npref import np_kf_column

    result = np_kf_column(**_sounding(unstable=True))
    assert result["triggered"]
    assert result["cape_before"] > 100.0
    assert 0.0 <= result["cape_after"] / result["cape_before"] <= 0.15
    assert 1 <= result["closure_iterations"] <= 10
    np.testing.assert_allclose(
        result["closure_fabe"],
        result["cape_after"] / result["cape_before"], rtol=0.0, atol=1.0e-14)
    assert result["closure_scale"] >= 0.05
    assert result["rainc"] > 0.0
    assert np.any(result["updraft_mass_flux"] > 0.0)
    assert np.any(result["downdraft_mass_flux"] < 0.0)
    # A deep mixed-phase column must exercise both TPMIX2 condensate
    # categories, TTFRZ..TBFRZ glaciation, and CONDLOAD fallout.
    assert np.any(result["updraft_liquid"] > 0.0)
    assert np.any(result["updraft_ice"] > 0.0)
    assert np.sum(result["liquid_precip_flux"]) > 0.0
    assert np.sum(result["ice_precip_flux"]) > 0.0
    for name, bound in (("rthcuten", 2.0e-2),
                        ("rqvcuten", 2.0e-5),
                        ("rqccuten", 2.0e-5), ("rqicuten", 2.0e-5),
                        ("rqrcuten", 2.0e-5), ("rqscuten", 2.0e-5)):
        value = result[name]
        assert np.isfinite(value).all(), name
        assert np.max(np.abs(value)) <= bound, name


def test_kf_mirror_stable_column_is_exact_noop():
    from gpuwm.verify.npref import np_kf_column

    result = np_kf_column(**_sounding(unstable=False))
    assert not result["triggered"]
    assert result["rainc"] == 0.0
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rqrcuten", "rqscuten",
                 "updraft_mass_flux", "downdraft_mass_flux"):
        np.testing.assert_array_equal(result[name], 0.0, err_msg=name)


def test_kf_mirror_selects_deepest_failed_deep_as_shallow():
    """Pin WRF 818-845, 1367-1424, 1911-1948, and 2569-2575."""
    from gpuwm.verify.npref import np_kf_column

    sounding = _shallow_sounding()
    result = np_kf_column(**sounding)
    assert result["triggered"] and result["shallow"]
    assert result["source_bottom"] == 5
    assert result["source_top"] == 7
    assert result["cloud_top"] > max(result["cloud_base"],
                                      result["source_top"] - 1)
    assert result["closure_iterations"] == 1
    assert result["timec"] == 2400.0
    assert result["nca_seconds"] == sounding["cudt"] == 300.0
    assert result["rainc"] == result["precip_rate"] == 0.0
    assert np.any(result["rqrcuten"] > 0.0)
    np.testing.assert_array_equal(result["downdraft_mass_flux"], 0.0)
    water, water_scale, mse, _ = _column_budget_residuals(sounding, result)
    assert abs(water) <= (4.0 * sounding["temperature"].size
                          * np.finfo(np.float64).eps
                          * max(water_scale, 1.0e-12))
    assert np.isfinite(mse)


def test_kf_triggered_column_failing_cloud_top_guards_is_exact_noop():
    from gpuwm.verify.npref import np_kf_column

    result = np_kf_column(**_guarded_sounding())
    assert result["triggered_candidates"] == result["guard_rejections"] == 1
    assert not result["triggered"]
    assert result["nca_seconds"] == result["rainc"] == 0.0
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rqrcuten", "rqscuten",
                 "updraft_mass_flux", "downdraft_mass_flux"):
        np.testing.assert_array_equal(result[name], 0.0, err_msg=name)


def test_kf_source_depth_and_shear_denominator_match_wrf_strict_forms():
    from gpuwm.verify.npref import np_kf_column

    root = Path(__file__).parents[1]
    mirror = (root / "gpuwm" / "verify" / "npref.py").read_text(
        encoding="utf-8")
    kernel = (root / "gpuwm" / "core" / "kernels" / "kf.cu").read_text(
        encoding="utf-8")
    assert "source_dp <= 5000.0" in mirror
    assert "sum_dp <= 5000.0f" in kernel
    assert "max(z[cloud_top] - z[klcl], 1.0)" not in mirror
    assert "fmaxf(z[cloud_top] - z[klcl], 1.0f)" not in kernel
    for sounding in (_sounding(unstable=True), _real74_sounding("unstable"),
                     _shallow_sounding(), _noitr_revert_sounding()):
        result = np_kf_column(**sounding)
        z = np.cumsum(sounding["dz"]) - 0.5 * sounding["dz"]
        assert z[result["cloud_top"]] > z[result["cloud_base"]]


def test_real74_12z_extracted_column_gates_and_provenance():
    """The plan's actual Phase-3 d01 warm-sector/northern column gates."""
    from gpuwm.verify.npref import np_kf_column

    with np.load(_REAL74_COLUMNS, allow_pickle=False) as data:
        assert (int(data["unstable_j"]), int(data["unstable_i"])) == (53, 113)
        assert 30.0 < float(data["unstable_latitude"]) < 40.0
        assert (int(data["stable_j"]), int(data["stable_i"])) == (194, 5)
        assert float(data["stable_latitude"]) > 45.0
    provenance = _REAL74_COLUMNS.with_name("KF_REAL74_PROVENANCE.md")
    provenance_text = provenance.read_text(encoding="utf-8")
    assert "prepare_phase3_case()" in provenance_text
    assert (hashlib.sha256(_REAL74_COLUMNS.read_bytes()).hexdigest()
            in provenance_text)

    unstable = np_kf_column(**_real74_sounding("unstable"))
    assert unstable["triggered"]
    # Both pins moved on 2026-07-25 when kf_lutab.npz was rebuilt against
    # glibc's expf/powf/logf instead of NumPy's float32 ufuncs, i.e. onto the
    # values WRF's own KF_LUTAB produces (see the table's PROVENANCE.md).
    # cape_before 432.29333945446024 -> 432.29276931463795 (5.70e-4 absolute,
    # 1.32e-6 relative), FABE 0.09302343911952345 -> 0.0930192928993745
    # (4.15e-6 absolute, **4.46e-5 relative**).  The FABE figure was published
    # here and in 4239185 as "4.5e-6"/"~1e-6" relative, which was its ABSOLUTE
    # move; FABE is cape_after/cape_before out of a secant iteration, so it
    # amplifies a table move by ~34x rather than tracking it.  See the erratum
    # table in gpuwm/data/kf_lutab/PROVENANCE.md.
    # Member attribution, by rerunning this column with each member reverted:
    # temperature/qsat carry 5.78e-4 K of the 5.70e-4 K move, the 21 one-ULP
    # log_ratio cells only 8.3e-6 (they partly cancel).  Tolerances are
    # unchanged -- these are exact pins moved onto WRF, not a widened gate.
    np.testing.assert_allclose(unstable["cape_before"],
                               432.29276931463795,
                               rtol=2.0e-13, atol=1.0e-10)
    np.testing.assert_allclose(
        unstable["cape_after"] / unstable["cape_before"],
        0.0930192928993745, rtol=2.0e-13, atol=1.0e-12)
    # Recompute the two published move sizes rather than trusting the prose
    # above: this is exactly the arithmetic that was got wrong once.
    cape_move = (432.29333945446024 - 432.29276931463795) / 432.29333945446024
    fabe_move = (0.09302343911952345 - 0.0930192928993745) / 0.09302343911952345
    np.testing.assert_allclose(cape_move, 1.32e-6, rtol=5.0e-3)
    np.testing.assert_allclose(fabe_move, 4.46e-5, rtol=5.0e-3)
    assert fabe_move > 30.0 * cape_move
    assert 1 <= unstable["closure_iterations"] <= 10
    assert unstable["rainc"] > 0.0
    assert np.any(unstable["updraft_liquid"] > 0.0)
    assert np.any(unstable["updraft_ice"] > 0.0)
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rqrcuten", "rqscuten"):
        assert np.isfinite(unstable[name]).all(), name
    for name, bound in (("rthcuten", 5.0e-3),
                        ("rqvcuten", 5.0e-6),
                        ("rqccuten", 5.0e-6), ("rqicuten", 5.0e-6),
                        ("rqrcuten", 5.0e-6), ("rqscuten", 5.0e-6)):
        assert np.max(np.abs(unstable[name])) <= bound, name
    assert unstable["rainc"] <= 5.0
    # Apply the converged tendencies, then re-integrate the original elevated
    # source parcel with a separately coded WRF updraft/ABE diagnostic.  This
    # never reads cape_after and never calls np_kf_column for the adjusted
    # sounding, so an assigned scheme diagnostic cannot satisfy the gate.
    independent_before = _independent_updraft_cape(
        _real74_sounding("unstable"), source_bottom=12)
    np.testing.assert_allclose(independent_before, unstable["cape_before"],
                               rtol=2.0e-12, atol=1.0e-10)
    adjusted = _real74_sounding("unstable")
    adjusted["temperature"] += (
        unstable["rthcuten"] * adjusted["exner"] * unstable["timec"])
    adjusted["qv"] += unstable["rqvcuten"] * unstable["timec"]
    adjusted["qc"] += unstable["rqccuten"] * unstable["timec"]
    independent_after = _independent_updraft_cape(
        adjusted, source_bottom=12)
    assert 0.0 <= independent_after <= 0.15 * independent_before

    # The scheme's rainc output is precip_rate*min(cudt, feedback_time)
    # (kf.cu:1140-1149); the Task-6b driver divides that increment back out
    # to recover WRF's persistent PRATEC, so the cudt scaling must be exact.
    short = np_kf_column(**{**_real74_sounding("unstable"), "cudt": 60.0})
    np.testing.assert_allclose(unstable["rainc"], 5.0 * short["rainc"],
                               rtol=1.0e-12, atol=0.0)

    stable = np_kf_column(**_real74_sounding("stable"))
    assert not stable["triggered"]
    assert stable["rainc"] == 0.0
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rqrcuten", "rqscuten",
                 "updraft_mass_flux", "downdraft_mass_flux"):
        np.testing.assert_array_equal(stable[name], 0.0, err_msg=name)


@pytest.mark.parametrize("column", ("synthetic", "unstable", "shallow",
                                    "stable"))
def test_kf_mirror_closes_water_and_reports_uncorrected_mse(column):
    from gpuwm.verify.npref import np_kf_column

    sounding = (_sounding(unstable=True) if column == "synthetic"
                else _shallow_sounding() if column == "shallow"
                else _real74_sounding(column))
    result = np_kf_column(**sounding)
    water, water_scale, mse, mse_scale = _column_budget_residuals(
        sounding, result)
    accumulated_operations = 4.0 * sounding["temperature"].size
    water_bound = (accumulated_operations * np.finfo(np.float64).eps
                   * max(water_scale, 1.0e-12))
    assert abs(water) <= water_bound
    assert np.isfinite(mse)
    assert "reported_mse_residual" in result
    assert np.isfinite(result["reported_mse_residual"])
    if result["triggered"]:
        np.testing.assert_allclose(result["reported_mse_residual"], mse,
                                   rtol=2.0e-12, atol=1.0e-10)
        # raw_rthcuten is an INDEPENDENT transcription of the uncorrected
        # feedback rate (tg - t0)/(timec*exner) from the closure locals
        # (WRF DTDT=(TG-T0)/TIMEC at module_cu_kfeta.F:2640, theta
        # conversion RTHCUTEN=DTDT/pi at :487), so this equality fails if
        # any in-scheme "correction" ever touches the returned rate.
        np.testing.assert_array_equal(result["raw_rthcuten"],
                                      result["rthcuten"])
        if not result["shallow"]:
            # WRF KF does not close column enthalpy (Lv(T)/melt
            # approximations are part of the scheme); a machine-zero
            # residual on a triggered deep column means somebody "fixed"
            # it, which the plan auto-rejects.
            assert result["reported_mse_residual"] != 0.0
    else:
        assert mse == result["reported_mse_residual"] == 0.0
        np.testing.assert_array_equal(result["raw_rthcuten"],
                                      result["rthcuten"])
    # Supplementary source-level tripwire only: the raw-rate equality above
    # shares the closure locals tg/timec/exner, so it would not catch a
    # correction folded into tg itself.  The independently re-derived CAPE
    # removal bound is the behavioral guard for that remaining evasion path.
    source = (Path(__file__).parents[1] / "gpuwm" / "core" / "kernels" /
              "kf.cu").read_text(encoding="utf-8")
    assert "temperature_rate_correction" not in source


@pytest.mark.gpu
@requires_gpu
def test_kf_kernel_matches_float64_mirror_at_fp32_floors():
    import cupy as cp

    from gpuwm.core.kf import launch_kf
    from gpuwm.verify.npref import np_kf_column

    columns = [_sounding(unstable=True), _real74_sounding("unstable"),
               _shallow_sounding(), _guarded_sounding(),
               _real74_sounding("stable")]
    names = ("u", "v", "temperature", "qv", "qc", "pressure", "exner",
             "dz", "w")
    host = {
        name: np.ascontiguousarray(
            np.stack([column[name] for column in columns], axis=1)[:, None],
            dtype=np.float32)
        for name in names
    }
    got = launch_kf(**{name: cp.asarray(value) for name, value in host.items()},
                    dx=12000.0, dt=60.0, cudt=300.0)

    refs = [np_kf_column(**{
        **{name: host[name][:, 0, i].astype(np.float64) for name in names},
        "dx": 12000.0, "dt": 60.0, "cudt": 300.0,
    }) for i in range(len(columns))]
    for name, scale in (("rthcuten", 1.0e-3),
                        ("rqvcuten", 1.0e-6),
                        ("rqccuten", 1.0e-6), ("rqicuten", 1.0e-6),
                        ("rqrcuten", 1.0e-6), ("rqscuten", 1.0e-6)):
        ref = np.stack([item[name] for item in refs], axis=1)[:, None]
        np.testing.assert_allclose(
            cp.asnumpy(got[name]), ref, rtol=5.0e-4,
            atol=256.0 * np.finfo(np.float32).eps * scale, err_msg=name)
    for name in ("updraft_mass_flux", "downdraft_mass_flux"):
        ref = np.stack([item[name] for item in refs], axis=1)[:, None]
        fp32_floor = max(
            0.5, 128.0 * np.finfo(np.float32).eps * np.max(np.abs(ref)))
        np.testing.assert_allclose(cp.asnumpy(got[name]), ref,
                                   rtol=5.0e-5, atol=fp32_floor, err_msg=name)
    np.testing.assert_allclose(
        cp.asnumpy(got["rainc"])[0], [item["rainc"] for item in refs],
        rtol=5.0e-4, atol=2.0e-6)
    np.testing.assert_array_equal(
        cp.asnumpy(got["triggered"])[0],
        [item["triggered"] for item in refs])
    np.testing.assert_array_equal(
        cp.asnumpy(got["shallow"])[0],
        [item["shallow"] for item in refs])
    np.testing.assert_allclose(
        cp.asnumpy(got["nca_seconds"])[0],
        [item["nca_seconds"] for item in refs], rtol=0.0, atol=0.0)
    device = {name: cp.asnumpy(got[name]) for name in
              ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
               "rqrcuten", "rqscuten", "rainc")}
    for column, sounding in enumerate(columns):
        result = {
            name: device[name][:, 0, column]
            for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                         "rqrcuten", "rqscuten")
        }
        result["rainc"] = device["rainc"][0, column]
        fp32_sounding = {
            **{name: host[name][:, 0, column].astype(np.float64)
               for name in names},
            "dx": 12000.0, "dt": 60.0, "cudt": 300.0,
        }
        water, water_scale, mse, _mse_scale = _column_budget_residuals(
            fp32_sounding, result)
        accumulated_operations = 4.0 * fp32_sounding["temperature"].size
        assert abs(water) <= (accumulated_operations
                              * np.finfo(np.float32).eps
                              * max(water_scale, 1.0e-12))
        assert np.isfinite(mse)
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rqrcuten", "rqscuten"):
        np.testing.assert_array_equal(device[name][:, 0, 4], 0.0,
                                      err_msg=name)


@pytest.mark.gpu
@requires_gpu
def test_kf_kernel_warm_rain_and_no_snow_phase_energy_match_mirror():
    """CUDA pins WRF's two latent-fusion feedback branches."""
    import cupy as cp

    from gpuwm.core.kf import KFPhaseMode, launch_kf
    from gpuwm.verify.npref import np_kf_column

    sounding = _shallow_sounding()
    names = ("u", "v", "temperature", "qv", "qc", "pressure", "exner",
             "dz", "w")
    host = {
        name: np.ascontiguousarray(
            np.asarray(sounding[name])[:, None, None], dtype=np.float32)
        for name in names
    }
    reference_inputs = {
        **{name: host[name][:, 0, 0].astype(np.float64) for name in names},
        "dx": sounding["dx"], "dt": sounding["dt"],
        "cudt": sounding["cudt"],
    }
    for mode in (KFPhaseMode.WARM_RAIN,
                 KFPhaseMode.NO_SEPARATE_SNOW):
        got = launch_kf(
            **{name: cp.asarray(value) for name, value in host.items()},
            dx=sounding["dx"], dt=sounding["dt"],
            cudt=sounding["cudt"], phase_mode=mode)
        reference = np_kf_column(**reference_inputs, phase_mode=mode)
        assert reference["triggered"] and reference["shallow"]
        for name, scale in (("rthcuten", 1.0e-3),
                            ("rqvcuten", 1.0e-6),
                            ("rqccuten", 1.0e-6),
                            ("rqicuten", 1.0e-6),
                            ("rqrcuten", 1.0e-6),
                            ("rqscuten", 1.0e-6)):
            np.testing.assert_allclose(
                cp.asnumpy(got[name])[:, 0, 0], reference[name],
                rtol=5.0e-4,
                atol=256.0 * np.finfo(np.float32).eps * scale,
                err_msg=f"{mode.name}:{name}")
        # Both branches return only QC/QR, but the sum must still equal the
        # independent four-category closure water rate.
        active = slice(0, reference["cloud_top"] + 1)
        expected_water = sum(
            reference[name][active] for name in
            ("closure_liquid", "closure_ice",
             "closure_rain", "closure_snow")) / reference["timec"]
        actual_water = (
            cp.asnumpy(got["rqccuten"])[active, 0, 0]
            + cp.asnumpy(got["rqrcuten"])[active, 0, 0])
        np.testing.assert_allclose(
            actual_water, expected_water, rtol=5.0e-4,
            atol=256.0 * np.finfo(np.float32).eps * 1.0e-6)


@pytest.mark.gpu
@requires_gpu
def test_kf_kernel_preserves_pre_revert_pptflx_on_noitr_column():
    """WRF assigns PPTFLX at rescale (2262), not after AINC reversion."""
    import cupy as cp

    from gpuwm.core.kf import launch_kf
    from gpuwm.verify.npref import np_kf_column

    sounding = _noitr_revert_sounding()
    reference = np_kf_column(**sounding)
    assert reference["triggered"] and reference["closure_noitr_revert"]
    assert reference["closure_precip_scale"] != reference["closure_scale"]
    names = ("u", "v", "temperature", "qv", "qc", "pressure", "exner",
             "dz", "w")
    host = {name: np.ascontiguousarray(sounding[name][:, None, None],
                                       dtype=np.float32)
            for name in names}
    got = launch_kf(
        **{name: cp.asarray(value) for name, value in host.items()},
        dx=sounding["dx"], dt=sounding["dt"], cudt=sounding["cudt"])
    np.testing.assert_allclose(cp.asnumpy(got["rainc"])[0, 0],
                               reference["rainc"], rtol=5.0e-4,
                               atol=2.0e-6)
    for name, scale in (("rthcuten", 1.0e-3), ("rqvcuten", 1.0e-6),
                        ("rqccuten", 1.0e-6), ("rqicuten", 1.0e-6),
                        ("rqrcuten", 1.0e-6), ("rqscuten", 1.0e-6)):
        np.testing.assert_allclose(
            cp.asnumpy(got[name])[:, 0, 0], reference[name], rtol=2.0e-3,
            atol=256.0 * np.finfo(np.float32).eps * scale, err_msg=name)


@pytest.mark.gpu
@requires_gpu
def test_kf_kernel_zeroes_tder_when_trial_downdraft_is_suppressed():
    import cupy as cp

    from gpuwm.core.kf import launch_kf
    from gpuwm.verify.npref import np_kf_column

    sounding = _tder_suppression_sounding()
    reference = np_kf_column(**sounding)
    assert 0.0 < reference["downdraft_evaporation_before_suppression"] < 1.0
    assert reference["downdraft_evaporation"] == 0.0
    np.testing.assert_array_equal(reference["downdraft_mass_flux"], 0.0)
    names = ("u", "v", "temperature", "qv", "qc", "pressure", "exner",
             "dz", "w")
    host = {name: np.ascontiguousarray(sounding[name][:, None, None],
                                       dtype=np.float32)
            for name in names}
    got = launch_kf(
        **{name: cp.asarray(value) for name, value in host.items()},
        dx=sounding["dx"], dt=sounding["dt"], cudt=sounding["cudt"])
    np.testing.assert_allclose(cp.asnumpy(got["rainc"])[0, 0],
                               reference["rainc"], rtol=5.0e-4,
                               atol=2.0e-6)


@pytest.mark.gpu
@requires_gpu
def test_native_kf_validation_batches_fields_and_resets_status():
    import cupy as cp

    from gpuwm.core.kf import validate_kf_outputs

    shape = (8, 2, 3)
    surface = shape[1:]
    values = tuple(
        cp.full(shape if index < 6 else surface, index + 1, cp.float32)
        for index in range(8))
    before = tuple(value.copy() for value in values)
    status = cp.full((1,), cp.uint32(0xFFFFFFFF), cp.uint32)

    assert validate_kf_outputs(values, 0xFF, status) == 0
    assert int(status[0].item()) == 0
    for got, expected in zip(values, before):
        cp.testing.assert_array_equal(got, expected)

    for index, value in enumerate(values):
        value.flat[0] = cp.nan
        assert validate_kf_outputs(values, 0xFF, status) == 1 << index
        value.flat[0] = cp.float32(index + 1)

    values[-1].flat[0] = cp.inf
    values[0].flat[0] = cp.nan
    invalid = validate_kf_outputs(values, 0xFF, status)
    assert invalid & 1
    assert invalid & (1 << 7)
    values[-1].flat[0] = cp.float32(8.0)
    values[0].flat[0] = cp.float32(1.0)

    # Warm/no-separate-snow modes pass valid placeholder pointers for the
    # two inactive frozen categories; their payloads must not be read.
    values[3].flat[0] = cp.nan
    values[5].flat[0] = cp.inf
    assert validate_kf_outputs(values, 0xD7, status) == 0


@pytest.mark.gpu
@requires_gpu
def test_kf_production_binding_nca_hold_and_domain_flag():
    """Task 6b NCA persistence against the float64 driver mirror.

    Finalizes the interim Task-1/Task-4 every-call-recompute pin: a
    column with time left on its NCA timer holds its stored tendencies,
    PRATEC, RAINCV, and NCA across STEPCU events
    (module_cu_kfeta.F:410-412), RAINC accumulates PRATEC*DT on every
    dynamics step, and the stored rates are zeroed on the WRF
    NINT(NCA/DT) <= 1 boundary while the rain rate survives expiry
    (module_physics_addtendc.F:2139-2231; solve_em.F:3558-3571).
    """
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.kf import KainFritsch
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import (
        CumulusResult, _prepare_atmosphere, initialize_physics,
    )
    from gpuwm.verify.npref import np_cumulus_nca_driver_step

    def state_for(cfg):
        coord = make_vertical_coord(cfg.nz)
        base = make_base_state(
            coord, lambda z: 300.0 + 0.003 * np.asarray(z),
            p_surf=cfg.p_surf, ztop=cfg.ztop)
        return init_moist_balanced(
            cfg, coord, base,
            lambda z: 0.010 * np.exp(-np.asarray(z) / 2200.0))

    # Morrison explicitly carries all four KF hydrometeor rates through the
    # NCA hold.  The separate CPU seam regression covers WRF's latent-energy
    # warm-rain and no-separate-snow closures.
    cfg = RunConfig(
        nx=3, ny=1, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
        dt=60.0, run_seconds=0.0, moist=True, mp_physics=10, cu_physics=1,
        cudt_minutes=5.0)
    state = state_for(cfg)
    # cu_physics=1 auto-binds the production Kain-Fritsch adapter.
    driver = initialize_physics(state, cfg)
    assert isinstance(driver.cumulus_callable, KainFritsch)
    production_result = driver.cumulus_callable(
        atmosphere=_prepare_atmosphere(state), fields=driver.fields,
        state=state, cfg=cfg)
    for value in (production_result.rthcuten, production_result.rqvcuten,
                   production_result.rqccuten, production_result.rqicuten,
                   production_result.rqrcuten, production_result.rqscuten,
                   production_result.rainc, production_result.nca_seconds,
                   production_result.pratec):
        assert bool(cp.isfinite(value).all())
    # kf_init seeds the hold timer at -100 s (module_cu_kfeta.F:3154).
    cp.testing.assert_array_equal(driver.cu_nca, -100.0)

    shape = tuple(state.p.shape)
    surface = tuple(state.mup.shape)

    # Deterministic per-column scheme: column 0 fires a deep 480 s hold
    # (NIC=8 at dt=60), column 1 never triggers, and column 2 fires a
    # rain-free shallow hold of one cudt (module_cu_kfeta.F:2570-2573).
    # Outputs change per call, so a held column visibly ignores them.
    def scheme_fields(call_index):
        scale = np.float32(1.0 + 0.5 * call_index)
        fields = {}
        for name, base in (("rthcuten", 1.0e-5), ("rqvcuten", -2.0e-8),
                           ("rqccuten", 3.0e-9), ("rqicuten", 3.5e-9),
                           ("rqrcuten", 4.0e-9), ("rqscuten", 4.5e-9)):
            field = np.zeros(shape, np.float32)
            field[:, 0, 0] = np.float32(base) * scale
            field[:, 0, 2] = np.float32(2.0 * base) * scale
            fields[name] = field
        pratec = np.zeros(surface, np.float32)
        pratec[0, 0] = np.float32(2.5e-4) * scale
        nca = np.zeros(surface, np.float32)
        nca[0, 0] = 480.0
        nca[0, 2] = 300.0
        fields.update(pratec=pratec, nca_seconds=nca)
        return fields

    calls = []

    def deterministic_cumulus(**kwargs):
        fields = scheme_fields(len(calls))
        calls.append(float(kwargs["state"].elapsed_seconds))
        return CumulusResult(
            cp.asarray(fields["rthcuten"]), cp.asarray(fields["rqvcuten"]),
            rqccuten=cp.asarray(fields["rqccuten"]),
            rqicuten=cp.asarray(fields["rqicuten"]),
            rqrcuten=cp.asarray(fields["rqrcuten"]),
            rqscuten=cp.asarray(fields["rqscuten"]),
            rainc=cp.zeros(surface, cp.float32),
            nca_seconds=cp.asarray(fields["nca_seconds"]),
            pratec=cp.asarray(fields["pratec"]))

    history_calls = []
    deterministic_cumulus.update_trigger_history = lambda **kwargs: (
        history_calls.append(float(kwargs["state"].elapsed_seconds)))
    driver.cumulus_callable = deterministic_cumulus

    mirror = {
        "nca": np.full(surface, -100.0), "pratec": np.zeros(surface),
        "raincv": np.zeros(surface), "rainc": np.zeros(surface),
        **{name: np.zeros(shape) for name in
           ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
            "rqrcuten", "rqscuten")},
    }
    chm = cp.asnumpy(state.c1h[:, None, None] * state.total_mu()[None]
                     + state.c2h[:, None, None]).astype(np.float64)
    due_calls = {0.0: 0, 240.0: 1, 540.0: 2}   # ITIMESTEP 1, 5, 10
    for step_index in range(11):
        now = step_index * cfg.dt
        state.elapsed_seconds = now
        held = driver.compute(state, cfg)
        scheme = (scheme_fields(due_calls[now]) if now in due_calls
                  else None)
        mirror, applied = np_cumulus_nca_driver_step(
            mirror, scheme, dt=cfg.dt)
        # This step's RK forcing couples the pre-expiry applied rates.
        np.testing.assert_allclose(
            cp.asnumpy(held.rtheta),
            (chm * applied["rthcuten"]).astype(np.float32),
            rtol=2.0e-6, atol=0.0)
        # The expiry transition is complete before compute returns: the
        # current RK target above still contains the pre-expiry forcing, but
        # Morrison's later raw-rate inputs and the next held compose already
        # match advance_ppt's cleared state.
        for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                     "rqrcuten", "rqscuten"):
            np.testing.assert_array_equal(
                cp.asnumpy(driver.cu_rates[name]), mirror[name],
                err_msg=name)
        if now == 420.0:
            assert bool(cp.any(held.rqr[:, 0, 0] != 0.0))
            for name in ("rqrcuten", "rqicuten", "rqscuten"):
                cp.testing.assert_array_equal(
                    driver.cu_rates[name][:, 0, 0], 0.0,
                    err_msg=f"expiry-step Morrison input {name}")
        np.testing.assert_array_equal(
            cp.asnumpy(driver.cu_nca), mirror["nca"])
        np.testing.assert_array_equal(
            cp.asnumpy(driver.cu_pratec), mirror["pratec"])
        np.testing.assert_allclose(
            cp.asnumpy(driver.cu_raincv), mirror["raincv"], rtol=1.0e-6)
        np.testing.assert_allclose(
            cp.asnumpy(driver.rainc), mirror["rainc"], rtol=1.0e-5,
            atol=1.0e-9)

    # The scheme runs on every STEPCU event; holding is per-column.  The
    # trigger-history hook follows the same due calendar: WRF's cumulus
    # driver early-returns on non-due steps (module_cumulus_driver.F:
    # 830-864, RETURN at :863), so W0AVG advances once per due event.
    assert calls == [0.0, 240.0, 540.0]
    assert history_calls == [0.0, 240.0, 540.0]
    rth0 = cp.asnumpy(driver.cu_rates["rthcuten"])
    # Column 0 was held straight through the 240 s STEPCU event (its call-1
    # doubled rates were discarded), expired after the 420 s step, and was
    # readmitted at 540 s with the call-2 rates.
    np.testing.assert_allclose(rth0[:, 0, 0], np.float32(2.0e-5),
                               rtol=0.0, atol=0.0)
    # Column 2's shallow hold expired after the 240 s step and its 540 s
    # recompute reported no new trigger history here beyond the call-2
    # values; column 1 never held anything.
    np.testing.assert_allclose(rth0[:, 0, 2], np.float32(4.0e-5),
                               rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(rth0[:, 0, 1], 0.0)
    # RAINC kept accumulating at the stored rate after column 0's expiry
    # (WRF leaves PRATEC alone at expiry: addtendc.F:2216-2217 commented
    # out): nine steps at the call-0 rate plus two at the call-2 rate.
    expected_rainc = (9 * 60.0 * float(np.float32(2.5e-4))
                      + 2 * 60.0 * float(np.float32(2.5e-4) * np.float32(2.0)))
    np.testing.assert_allclose(
        float(driver.rainc[0, 0]), expected_rainc, rtol=1.0e-5)
    assert float(driver.rainc[0, 1]) == float(driver.rainc[0, 2]) == 0.0

    # A nested-domain-style independent config with cu_physics=0 does not
    # inherit d01's scheme; SFCLAY merely supplies another enabled slot so a
    # second driver can be constructed for the assertion.
    off_values = {**cfg.__dict__, "cu_physics": 0,
                  "sf_sfclay_physics": 1}
    cfg_off = RunConfig(**off_values)
    off_driver = initialize_physics(state_for(cfg_off), cfg_off)
    assert off_driver.cu_physics == 0
    assert off_driver.cumulus_callable is None


@pytest.mark.gpu
@requires_gpu
def test_kf_trigger_history_advances_per_due_call_and_scheme_consumes_it(
        monkeypatch):
    """Pin W0AVG cadence: one recursive-mean sample per STEPCU due event.

    WRF's cumulus driver early-returns on non-due steps
    (module_cumulus_driver.F:830-864, RETURN at :863), so the fixed-step
    W0AVG loop (module_cu_kfeta.F:232-250) runs once per due call with
    weight 1/TST, TST = 2*STEPCU on the model clock (controller
    re-adjudication 2026-07-16 superseding the T1 every-step hook).
    """
    import cupy as cp

    from gpuwm.core.kf import KFPhaseMode, KainFritsch

    shape = (8, 1, 1)
    atmosphere = {
        name: cp.ones(shape, dtype=cp.float32)
        for name in ("u", "v", "temperature", "qv", "qc", "pressure",
                     "exner", "dz")
    }
    state = SimpleNamespace(
        w=cp.arange(9, dtype=cp.float32).reshape(9, 1, 1),
        p=cp.ones(shape, dtype=cp.float32),
    )

    def scratch(requested_shape, _name):
        return cp.zeros(requested_shape, dtype=cp.float32)

    state.scratch = scratch
    captured = {}

    def fake_launch(*args, **kwargs):
        captured["w"] = args[8].copy()
        captured.update(kwargs)
        zeros = cp.zeros(shape, dtype=cp.float32)
        return {
            "rthcuten": zeros, "rqvcuten": zeros.copy(),
            "rqccuten": zeros.copy(), "rqicuten": zeros.copy(),
            "rqrcuten": zeros.copy(), "rqscuten": zeros.copy(),
            "rainc": cp.zeros((1, 1), dtype=cp.float32),
            "nca_seconds": cp.zeros((1, 1), dtype=cp.float32),
        }

    monkeypatch.setattr("gpuwm.core.kf.launch_kf", fake_launch)
    cfg = SimpleNamespace(cudt_minutes=5.0, dt=60.0, dx=12000.0,
                          mp_physics=10)
    scheme = KainFritsch()
    expected = cp.zeros(shape, dtype=cp.float32)
    # Three consecutive STEPCU due events (cudt=5 min apart): each hook
    # invocation advances the mean once with TST = 2*STEPCU = 10.
    for event, offset in enumerate((0.0, 2.0, -1.0)):
        state.elapsed_seconds = event * 300.0
        state.w[...] = (cp.arange(9, dtype=cp.float32)
                        .reshape(9, 1, 1) + offset)
        instantaneous = 0.5 * (state.w[:-1] + state.w[1:])
        expected = (expected * 9.0 + instantaneous) / 10.0
        scheme.update_trigger_history(state=state, cfg=cfg)
        cp.testing.assert_array_equal(scheme.w0avg, expected)

    scheme(atmosphere=atmosphere, fields={}, state=state, cfg=cfg)

    # The scheme call at the same elapsed time consumes the already-updated
    # mean without double-counting the sample.
    cp.testing.assert_array_equal(captured["w"], expected)
    assert captured["cudt"] == 300.0
    assert captured["dt"] == 60.0
    assert captured["phase_mode"] == KFPhaseMode.SEPARATE_ICE_SNOW

    # Under the compatibility substep integration (clock_dt > dt) a due
    # call still advances the mean exactly once with the CLOCK-derived
    # TST = 2*NINT(cudt*60/clock_dt) = 10 (not 2*40 from the substep dt),
    # and the scheme receives the model-clock DT (WRF hands KF its model
    # DT, module_cumulus_driver.F:1028).
    sub_cfg = SimpleNamespace(cudt_minutes=5.0, dt=7.5, clock_dt=60.0,
                              dx=12000.0, mp_physics=10)
    sub_scheme = KainFritsch()
    state.w[...] = cp.arange(9, dtype=cp.float32).reshape(9, 1, 1)
    instantaneous = 0.5 * (state.w[:-1] + state.w[1:])
    state.elapsed_seconds = 292.5   # substep-calendar due instant
    sub_scheme.update_trigger_history(state=state, cfg=sub_cfg)
    cp.testing.assert_array_equal(sub_scheme.w0avg, instantaneous / 10.0)
    sub_scheme(atmosphere=atmosphere, fields={}, state=state, cfg=sub_cfg)
    cp.testing.assert_array_equal(captured["w"], instantaneous / 10.0)
    assert captured["dt"] == 60.0
    assert captured["cudt"] == 300.0
    assert captured["phase_mode"] == KFPhaseMode.SEPARATE_ICE_SNOW
