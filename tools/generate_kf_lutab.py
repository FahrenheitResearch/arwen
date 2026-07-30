"""Generate the packaged WRF v4.6.1 Kain--Fritsch lookup table.

Which precision, and which libm
-------------------------------
``KF_LUTAB`` (``phys/module_cu_kfeta.F:3174-3301``) declares every table
variable as default ``REAL``, and the WRF build does not promote it: WRF v4.6.1
ships ``NATIVE_RWORDSIZE = 4`` (``arch/preamble:63``) and every
``PROMOTION`` line in ``arch/configure.defaults`` has ``-fdefault-real-8``
commented out.  So the table WRF actually builds is **binary32 throughout**,
and matching it means matching single-precision elementary functions, not
rounding a double result.

On the reference toolchain (gfortran 13.3.0 / glibc 2.39) the three
transcendentals in this routine lower to *scalar* glibc calls -- ``nm -u`` on
the oracle binary shows exactly ``expf``, ``logf``, ``powf`` (all
``@GLIBC_2.27``), at ``-O0`` and at ``-O2 -ftree-vectorize`` alike.  The
Newton loop is serially dependent, so gfortran's vectoriser never reaches for
libmvec here and the two builds agree bit-for-bit.

None of those three is correctly rounded, and none of them is what NumPy
computes.  Building this table on ``np.exp``/``np.power``/``np.log`` at
``float32`` therefore produced a table that was
:strong:`both` wrong against WRF and unstable across NumPy releases -- see
``gpuwm/data/kf_lutab/PROVENANCE.md`` for the measured size of each error.
This module instead calls the glibc 2.39 transcriptions in
:mod:`gpuwm.core.noahmp_libm`, which reproduce the table bit-for-bit and
depend on NumPy for nothing but storage and correctly-rounded ``+ - * /``.

**Do not reintroduce a NumPy transcendental here.**  ``tests/test_kf.py``
fails the build if one appears.

(``noahmp_libm`` asks to be promoted to a shared module once a second lane
needs it.  This is that second lane, but renaming it would move a file the
Noah-MP lane owns, so the import stays as-is and the promotion is left to
whoever holds both.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.core.noahmp_libm import expf, logf, powf


F32 = np.float32


def saturation_theta_e(
        temperature: np.float32, pressure: np.float32,
) -> tuple[np.float32, np.float32]:
    """Evaluate one WRF default-``REAL`` saturated parcel state.

    Transcribes ``module_cu_kfeta.F:3236-3241`` (and its twin at 3255-3260,
    3266-3270).  Operand order is WRF's; every ``+ - * /`` is a correctly
    rounded binary32 operation and every transcendental is the glibc 2.39
    ``float`` entry point gfortran emits for ``REAL(4)``.
    """
    temperature = F32(temperature)
    pressure = F32(pressure)
    aliq = F32(F32(0.6112) * F32(1000.0))
    bliq = F32(17.67)
    cliq = F32(bliq * F32(273.15))
    dliq = F32(29.65)
    exponent = F32((bliq * temperature - cliq) / (temperature - dliq))
    es = F32(aliq * expf(exponent))
    qs = F32(F32(0.622) * es / (pressure - es))
    pi_exponent = F32(F32(0.2854) * (F32(1.0) - F32(0.28) * qs))
    pi = powf(F32(F32(1.0e5) / pressure), pi_exponent)
    theta_exponent = F32(
        (F32(3374.6525) / temperature - F32(2.5403))
        * qs * (F32(1.0) + F32(0.81) * qs))
    theta_e = F32(temperature * pi * expf(theta_exponent))
    return theta_e, qs


def generate() -> dict[str, np.ndarray]:
    """Transcribe ``KF_LUTAB`` default-``REAL`` arithmetic exactly.

    WRF source ``module_cu_kfeta.F:3174-3301`` constructs the table through
    scalar single-precision loops.  Repeated increments and the final
    in-loop ``QS`` store are intentional; constructing in float64 and casting
    afterward produces a different table.
    """
    kfnt, kfnp = 250, 220
    tmin, dth = F32(150.0), F32(1.0)
    pressure_top, pressure_bottom = F32(5000.0), F32(110000.0)
    dpr = F32((pressure_bottom - pressure_top) / F32(kfnp - 1))
    thetae_base = np.empty(kfnp, np.float32)
    pressure = F32(pressure_top - dpr)
    for kp in range(kfnp):
        pressure = F32(pressure + dpr)
        thetae_base[kp], _ = saturation_theta_e(tmin, pressure)

    temperature = np.empty((kfnt, kfnp), np.float32)
    qsat = np.empty_like(temperature)
    pressure = F32(pressure_top - dpr)
    for kp in range(kfnp):
        thetae_wanted = F32(thetae_base[kp] - dth)
        pressure = F32(pressure + dpr)
        for it in range(kfnt):
            thetae_wanted = F32(thetae_wanted + dth)
            guess = tmin if it == 0 else temperature[it - 1, kp]
            f0 = F32(saturation_theta_e(guess, pressure)[0]
                     - thetae_wanted)
            t0 = F32(guess)
            t1 = F32(guess - F32(0.5) * f0)
            qs = F32(0.0)
            for _ in range(11):
                thetae, qs = saturation_theta_e(t1, pressure)
                f1 = F32(thetae - thetae_wanted)
                if abs(f1) < F32(0.001):
                    break
                delta = F32(f1 * F32(t1 - t0) / F32(f1 - f0))
                t0, f0 = t1, f1
                t1 = F32(t1 - delta)
            temperature[it, kp] = t1
            qsat[it, kp] = qs

    log_ratio = np.empty(200, np.float32)
    start, increment = F32(0.001), F32(0.075)
    ratio = F32(start - increment)
    for index in range(log_ratio.size):
        ratio = F32(ratio + increment)
        log_ratio[index] = logf(ratio)
    return {
        "temperature": temperature,
        "qsat": qsat,
        "thetae_base": thetae_base,
        "log_ratio": log_ratio,
        "pressure_top": np.asarray(pressure_top, dtype=np.float32),
        "pressure_reciprocal": np.asarray(F32(F32(1.0) / dpr),
                                           dtype=np.float32),
        "thetae_reciprocal": np.asarray(F32(F32(1.0) / dth),
                                         dtype=np.float32),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    destination = root / "gpuwm" / "data" / "kf_lutab" / "kf_lutab.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **generate())
