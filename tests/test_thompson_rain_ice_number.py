"""make_RainNumber and make_IceNumber against scalar Fortran transcriptions.

WRF v4.7.1 dyn_em/module_initialize_real.F:9163-9194 and :9044-9114, the
functions real.exe uses to give analysed rain and ice mass a number where
the analysis carries none (:4840-4852).  Each transcription below is
written statement by statement, one value at a time, with the REAL/DOUBLE
declarations of the Fortran, and the vectorised port must agree with it
bit for bit across the intercept ramp, both ends of the ice table and the
temperatures between its rows.
"""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gpuwm.core.noahmp_libm import powf
from gpuwm.core.thompson_entry import (
    ICE_RETAB, ice_mean_diameter_m, make_ice_number, make_rain_number,
    rain_median_volume_diameter_m,
)

F = np.float32


def _fortran_make_rain_number(q_rain, temp):
    """One call of make_RainNumber, line for line."""
    q_rain, temp = F(q_rain), F(temp)
    pi = F(3.1415926536)                                  # :9169
    am_r = F(F(pi * F(1000.0)) / F(6.0))                  # :9170
    n0 = float(F(8.0e6))                                  # :9181
    if temp <= F(271.15):                                 # :9183
        n0 = float(F(8.0e8))                              # :9184
    elif temp > F(271.15) and temp < F(273.15):           # :9185
        # 10**(REAL): the INTEGER base becomes a REAL, a REAL power, which
        # gfortran hands to glibc's powf (never NumPy's power loop, which
        # rounds differently on an AVX-512 Linux host, A143).
        n0 = float(F(F(8.0) * powf(F(10.0), F(F(279.15) - temp))))
    lam = math.sqrt(math.sqrt(n0 * float(am_r) * 6.0 / float(q_rain)))
    first = float(F(q_rain / F(6.0)))                     # :9190, REAL
    qnr = first * lam * lam * lam / float(am_r)           # :9190, DOUBLE
    return F(qnr)                                         # :9191 SNGL


def _fortran_make_ice_number(q_ice, temp):
    """One call of make_IceNumber, line for line."""
    q_ice, temp = F(q_ice), F(temp)
    ice_density = F(890.0)                                # :9047
    pi = F(3.1415926536)                                  # :9048
    idx_rei = int(F(temp - F(179.0)))                     # :9085, INT
    idx_rei = min(max(idx_rei, 1), 94)                    # :9086
    corr = F(temp - F(int(temp)))                         # :9087
    reice = F(F(ICE_RETAB[idx_rei - 1] * F(F(1.0) - corr))
              + F(ICE_RETAB[idx_rei] * corr))             # :9088
    deice = F(F(F(2.0) * reice) * F(1.0e-6))              # :9089
    lam = float(F(F(3.0) / deice))                        # :9101
    number = (float(q_ice) * lam * lam * lam
              / float(F(pi * ice_density)))               # :9102
    return F(number)


#: Per-volume mass (kg m^-3), from a trace to a heavy core.
_MASS = (1.0e-9, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 5.0e-3)
#: Temperatures (K) across the rain intercept ramp (271.15..273.15), its
#: two ends, whole kelvins, both ends of the ice table and past them.
_TEMPERATURE = (150.0, 178.5, 179.0, 180.25, 205.7, 223.15, 250.0, 263.15,
                270.0, 271.15, 271.1501, 271.6, 272.0, 272.15, 272.9,
                273.15, 273.16, 275.0, 290.0, 305.0)


@pytest.mark.parametrize("port, fortran", [
    (make_rain_number, _fortran_make_rain_number),
    (make_ice_number, _fortran_make_ice_number),
])
def test_port_matches_the_fortran_bit_for_bit(port, fortran):
    q, t = np.meshgrid(np.asarray(_MASS, dtype=np.float32),
                       np.asarray(_TEMPERATURE, dtype=np.float32),
                       indexing="ij")
    got = port(q, t)
    assert got.dtype == np.float32
    reference = np.array(
        [[fortran(qq, tt) for tt in _TEMPERATURE] for qq in _MASS],
        dtype=np.float32)
    np.testing.assert_array_equal(got.view(np.uint32),
                                  reference.view(np.uint32))


def test_ice_table_is_the_95_rows_the_function_carries():
    assert ICE_RETAB.shape == (95,) and ICE_RETAB.dtype == np.float32
    assert float(ICE_RETAB[0]) == pytest.approx(5.92779)
    assert float(ICE_RETAB[-1]) == pytest.approx(250.639)
    assert float(ICE_RETAB[66]) == pytest.approx(71.2885)


def test_supercooled_rain_starts_as_drizzle_and_warm_rain_as_rain():
    """0.1 g/m3: about 0.3 mm at or below -2 C, 0.9 mm above 0 C."""
    rr = np.float32(1.0e-4)
    cold = make_rain_number(rr, np.float32(265.0))
    warm = make_rain_number(rr, np.float32(280.0))
    assert float(rain_median_volume_diameter_m(rr, cold)) == pytest.approx(
        0.29e-3, abs=0.01e-3)
    assert float(rain_median_volume_diameter_m(rr, warm)) == pytest.approx(
        0.92e-3, abs=0.01e-3)
    # Marshall-Palmer: N = N0/lambda, so the cold number is 10**(2*3/4)
    # times the warm one at the same mass.
    assert float(cold) / float(warm) == pytest.approx(10.0 ** 1.5, rel=1e-5)
    ramp = make_rain_number(np.full(5, rr),
                            np.linspace(271.2, 273.1, 5, dtype=np.float32))
    assert np.all(np.diff(ramp.astype(np.float64)) < 0.0)
    assert float(ramp.max()) < float(cold) and float(ramp.min()) > float(warm)


def test_ice_size_follows_temperature_as_the_fortran_comment_says():
    """:9108-9110: 0.1 g/kg at -10 C is about 28 crystals per litre."""
    number = make_ice_number(np.float32(1.0e-4), np.float32(263.15))
    assert float(number) == pytest.approx(28122.0, rel=2.0e-3)
    size = ice_mean_diameter_m(1.0e-4, number)
    assert float(size) == pytest.approx(2.0 * 162.52e-6, rel=1.0e-3)
    colder = make_ice_number(np.float32(1.0e-4), np.float32(223.15))
    assert float(colder) > 50.0 * float(number)


#: NumPy's AVX-512 dispatch targets across its releases (2.4 renamed the
#: group X86_V4); the ones this host's NumPy actually dispatches to are
#: what the test below switches off.
_AVX512_TARGETS = ("X86_V4", "AVX512F", "AVX512CD", "AVX512_KNL",
                   "AVX512_KNM", "AVX512_SKX", "AVX512_CLX", "AVX512_CNL",
                   "AVX512_ICL", "AVX512_SPR")

#: The cold-start number writers over one fixed synthetic field (half of
#: it on make_RainNumber's 271.15..273.15 K ramp), printed as one digest
#: per output.  Run in a fresh interpreter, because NumPy reads
#: NPY_DISABLE_CPU_FEATURES once, at import.
_WRITER_PROBE = """
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from gpuwm.core import thompson_entry as te
rng = np.random.default_rng(20260930)
n = 60000
temp = np.concatenate([rng.uniform(180.0, 310.0, n // 2),
                       rng.uniform(271.15, 273.15, n // 2)]).astype(np.float32)
q = (10.0 ** rng.uniform(-9.0, -2.0, n)).astype(np.float32)
rho = rng.uniform(0.3, 1.3, n).astype(np.float32)
number = np.where(rng.random(n) < 0.35, np.float32(0.0),
                  (10.0 ** rng.uniform(-3.0, 9.2, n)).astype(np.float32))
out = {"make_rain_number": te.make_rain_number(q, temp),
       "make_ice_number": te.make_ice_number(q, temp)}
seeded = {"rain": out["make_rain_number"], "ice": out["make_ice_number"],
          "cloud": number}
for species in ("cloud", "rain", "ice"):
    out["entry_" + species] = te.np_thompson_entry_numbers(
        species, q / rho, seeded[species] / rho, rho)
print(json.dumps({k: hashlib.sha256(np.ascontiguousarray(v).tobytes())
                  .hexdigest() for k, v in out.items()}))
"""


def _dispatched_avx512_targets():
    try:
        from numpy._core import _multiarray_umath as umath
    except ImportError:                              # NumPy 1.x
        from numpy.core import _multiarray_umath as umath
    baseline = set(umath.__cpu_baseline__)
    return [name for name in _AVX512_TARGETS
            if name in umath.__cpu_dispatch__ and name not in baseline
            and umath.__cpu_features__.get(name)]


def test_the_cold_start_number_writers_do_not_move_with_numpys_avx512_loops():
    """The Thompson cold-start numbers are the same bits with NumPy's
    AVX-512 loops on and off.

    A143: NumPy 2.5's float32 and float64 ``power`` loops round differently
    on an AVX-512 Linux host, the CPU family of the product boxes, so the
    seeded rain number (``10**(279.15-T)`` on the -2..0 C ramp) and the
    mp=28 droplet number took the host's bits into every preparation made
    there.  On a host with no AVX-512 dispatch there is nothing to switch
    off, and the test says so rather than passing.
    """
    targets = _dispatched_avx512_targets()
    if not targets:
        pytest.skip("this host's NumPy dispatches no AVX-512 loop, so there "
                    "is no second arm to compare")
    root = str(Path(__file__).resolve().parents[1])

    def digests(disabled):
        env = {name: value for name, value in os.environ.items()
               if name != "NPY_DISABLE_CPU_FEATURES"}
        env["GPUWM_NO_LOCAL_GPU"] = "1"
        if disabled:
            env["NPY_DISABLE_CPU_FEATURES"] = " ".join(disabled)
        run = subprocess.run([sys.executable, "-c", _WRITER_PROBE, root],
                             capture_output=True, text=True, cwd=root,
                             env=env, check=True)
        return json.loads(run.stdout.strip().splitlines()[-1])

    on = digests(())
    off = digests(targets)
    moved = sorted(name for name in on if on[name] != off[name])
    assert not moved, (
        f"{moved} change with NumPy's AVX-512 loops ({targets}) switched off")
