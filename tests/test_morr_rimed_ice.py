"""CPU and structural regressions for Morrison's hail/graupel switch.

WRF v4.6.1 makes ``morr_rimed_ice`` a scalar ``&physics`` option with
Registry default 1 (hail).  ``MORR_TWO_MOMENT_INIT`` then selects AG/BG and
RHOG before deriving CG and every BG-dependent gamma/exponent product:
Registry/Registry.EM_COMMON:2663-2666 and
phys/module_mp_morr_two_moment.F:337-411, :483-510.
"""

from __future__ import annotations

import ctypes
import math
import shutil
import subprocess
from pathlib import Path
import tomllib

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config


REPO = Path(__file__).resolve().parents[1]


def _config(**updates) -> RunConfig:
    values = dict(nx=3, ny=2, nz=4, dx=1000.0, dy=1000.0,
                  ztop=4000.0, dt=6.0, run_seconds=0.0,
                  moist=True, mp_physics=10)
    values.update(updates)
    return RunConfig(**values)


def test_run_config_defaults_to_wrf_hail_and_validates_switch():
    assert _config().morr_rimed_ice == 1
    assert validate_run_config(_config(morr_rimed_ice=0)).morr_rimed_ice == 0
    for bad in (-1, 2):
        with pytest.raises(ValueError, match="morr_rimed_ice"):
            validate_run_config(_config(morr_rimed_ice=bad))


def test_central_rimed_ice_constants_match_wrf_branches():
    from gpuwm.core.morrison_constants import rimed_ice_constants

    graupel = rimed_ice_constants(0)
    hail = rimed_ice_constants(1)
    assert (graupel.ag, graupel.bg, graupel.rhog) == (19.3, 0.37, 400.0)
    assert (hail.ag, hail.bg, hail.rhog) == (114.5, 0.5, 900.0)
    assert graupel.cg == pytest.approx(400.0 * math.pi / 6.0)
    assert hail.cg == pytest.approx(900.0 * math.pi / 6.0)
    with pytest.raises(ValueError, match="morr_rimed_ice"):
        rimed_ice_constants(7)


def test_documented_morrison_constants_default_hail_and_retain_graupel():
    with (REPO / "gpuwm" / "data" / "morrison" /
          "constants.toml").open("rb") as stream:
        constants = tomllib.load(stream)
    assert constants["switches"]["ihail"] == 1
    assert constants["rimed_ice"]["hail"] == {
        "ag": 114.5, "bg": 0.5, "rhog": 900.0}
    assert constants["rimed_ice"]["graupel"] == {
        "ag": 19.3, "bg": 0.37, "rhog": 400.0}


def test_numpy_graupel_fall_speed_uses_selected_wrf_constants():
    from gpuwm.core.morrison_constants import rimed_ice_constants
    from gpuwm.verify.npref import (_MORR_RHOSU,
                                    _np_morrison_fall_speeds)

    q = {name: np.zeros(1) for name in "crisg"}
    n = {name: np.zeros(1) for name in "crisg"}
    q["g"][0] = 1.0e-3
    n["g"][0] = 1.0e4
    rho = np.ones(1)
    temperature = np.full(1, 270.0)

    for option in (0, 1):
        selected = rimed_ice_constants(option)
        vm, vn, _ = _np_morrison_fall_speeds(
            "g", q, n, rho, temperature, morr_rimed_ice=option)
        lam = (selected.rhog * math.pi * n["g"][0] / q["g"][0]) ** (1 / 3)
        an = selected.ag * (_MORR_RHOSU / rho[0]) ** 0.54
        expected_vm = min(
            an * math.gamma(4.0 + selected.bg) / 6.0
            / lam ** selected.bg,
            20.0 * (_MORR_RHOSU / rho[0]) ** 0.54)
        expected_vn = min(
            an * math.gamma(1.0 + selected.bg) / lam ** selected.bg,
            20.0 * (_MORR_RHOSU / rho[0]) ** 0.54)
        np.testing.assert_allclose(vm, [expected_vm], rtol=1.0e-14)
        np.testing.assert_allclose(vn, [expected_vn], rtol=1.0e-14)


def test_radar_init_defaults_hail_and_keeps_explicit_graupel():
    from gpuwm.core.refl import radar_init

    assert radar_init().xam_g == pytest.approx(900.0 * math.pi / 6.0)
    assert radar_init(0).xam_g == pytest.approx(400.0 * math.pi / 6.0)


def test_cuda_sources_receive_selected_rimed_ice_constants_at_runtime():
    morrison = (REPO / "gpuwm" / "core" / "kernels" /
                "morrison.cu").read_text(encoding="utf-8")
    reflectivity = (REPO / "gpuwm" / "core" / "kernels" /
                    "refl.cu").read_text(encoding="utf-8")
    assert "#define MRHOG 400.0f" not in morrison
    assert "19.3f" not in morrison
    assert "0.37f" not in morrison
    assert "morr_ag" in morrison
    assert "morr_bg" in morrison
    assert "morr_rhog" in morrison
    assert "#define RXAM_G (400.0" not in reflectivity
    assert "const double xam_g" in reflectivity


#: CUDA language shims for building a ``.cu`` as host C++, the same trick
#: ``tools/gf_wrf461_oracle/gf_host_harness.cpp`` uses on gf.cu.  x86-64 SSE
#: evaluates float arithmetic with the same IEEE-754 round-to-nearest
#: semantics, and ``-ffp-contract=off`` keeps it unfused.
_HOST_SHIMS = r"""
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#define __device__
#define __global__
#define __constant__ static const
#define __forceinline__ inline
#define __restrict__
struct MS3 { unsigned x, y, z; };
static MS3 blockIdx = {0, 0, 0};
static MS3 blockDim = {1, 1, 1};
static MS3 threadIdx = {0, 0, 0};
static MS3 gridDim = {1, 1, 1};
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline float __uint_as_float(unsigned int u)
{ float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned int __float_as_uint(float f)
{ unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int i)
{ float f; std::memcpy(&f, &i, 4); return f; }
static inline long long __double_as_longlong(double d)
{ long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l)
{ double d; std::memcpy(&d, &l, 8); return d; }
static inline float __double2float_rn(double d) { return (float)d; }
using std::max; using std::min;
#include "morrison_assembled.cu"

extern "C" float morr_probe_pgam(float qc, float nc, float rhoa,
                                 float pres, float temp)
{
    float n = nc, nr = 0.0f, ni = 0.0f, ns = 0.0f, ng = 0.0f;
    MorrMoments m = morr_bound(qc, 0.0f, 0.0f, 0.0f, 0.0f, rhoa,
                               pres, temp, &n, &nr, &ni, &ns, &ng,
                               false, 400.0f);
    return m.pg;
}
"""


def test_pgam_reference_density_is_rebuilt_from_the_current_temperature(
        tmp_path):
    """``module_mp_morr_two_moment.F:3920`` reads T3D, not the entry density.

    All four PGAM sites in the authority spell the reference density the
    same way -- ``DUM = PRES(K)/(287.15*T3D(K))`` at ``:1558``, ``:2182``,
    ``:3405`` and ``:3920`` -- and ``T3D(K)`` is a live variable.  By
    ``:3920`` it has moved: ``:3710`` applies the tendencies, ``:3735-3758``
    the sedimentation evaporation, and ``:3805-3843`` the ice melt and both
    homogeneous freezings.  ``RHO(K)`` by contrast is built ONCE at ``:1325``
    from the pre-melt T3D and never rebuilt.

    ``morr_bound`` took a ``temp`` argument and never read it, deriving the
    density from ``rhoa`` -- which is ``RHO(K)`` -- instead.  That is a wrong
    INPUT, not a rounding residue: the two codes evaluated a different
    function of a different state.  What escapes the routine is ``effc``,
    the cloud effective radius, since ``LAMC``'s clip bounds and ``NC3D``
    are overwritten three statements later.

    The gate has a source half that always runs and a numerical half that
    runs wherever a C++17 toolchain exists.  The numerical half builds the
    REAL kernel source as host C++ -- gf.cu's own no-GPU trick -- and calls
    ``morr_bound``, so it grades the artifact rather than a restatement.
    """
    kernel = (Path(__file__).parents[1] / "gpuwm" / "core" / "kernels"
              / "morrison.cu").read_text(encoding="utf-8")
    body = kernel[kernel.index("MorrMoments morr_bound("):
                  kernel.index("morr_terminal_velocity(")]
    assert "real rho_cloud = pres / (287.15f * temp);" in body, (
        "morr_bound no longer rebuilds WRF's PGAM density from the current "
        "temperature")
    assert "rhoa * RD" not in body, (
        "morr_bound is back to deriving the PGAM density from the entry-time "
        "rhoa, which is RHO(K) frozen at WRF :1325")

    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if cxx is None:                                  # pragma: no cover
        pytest.skip("no C++17 toolchain for the host build of morrison.cu")

    from gpuwm.core.kernels import module_source
    (tmp_path / "morrison_assembled.cu").write_text(
        module_source("morrison"), encoding="utf-8", newline="\n")
    source = tmp_path / "morr_probe.cpp"
    source.write_text(_HOST_SHIMS, encoding="utf-8", newline="\n")
    out = tmp_path / "morr_probe.so"
    built = subprocess.run(
        [cxx, "-O2", "-std=c++17", "-ffp-contract=off",
         "-fno-unsafe-math-optimizations", "-shared", "-fPIC",
         "-I", str(tmp_path), str(source), "-o", str(out), "-lm"],
        capture_output=True, check=False)
    if not out.exists():                             # pragma: no cover
        pytest.skip("morrison.cu does not build as host C++ here: "
                    + built.stderr.decode("utf-8", "replace")[-800:])

    library = ctypes.CDLL(str(out))
    library.morr_probe_pgam.restype = ctypes.c_float
    library.morr_probe_pgam.argtypes = (ctypes.c_float,) * 5

    # A cloudy cell whose temperature moved 5 K between entry and the final
    # PSD reconstruction -- latent release plus melt, an ordinary step.
    f32 = np.float32
    pres, t_entry, t_post = f32(85000.0), f32(273.15), f32(268.15)
    rhoa = f32(pres / (f32(287.0) * t_entry))
    qc, nc = f32(1.0e-3), f32(f32(250.0e6) / rhoa)

    def pgam(density):
        p = f32(f32(0.0005714) * (nc / f32(1.0e6) * density) + f32(0.2714))
        return min(max(f32(f32(1.0) / (p * p) - f32(1.0)), f32(2.0)),
                   f32(10.0))

    wrf = pgam(f32(pres / (f32(287.15) * t_post)))
    entry_density = pgam(f32(rhoa * f32(287.0) / f32(287.15)))
    assert wrf.tobytes() != entry_density.tobytes(), (
        "the two densities give the same PGAM here, so the gate cannot tell "
        "them apart")

    got = f32(library.morr_probe_pgam(qc, nc, rhoa, pres, t_post))
    assert got.tobytes() == wrf.tobytes(), (
        f"morr_bound returned PGAM={got!r}; WRF's PRES/(287.15*T3D) gives "
        f"{wrf!r} and the entry-time density gives {entry_density!r}")
