"""WDM6 rain that carries mass but no number cannot condense vapour away (A144).

THE BREAKAGE THIS PREVENTS.  Both WDM6 convective cases failed the full-state
health gate at step 4, ``qv < 0``, on the base engine.  The cause is WRF's own
rain evaporation branch at ``module_mp_wdm6.F:1240-1256``: with ``nr = 0`` the
rate is ``-0``, ``-0 < 0`` is false, and the condensation cap ``min(prevp,
satdt/2)`` in subsaturated air becomes an evaporation of half the saturation
deficit, unbounded by the rain there is.  Ice deposition counts that vapour
source in ``supice`` and the rain limiter never revisits it, so vapour goes
negative.  WRF v4.6.1's Fortran does the same on the captured columns
(docs/wdm6_oracle_known_deltas.md, section 6).  The kernel keeps the cap and
never lets it change the rate's sign.

``tests/data/wdm6_numberless_rain_columns.npz`` holds the two real columns the
step-4 call failed on (HRRR 2026-06-14 19Z, 3 km, 49 levels, one with
Grell-Freitas cumulus and one without), as the kernel received them: fields
in wdm6_column argument order, theta, qv, qc, qi, qr, qs, qg, nn, nc, nr,
rho, p, pii, dz.

Every test compiles the SHIPPED kernel source for the host, and a twin with
only that one statement put back to WRF's, so each assertion that the defect
is gone is paired with one that the fixture still produces it.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from conftest import requires_gpu

_ROOT = Path(__file__).resolve().parents[1]
_KERNEL = _ROOT / "gpuwm/core/kernels/wdm6.cu"
_COLUMNS = _ROOT / "tests/data/wdm6_numberless_rain_columns.npz"
_SHIPPED = "prevp = fmaxf(fminf(prevp, satdt / 2.0f), 0.0f);"
_WRF = "prevp = fminf(prevp, satdt / 2.0f);"

_WRAPPER = r'''
extern "C" unsigned column(int nz, float dt, float xland, float* f, float* surface) {
    float rainnc = 0, rainncv = 0, snownc = 0, snowncv = 0;
    float graupelnc = 0, graupelncv = 0, sr = 0;
    float effc[WDM6_KMAX], effi[WDM6_KMAX], effs[WDM6_KMAX];
    unsigned flags = wdm6_column_impl(
        f, f + nz, f + 2 * nz, f + 3 * nz, f + 4 * nz, f + 5 * nz, f + 6 * nz,
        f + 7 * nz, f + 8 * nz, f + 9 * nz, f + 10 * nz, f + 11 * nz,
        f + 12 * nz, f + 13 * nz, &xland, &rainnc, &rainncv, &snownc,
        &snowncv, &graupelnc, &graupelncv, &sr, effc, effi, effs, dt, 0, nz,
        1, 0);
    surface[0] = rainncv; surface[1] = snowncv; surface[2] = graupelncv;
    return flags;
}
'''


def _build(directory: Path, name: str, source: str):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("the CPU mirror needs a C++ compiler")
    kernel = directory / f"{name}.cu"
    kernel.write_text(source, encoding="utf-8")
    wrapper = directory / f"{name}.cpp"
    wrapper.write_text('#define WDM6_CPU_MIRROR\n#include "' + kernel.as_posix()
                       + '"\n' + _WRAPPER, encoding="utf-8")
    library = directory / f"{name}.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC",
                    "-ffp-contract=off", str(wrapper), "-o", str(library)],
                   check=True, capture_output=True, text=True)
    function = ctypes.CDLL(str(library)).column
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float,
                         array, array]
    function.restype = ctypes.c_uint

    def run(columns: np.ndarray, dt: float = 15.0, xland: float = 1.0):
        """``columns`` is (nz, 14) in kernel argument order; returns a copy."""
        fields = np.ascontiguousarray(np.asarray(columns, np.float32).T.copy())
        surface = np.zeros(3, np.float32)
        flags = function(fields.shape[1], dt, xland, fields.reshape(-1), surface)
        assert flags == 0, flags
        return np.ascontiguousarray(fields.T), surface
    return run


@pytest.fixture(scope="module")
def kernels(tmp_path_factory):
    source = _KERNEL.read_text(encoding="utf-8")
    # The one statement this file is about, exactly once: the twin below is
    # only WRF's arithmetic if the substitution cannot land anywhere else.
    assert source.count(_SHIPPED) == 1
    assert source.count(_WRF) == 0
    directory = tmp_path_factory.mktemp("wdm6-numberless-rain")
    return (_build(directory, "shipped", source),
            _build(directory, "wrf", source.replace(_SHIPPED, _WRF)))


def _column(nz: int = 4) -> np.ndarray:
    """A thin cold upper-air column: 150 hPa, 203 K, ice subsaturated.

    The values follow the captured cell (qv 1e-6, qi 5e-6, T 203 K): vapour
    is a twelfth of ice saturation and water saturation is twice ice
    saturation, the regime where WRF's zero-rate cap overdraws vapour.
    """
    columns = np.zeros((nz, 14), np.float32)
    p = np.linspace(15000.0, 12000.0, nz).astype(np.float32)
    t = np.full(nz, 203.0, np.float32)
    pii = (p / 1.0e5) ** (287.0 / 1004.5)
    columns[:, 0] = t / pii
    columns[:, 1] = 1.0e-6
    columns[:, 3] = 5.0e-6
    columns[:, 7] = 1.0e8
    columns[:, 10] = p / (287.0 * t)
    columns[:, 11] = p
    columns[:, 12] = pii
    columns[:, 13] = 500.0
    return columns


def test_the_captured_step_four_columns_keep_vapour_nonnegative(kernels):
    shipped, wrf = kernels
    data = np.load(_COLUMNS)
    dt = float(data["dt"])
    for columns, (k, _, _), xland in zip(data["columns"], data["cells"],
                                         data["xland"]):
        before, _ = wrf(columns, dt, float(xland))
        after, _ = shipped(columns, dt, float(xland))
        # Teeth: WRF's statement still drives these real columns negative.
        assert before[:, 1].min() < 0.0
        assert np.isfinite(after).all()
        assert after[:, 1].min() >= 0.0
        assert after[k, 1] > 0.0


def test_numberless_tiny_rain_in_dry_cold_air_does_not_feed_deposition(kernels):
    shipped, wrf = kernels
    dry, _ = shipped(_column())
    columns = _column()
    columns[:, 4] = 1.0e-13          # tiny rain mass
    columns[:, 9] = 0.0              # and no rain number
    before, _ = wrf(columns)
    after, _ = shipped(columns)
    # Teeth: WRF's statement deposits ice from vapour the air does not have.
    assert before[:, 1].min() < 0.0
    assert (before[:, 3] > dry[:, 3]).any()
    # The shipped kernel treats the numberless rain as no evaporation at
    # all: the ice is the rain-free column's to the bit, and vapour differs
    # only by the 1e-13 of rain that joins the cloud water in the di82
    # collapse and evaporates there.
    assert after[:, 1].min() >= 0.0
    assert after[:, 3].tobytes() == dry[:, 3].tobytes()
    np.testing.assert_allclose(after[:, 1], dry[:, 1], rtol=0.0, atol=1.0e-12)
    assert (after[:, 4] == 0.0).all()


def test_numberless_warm_rain_still_returns_its_water_to_the_vapour(kernels):
    shipped, _ = kernels
    nz = 4
    columns = np.zeros((nz, 14), np.float32)
    p = np.linspace(90000.0, 85000.0, nz).astype(np.float32)
    t = np.full(nz, 290.0, np.float32)
    pii = (p / 1.0e5) ** (287.0 / 1004.5)
    columns[:, 0] = t / pii
    columns[:, 1] = 5.0e-3           # well below water saturation
    columns[:, 4] = 1.0e-11
    columns[:, 7] = 1.0e8
    columns[:, 10] = p / (287.0 * t)
    columns[:, 11] = p
    columns[:, 12] = pii
    columns[:, 13] = 200.0
    after, surface = shipped(columns)
    water = lambda c: np.sum(c[:, 1:7].astype(np.float64), axis=1)  # noqa: E731
    mass = columns[:, 10].astype(np.float64) * columns[:, 13]
    fallen = float(surface[0])       # rainncv, mm = kg m-2
    total_before = float(np.sum(water(columns) * mass))
    total_after = float(np.sum(water(after) * mass)) + fallen
    assert abs(total_after - total_before) <= 8 * np.finfo(np.float32).eps * total_before
    assert (after[:, 4] == 0.0).all()
    assert (after[:, 2] == 0.0).all()
    assert (after[:, 1] >= columns[:, 1]).all()


def test_rain_that_carries_number_takes_wrf_s_arithmetic_bit_for_bit(kernels):
    """The change acts only where the rate is exactly zero.

    Warm columns (no freezing can zero a rain number inside the call) with
    rain number wherever rain mass is present, at both signs of water
    saturation, give identical bytes from the shipped kernel and the twin
    carrying WRF's statement.
    """
    shipped, wrf = kernels
    rng = np.random.default_rng(20260930)
    nz = 24
    for _ in range(40):
        columns = np.zeros((nz, 14), np.float32)
        p = np.sort(rng.uniform(60000.0, 100000.0, nz))[::-1].astype(np.float32)
        t = rng.uniform(275.0, 303.0, nz).astype(np.float32)
        pii = (p / 1.0e5) ** (287.0 / 1004.5)
        es = 610.78 * np.exp(17.27 * (t - 273.16) / (t - 35.86))
        qsat = (0.622 * es / (p - es)).astype(np.float32)
        columns[:, 0] = t / pii
        columns[:, 1] = qsat * rng.uniform(0.5, 1.02, nz)
        columns[:, 2] = rng.uniform(0.0, 5.0e-4, nz) * (rng.uniform(size=nz) < 0.5)
        rain = rng.uniform(1.0e-8, 3.0e-3, nz) * (rng.uniform(size=nz) < 0.7)
        columns[:, 4] = rain
        columns[:, 7] = 1.0e8
        columns[:, 8] = np.where(columns[:, 2] > 0, 1.0e8, 0.0)
        columns[:, 9] = np.where(rain > 0, rng.uniform(1.0e2, 1.0e5, nz), 0.0)
        columns[:, 10] = p / (287.0 * t)
        columns[:, 11] = p
        columns[:, 12] = pii
        columns[:, 13] = rng.uniform(50.0, 600.0, nz)
        a, sa = shipped(columns)
        b, sb = wrf(columns)
        assert a.tobytes() == b.tobytes()
        assert sa.tobytes() == sb.tobytes()


@pytest.mark.gpu
@requires_gpu
def test_the_device_kernel_keeps_the_captured_columns_nonnegative():
    """The NVRTC build the forecast runs, on the same two real columns."""
    import cupy as cp

    from gpuwm.core.wdm6 import launch_wdm6

    data = np.load(_COLUMNS)
    columns = data["columns"]                       # (2, nz, 14)
    nz = columns.shape[1]
    volume = np.ascontiguousarray(
        np.transpose(columns, (2, 1, 0))[:, :, None, :])   # (14, nz, 1, 2)
    fields = [cp.asarray(volume[i]) for i in range(14)]
    theta, qv, qc, qi, qr, qs, qg, nn, nc, nr, rho, p, pii, dz = fields
    surface = {name: cp.zeros((1, 2), cp.float32) for name in (
        "rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv",
        "sr")}
    radii = {name: cp.full((nz, 1, 2), value, cp.float32) for name, value in (
        ("effc", 2.49), ("effi", 4.99), ("effs", 9.99))}
    launch_wdm6(theta, qv, qc, qr, qi, qs, qg, nn, nc, nr, rho, pii, p, dz,
                cp.asarray(data["xland"].reshape(1, 2)),
                surface["rainnc"], surface["rainncv"], surface["snownc"],
                surface["snowncv"], surface["graupelnc"],
                surface["graupelncv"], surface["sr"], float(data["dt"]),
                **radii)
    out = cp.asnumpy(qv)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0
    for column, (k, _, _) in enumerate(data["cells"]):
        assert out[k, 0, column] > 0.0
