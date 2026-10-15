"""Correct mode, water-budget findings WB4 (New Tiedtke) and WB5 (WSM6, WDM6).

Each finding has a test that FAILS in exact mode and PASSES in correct mode,
and a test that exact mode is the arithmetic the engine ran before the
switch existed (gpuwm/correct_mode.py).

CPU tests compile kernels/wsm6.cu through its WSM6_CPU_MIRROR route with g++
and skip when no compiler is present.  The GPU tests need CuPy and a card.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WSM6_CU = ROOT / "gpuwm" / "core" / "kernels" / "wsm6.cu"

#: sha256 of kernels/wsm6.cu on integrate/2.9.0 before correct mode
#: (0bf505764) -- the arithmetic exact mode must keep.
WSM6_287_SHA256 = (
    "c8e0db791d9561372f39ef8ca9810a9ad95b44ef96487fc81dd4a372d76a955a")


# ------------------------------------------- WB5 exact mode: the same bytes --
def _strip_correct_block(text: str) -> str:
    """The source with the correct-mode #ifdef arm removed, #else kept."""
    pattern = re.compile(
        r"#ifdef WSM6_PLM_SURFACE\n.*?#else\n(.*?)#endif\n", re.S)
    out, n = pattern.subn(lambda m: m.group(1), text)
    assert n == 1
    return out


def test_wb5_exact_mode_is_the_published_287_kernel_byte_for_byte():
    """Removing the correct-mode arm gives back the 2.8.7 kernel's bytes, so
    with WSM6_PLM_SURFACE undefined the preprocessor hands NVRTC exactly the
    published translation unit."""
    stripped = _strip_correct_block(WSM6_CU.read_text(encoding="utf-8"))
    assert hashlib.sha256(stripped.encode("utf-8")).hexdigest() == WSM6_287_SHA256


# ------------------------------------------------ WB5 CPU-mirror budget -----
_HARNESS = r"""
#include <cstdio>
#include <cmath>
#define WSM6_CPU_MIRROR
#include WSM6_SOURCE
int main() {
    // Two controlled remap cases: six 100 m layers, 1 m/s, dt 150 s.
    for (int profile = 0; profile < 2; ++profile) {
        const int n = 6;
        float dz[n], den[n], vel[n], qq[n], zi[n+1], za[n+1], dza[n+1];
        float qa[n+1], qa2[n+1], qn[n];
        double before = 0, after = 0;
        for (int k = 0; k < n; ++k) {
            dz[k] = 100; den[k] = 1; vel[k] = 1;
            qq[k] = k < 4 ? .001f * (profile ? 4 - k : k + 1) : 0;
            before += double(qq[k]) * dz[k];
        }
        plm_arrival(qq, nullptr, vel, dz, den, n, 150, zi, za, dza, qa, qa2);
        float precip = plm_regular_remap(qa, zi, za, dza, n, qn);
        for (int k = 0; k < n; ++k) after += double(qn[k]) * dz[k];
        printf("boundary,%d,%.12g\n", profile, after + precip - before);
    }
    // Sixteen full-kernel columns: both hail options, eight time steps.
    for (int hail = 0; hail < 2; ++hail) for (int it = 1; it <= 8; ++it) {
        const int n = 40; float dt = it <= 6 ? 20.f * it : (it == 7 ? 18.75f : 150.f);
        float theta[n], qv[n], qc[n] = {}, qi[n] = {}, qr[n] = {}, qs[n] = {}, qg[n] = {};
        float den[n], p[n], pii[n], dz[n], effc[n], effi[n], effs[n];
        float rain = 0, raincv = 0, snow = 0, snowcv = 0, graup = 0, graupcv = 0, sr = 0, z = 0;
        double before = 0, after = 0;
        for (int k = 0; k < n; ++k) {
            dz[k] = 30.f * powf(1.06f, k); z += .5f * dz[k];
            float t = fmaxf(210, 285 - .007f * z); p[k] = 95000 * expf(-z / 8000);
            qv[k] = .90f * qsat(t, p[k], false); den[k] = p[k] / (287 * t * (1 + .608f * qv[k]));
            pii[k] = powf(p[k] / 100000.f, 287.f / 1004.5f); theta[k] = t / pii[k]; z += .5f * dz[k];
            if (k >= 4 && k < 10) qc[k] = .0002f;
            if (k >= 2 && k < 9) qr[k] = .001f;
            if (k >= 29 && k < 34) qi[k] = .0001f;
            if (k >= 27 && k < 36) qs[k] = .0005f;
            if (k >= 19 && k < 32) qg[k] = .0007f;
            before += double(den[k]) * dz[k] * (double(qv[k]) + qc[k] + qi[k] + qr[k] + qs[k] + qg[k]);
        }
        wsm6_column_impl(theta, qv, qc, qi, qr, qs, qg, den, p, pii, dz, &rain, &raincv,
                         &snow, &snowcv, &graup, &graupcv, &sr, effc, effi, effs, dt, hail, n, 1, 0);
        for (int k = 0; k < n; ++k) {
            after += double(den[k]) * dz[k] * (double(qv[k]) + qc[k] + qi[k] + qr[k] + qs[k] + qg[k]);
            printf("state,%d,%d,%d,%a,%a,%a,%a,%a,%a,%a,%a,%a,%a\n", hail, it, k, theta[k], qv[k],
                   qc[k], qi[k], qr[k], qs[k], qg[k], effc[k], effi[k], effs[k]);
        }
        printf("surface,%d,%d,%a,%a,%a\n", hail, it, rain, snow, graup);
        printf("column,%d,%d,%.12g\n", hail, it, after + rain - before);
    }
    return 0;
}
"""


def _mirror_run(tmp_path: Path, source: Path, *, correct: bool, opt: str) -> list[str]:
    if "WSM6_CPU_MIRROR" not in source.read_text(encoding="utf-8"):
        pytest.skip("this kernels/wsm6.cu has no WSM6_CPU_MIRROR route; the "
                    "card test carries the WB5 budget")
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("g++ is not on PATH; the WSM6 CPU mirror needs it")
    harness = tmp_path / "wsm6_budget.cpp"
    harness.write_text(_HARNESS, encoding="utf-8")
    exe = tmp_path / f"wsm6_budget_{'correct' if correct else 'exact'}_{opt}"
    cmd = [gxx, "-std=c++17", f"-{opt}", "-fno-fast-math", "-ffp-contract=off",
           f'-DWSM6_SOURCE="{source.as_posix()}"', str(harness), "-o", str(exe)]
    if correct:
        cmd.insert(1, "-DWSM6_PLM_SURFACE=1")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    return out.stdout.splitlines()


def _residuals(lines, tag):
    return [float(line.split(",")[-1]) for line in lines if line.startswith(tag + ",")]


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_wb5_budget_fails_in_exact_mode_and_closes_in_correct_mode(tmp_path, opt):
    exact = _mirror_run(tmp_path, WSM6_CU, correct=False, opt=opt)
    fixed = _mirror_run(tmp_path, WSM6_CU, correct=True, opt=opt)
    # Exact: the released flat partial-cell booking misses 0.0125 kg/m2 in
    # both remap cases (either sign) and up to 0.0024 kg/m2 per column call.
    assert min(abs(r) for r in _residuals(exact, "boundary")) > 0.0124
    assert max(abs(r) for r in _residuals(exact, "column")) > 1e-3
    # Correct: closes to the float32 kernel's own rounding.
    assert max(abs(r) for r in _residuals(fixed, "boundary")) < 2e-7
    assert max(abs(r) for r in _residuals(fixed, "column")) < 5e-6
    # Only the surface booking moves: every theta, water species and radius
    # of all 16 columns (6,400 values) is the same float in both modes.
    sx = [line for line in exact if line.startswith("state,")]
    sf = [line for line in fixed if line.startswith("state,")]
    assert len(sx) == 640 and sx == sf


def test_wb5_exact_mirror_matches_the_published_kernel(tmp_path):
    """Exact-mode bytes of every output (atmosphere AND surface) equal those
    of the published 2.8.7 kernel source compiled the same way."""
    published = tmp_path / "wsm6_287.cu"
    published.write_text(
        _strip_correct_block(WSM6_CU.read_text(encoding="utf-8")),
        encoding="utf-8", newline="\n")
    assert hashlib.sha256(published.read_bytes()).hexdigest() == WSM6_287_SHA256
    for opt in ("O0", "O2"):
        assert (_mirror_run(tmp_path, WSM6_CU, correct=False, opt=opt)
                == _mirror_run(tmp_path, published, correct=False, opt=opt))


# ------------------------------------------------------- WB5 on the card ----
def test_wb5_exact_mode_compiles_the_same_ptx_as_2_8_7():
    """NVRTC sees the exact-mode unit with the macro undefined; its PTX is
    the published kernel's PTX."""
    cp = pytest.importorskip("cupy")
    from cupy.cuda import compiler
    from gpuwm.core.kernels import module_options, module_source
    try:
        new = module_source("wsm6")
    except Exception as error:  # pragma: no cover - environment
        pytest.skip(f"kernel source assembly unavailable: {error}")
    old = new.replace(WSM6_CU.read_text(encoding="utf-8"),
                      _strip_correct_block(WSM6_CU.read_text(encoding="utf-8")))
    assert old != new

    def image(src):
        """The image NVRTC compiles from ``src`` (CuPy 14 returns a
        (binary, mapping) pair; older CuPy returns PTX text)."""
        out = compiler.compile_using_nvrtc(src, options=module_options("wsm6"))
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, str):
            out = "\n".join(line for line in out.splitlines()
                            if not line.startswith("//")).encode()
        return bytes(out)

    assert image(new) == image(old)
    del cp


def _wsm6_columns(cp, ncol=64, nz=40, seed=7):
    rng = np.random.default_rng(seed)
    z = np.zeros((nz, ncol), np.float32)
    dz = np.empty((nz, ncol), np.float32)
    for k in range(nz):
        dz[k] = 30.0 * 1.06 ** k * (1.0 + 0.1 * rng.random(ncol))
        z[k] = (dz[:k].sum(axis=0) + 0.5 * dz[k])
    t = np.maximum(210.0, 288.0 - 0.0065 * z).astype(np.float32)
    p = (96000.0 * np.exp(-z / 8000.0)).astype(np.float32)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qv = (0.9 * 0.622 * es / (p - es)).astype(np.float32)
    rho = (p / (287.0 * t * (1 + 0.608 * qv))).astype(np.float32)
    pii = ((p / 1.0e5) ** (287.0 / 1004.5)).astype(np.float32)
    zero = np.zeros_like(t)
    q = {name: zero.copy() for name in ("qc", "qr", "qi", "qs", "qg")}
    q["qr"][2:9] = 0.001 * rng.random((7, ncol))
    q["qc"][4:10] = 0.0002 * rng.random((6, ncol))
    q["qs"][27:36] = 0.0005 * rng.random((9, ncol))
    q["qg"][19:32] = 0.0007 * rng.random((13, ncol))
    q["qi"][29:34] = 0.0001 * rng.random((5, ncol))
    host = dict(theta=t / pii, qv=qv, rho=rho, pii=pii, pressure=p, dz=dz, **q)
    shape = (nz, 1, ncol)
    dev = {k: cp.asarray(np.ascontiguousarray(v.reshape(shape))) for k, v in host.items()}
    for name in ("effc", "effi", "effs"):
        dev[name] = cp.zeros(shape, cp.float32)
    for name in ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr"):
        dev[name] = cp.zeros((1, ncol), cp.float32)
    return dev


def _run_wsm6(cp, conservative):
    from gpuwm.core.wsm6 import launch_wsm6
    d = _wsm6_columns(cp)
    water0 = sum((d["rho"] * d["dz"] * d[n]).astype(cp.float64).sum(axis=0)
                 for n in ("qv", "qc", "qr", "qi", "qs", "qg"))
    launch_wsm6(d["theta"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"], d["qg"],
                d["rho"], d["pii"], d["pressure"], d["dz"], d["rainnc"], d["rainncv"],
                d["snownc"], d["snowncv"], d["graupelnc"], d["graupelncv"], d["sr"], 60.0,
                effc=d["effc"], effi=d["effi"], effs=d["effs"], hail_opt=0,
                conservative_surface=conservative)
    water1 = sum((d["rho"] * d["dz"] * d[n]).astype(cp.float64).sum(axis=0)
                 for n in ("qv", "qc", "qr", "qi", "qs", "qg"))
    residual = cp.asnumpy(water1 + d["rainncv"].astype(cp.float64) - water0).ravel()
    return d, residual


def test_wb5_on_the_card_budget_and_atmosphere():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - environment
        pytest.skip("no CUDA device")
    d_exact, r_exact = _run_wsm6(cp, False)
    d_fixed, r_fixed = _run_wsm6(cp, True)
    assert np.abs(r_exact).max() > 1e-4
    assert np.abs(r_fixed).max() < 5e-6
    for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "effc", "effi", "effs"):
        np.testing.assert_array_equal(cp.asnumpy(d_exact[name]).view(np.uint32),
                                      cp.asnumpy(d_fixed[name]).view(np.uint32),
                                      err_msg=name)


# ------------------------------------------------------- WB4 on the card ----
def _nt_budget(dry_mass):
    """Per-column dry-mass water budget residual of one New Tiedtke call on
    the WRF-oracle fixture (108 columns), as a fraction of precipitation."""
    cp = pytest.importorskip("cupy")
    import test_ntiedtke_pipeline_boundaries as fx
    from gpuwm.core.ntiedtke import NtPipeline
    from gpuwm.verify.ntiedtke_oracle import NT_DT, NT_ITIMESTEP, NT_NZ, NT_STEPCU

    keys = fx._KEYS
    p = NtPipeline(ncol=len(keys), nz=NT_NZ, dt=float(NT_DT),
                   stepcu=NT_STEPCU, itimestep=NT_ITIMESTEP)
    p.dry_mass_coupling = dry_mass

    def seed(name, rows):
        a = np.zeros((rows, len(keys)), dtype=np.float32)
        for c, key in enumerate(keys):
            a[:, c] = fx._INP[key][name][:rows]
        p.w.bind(name, 0)[:rows, :] = cp.asarray(a)

    for f in fx._DRIVER_LEV.values():
        seed(f, NT_NZ)
    for f in fx._DRIVER_IFACE:
        seed(f, NT_NZ + 1)
    for f in ("xland", "hfx", "qfx", "dx"):
        p.w.bind(f, 1)[...] = cp.asarray(
            np.array([float(fx._SUR[k][f]) for k in keys], dtype=np.float32))
    p.run_chunk()
    nz = NT_NZ
    get = lambda name, rows=nz: p.w.bind(name, 0)[:rows].get().astype(np.float64)
    qv = get("qv3d")
    p8w = get("p8w", nz + 1)
    m_dry = (p8w[:-1] - p8w[1:]) / 9.81 / (1.0 + qv)     # MPAS/hex: dp/g = m_d (1+qv)
    water_rate = (m_dry * (get("rqvcuten") + get("rqccuten") + get("rqicuten"))).sum(axis=0)
    delt = float(NT_DT) * NT_STEPCU
    precip_rate = p.w.bind("raincv", 1).get().astype(np.float64)[:len(keys)] / delt
    outputs = {n: p.w.bind(n, 0)[:nz].get() for n in
               ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")}
    outputs["raincv"] = p.w.bind("raincv", 1).get()[:len(keys)]
    p.w.release()
    raining = precip_rate > 1e-7
    return (water_rate + precip_rate)[raining] / precip_rate[raining], outputs


def test_wb4_dry_mass_budget_fails_in_exact_mode_and_closes_in_correct_mode():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - environment
        pytest.skip("no CUDA device")
    rel_exact, out_exact = _nt_budget(False)
    rel_fixed, out_fixed = _nt_budget(True)
    assert rel_exact.size >= 10
    # Exact: about 1% of the convective precipitation is never removed from
    # (or is removed twice from) the dry-mass column water.
    assert np.median(np.abs(rel_exact)) > 3e-3
    # Correct: closes to float32 rounding of the scheme's own budget.
    assert np.median(np.abs(rel_fixed)) < 2e-4
    assert np.median(np.abs(rel_fixed)) < np.median(np.abs(rel_exact)) / 20
    # The forcing conversion moves the closure, so heating, momentum and
    # precipitation may differ; they must stay finite and of the same shape.
    for name in ("rthcuten", "rucuten", "rvcuten", "raincv"):
        assert out_exact[name].shape == out_fixed[name].shape
        assert np.all(np.isfinite(out_fixed[name]))


# ------------------------------------------------- WB5 in WDM6 (lane/290-A4) --
WDM6_CU = ROOT / "gpuwm" / "core" / "kernels" / "wdm6.cu"

#: sha256 of kernels/wdm6.cu on integrate/2.9.0 before correct mode
#: (0bf505764) -- the arithmetic exact mode must keep.
WDM6_290_SHA256 = (
    "add2512fb3ad047bc0fff4a89e85e9a22f278266ac7f4aaa4ccc58f9f853e6d7")


def _strip_wdm6_correct_block(text: str) -> str:
    pattern = re.compile(
        r"#ifdef WDM6_PLM_SURFACE\n.*?#else\n(.*?)#endif\n", re.S)
    out, n = pattern.subn(lambda m: m.group(1), text)
    assert n == 1
    return out


def test_wb5_wdm6_exact_mode_is_the_2_9_0_kernel_byte_for_byte():
    """With WDM6_PLM_SURFACE undefined the preprocessor hands NVRTC exactly
    the 2.9.0 translation unit (the WRF v4.6.1 word-for-word WDM6)."""
    stripped = _strip_wdm6_correct_block(WDM6_CU.read_text(encoding="utf-8"))
    assert hashlib.sha256(stripped.encode("utf-8")).hexdigest() == WDM6_290_SHA256


def test_wb5_wdm6_exact_mode_compiles_the_same_image():
    cp = pytest.importorskip("cupy")
    from cupy.cuda import compiler
    from gpuwm.core.kernels import module_options, module_source
    try:
        new = module_source("wdm6")
    except Exception as error:  # pragma: no cover - environment
        pytest.skip(f"kernel source assembly unavailable: {error}")
    text = WDM6_CU.read_text(encoding="utf-8")
    old = new.replace(text, _strip_wdm6_correct_block(text))
    assert old != new

    def image(src):
        out = compiler.compile_using_nvrtc(src, options=module_options("wdm6"))
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, str):
            out = "\n".join(line for line in out.splitlines()
                            if not line.startswith("//")).encode()
        return bytes(out)

    assert image(new) == image(old)
    del cp


def _wdm6_columns(cp):
    """WSM6's columns with snow and graupel in the lowest twelve levels.

    WDM6 sediments rain in sub-steps that never carry a cell past the lowest
    layer (``wdm6_rain_steps``), so its rain straddles the ground only in
    cell 0, whose PLM profile is flat (qmi = qpi = qa) and books the same
    either way.  Snow and graupel fall in one step across several of these
    30 m layers, so a cell above the lowest one straddles the ground with a
    sloped profile: that is where WB5 bites WDM6 (measured on an RTX 4090:
    exact 0.0133 kg m-2, correct 2.5e-7)."""
    d = _wsm6_columns(cp)
    rng = np.random.default_rng(11)
    nz, _, ncol = d["qg"].shape
    low_g = np.zeros((nz, 1, ncol), np.float32)
    low_s = np.zeros((nz, 1, ncol), np.float32)
    low_g[0:12] = 0.002 * rng.random((12, 1, ncol))
    low_s[0:12] = 0.001 * rng.random((12, 1, ncol))
    d["qg"][...] = cp.asarray(low_g)
    d["qs"][...] = cp.asarray(low_s)
    return d


def _run_wdm6(cp, conservative):
    from gpuwm.core.wdm6 import launch_wdm6
    d = _wdm6_columns(cp)
    shape = d["qc"].shape
    d["nn"] = cp.full(shape, 1.0e8, cp.float32)
    d["nc"] = cp.where(d["qc"] > 0, cp.float32(1.0e8), cp.float32(0.0)).astype(cp.float32)
    d["nr"] = cp.where(d["qr"] > 0, cp.float32(5.0e3), cp.float32(0.0)).astype(cp.float32)
    d["xland"] = cp.ones(shape[1:], cp.float32)
    species = ("qv", "qc", "qr", "qi", "qs", "qg")
    water0 = sum((d["rho"] * d["dz"] * d[n]).astype(cp.float64).sum(axis=0)
                 for n in species)
    launch_wdm6(d["theta"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"], d["qg"],
                d["nn"], d["nc"], d["nr"], d["rho"], d["pii"], d["pressure"], d["dz"],
                d["xland"], d["rainnc"], d["rainncv"], d["snownc"], d["snowncv"],
                d["graupelnc"], d["graupelncv"], d["sr"], 60.0,
                effc=d["effc"], effi=d["effi"], effs=d["effs"], hail_opt=0,
                conservative_surface=conservative)
    water1 = sum((d["rho"] * d["dz"] * d[n]).astype(cp.float64).sum(axis=0)
                 for n in species)
    residual = cp.asnumpy(water1 + d["rainncv"].astype(cp.float64) - water0).ravel()
    return d, residual


def test_wb5_wdm6_on_the_card_budget_and_atmosphere():
    """WDM6 books the straddling cell's below-ground part at the flat cell
    mean in exact mode (column + surface water does not close), and with
    the remap's own profile in correct mode (it closes); the atmosphere is
    bit-identical between the two, only the booked surface amount moves."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - environment
        pytest.skip("no CUDA device")
    d_exact, r_exact = _run_wdm6(cp, False)
    d_fixed, r_fixed = _run_wdm6(cp, True)
    assert np.abs(r_exact).max() > 1e-4
    assert np.abs(r_fixed).max() < 5e-6
    assert np.abs(r_fixed).max() < np.abs(r_exact).max() / 20
    for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr",
                 "effc", "effi", "effs"):
        np.testing.assert_array_equal(cp.asnumpy(d_exact[name]).view(np.uint32),
                                      cp.asnumpy(d_fixed[name]).view(np.uint32),
                                      err_msg=name)


# ------------------------------------------- WB1 in Thompson (lane/290-A4) --
def _thompson_call(cp, booking):
    import dataclasses

    import test_thompson_adapter_composition as composition
    from gpuwm.core import microphysics

    state, cfg = composition._device_adapter_state()
    cfg = dataclasses.replace(cfg, thompson_cloud_surface_booking=booking)
    diagnostics = microphysics._apply_thompson(state, cfg, float(cfg.dt))
    cp.cuda.runtime.deviceSynchronize()
    return state, diagnostics


def test_wb1_thompson_books_settled_cloud_and_changes_nothing_else():
    """Correct mode books the cloud water the fallout takes out of the
    lowest level in CLOUDNC; exact books nothing, and every prognostic word,
    rain total and radius is bit-identical between the two (the class B half
    of WB1: settled cloud is not rain)."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - environment
        pytest.skip("no CUDA device")
    exact, d_exact = _thompson_call(cp, False)
    fixed, d_fixed = _thompson_call(cp, True)
    assert exact.existing_scratch("mp_cloudnc") is None
    cloudnc = cp.asnumpy(fixed.existing_scratch("mp_cloudnc"))
    cloudncv = cp.asnumpy(fixed.existing_scratch("mp_cloudncv"))
    np.testing.assert_array_equal(cloudnc, cloudncv)
    assert np.all(np.isfinite(cloudnc))
    assert cloudnc.min() >= -1e-9
    assert cloudnc.max() > 0.0, "the fixture's low cloud settles nothing"
    # One call settles a small fraction of a millimetre, never more.
    assert cloudnc.max() < 0.1
    for name in ("thp", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr",
                 "effc", "effi", "effs", "h_diabatic"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(exact, name)).view(np.uint32),
            cp.asnumpy(getattr(fixed, name)).view(np.uint32), err_msg=name)
    for name in ("rainnc", "rainncv", "snownc", "graupelnc", "sr"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(d_exact, name)).view(np.uint32),
            cp.asnumpy(getattr(d_fixed, name)).view(np.uint32), err_msg=name)


def test_wb1_booking_equals_the_column_water_the_fallout_removed():
    """The booking is the column cloud water on the fallout kernel's own
    density, before minus after the launch (gpuwm/core/thompson_cloud_booking
    .py): checked against an independent float64 NumPy evaluation."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - environment
        pytest.skip("no CUDA device")
    from gpuwm.core.thompson_cloud_booking import cloud_column_water
    rng = np.random.default_rng(3)
    nz, ny, nx = 12, 3, 5
    t = (280.0 + rng.random((nz, ny, nx))).astype(np.float32)
    p = np.linspace(95000, 60000, nz, dtype=np.float32)[:, None, None] * np.ones((1, ny, nx), np.float32)
    qv = (0.008 * rng.random((nz, ny, nx))).astype(np.float32)
    qc = (0.0005 * rng.random((nz, ny, nx))).astype(np.float32)
    dz = (50.0 + 10.0 * rng.random((nz, ny, nx))).astype(np.float32)
    got = cp.asnumpy(cloud_column_water(*(cp.asarray(a) for a in (qc, t, p, qv, dz))))
    rho = 0.622 * p.astype(np.float64) / (287.04 * t * (np.maximum(qv, 1e-10) + 0.622))
    want = (qc * rho * dz).sum(axis=0)
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_wb1_pair_exists_from_initialization_so_streaming_carries_it():
    """The physics driver allocates the settled-cloud pair with RAINNC, at
    initialization, exactly when the run books it (Thompson, key on), so the
    carrier inventory a tile-streamed run builds its store from already holds
    it.  Breakage prevented: the pair appeared on the first tile step and the
    scatter back refused ("inventories disagree; only in source
    ['scratch/mp_cloudnc', 'scratch/mp_cloudncv']",
    tests/test_cam_ozone.py's nested legacy consumer)."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - environment
        pytest.skip("no CUDA device")
    from tilestream import harness
    from tilestream.physics_inventory import carrier_manifest

    cases = ((8, True, True), (8, False, False), (6, True, False))
    for mp_physics, booking, expected in cases:
        cfg = harness.make_config(
            24, 20, 12, dt=3.0, dx=3000.0, dy=3000.0, ztop=16000.0,
            moist=True, mp_physics=mp_physics,
            thompson_cloud_surface_booking=booking)
        state, _ = harness.make_physics_state(cfg, 4242)
        keys = set(carrier_manifest(state))
        for slot in ("mp_cloudnc", "mp_cloudncv"):
            present = state.existing_scratch(slot) is not None
            assert present is expected, (mp_physics, booking, slot)
            assert (f"scratch/{slot}" in keys) is expected, (mp_physics, booking, slot)
            if expected:
                assert state.existing_scratch(slot).shape == state.mup.shape
                assert float(cp.abs(state.existing_scratch(slot)).max()) == 0.0


def test_wb1_resume_keeps_drops_or_zero_fills_the_total_by_the_live_booking(tmp_path, capsys):
    """thompson_cloud_surface_booking is output-only, so a resume may change
    it.  Booking on both sides: the total is restored.  Booking turned off:
    the stored total is dropped with a note (it would otherwise sit in every
    frame, never updated).  Booking turned on over a checkpoint without the
    pair: it starts from zero with a note (CHANGELOG: counts from the
    resume)."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - environment
        pytest.skip("no CUDA device")
    import dataclasses

    from gpuwm.io import restart
    from tilestream import harness

    on = harness.make_config(
        24, 20, 12, dt=3.0, dx=3000.0, dy=3000.0, ztop=16000.0,
        moist=True, mp_physics=8, thompson_cloud_surface_booking=True)
    off = dataclasses.replace(on, thompson_cloud_surface_booking=False)
    written, _ = harness.make_physics_state(on, 4242)
    written.existing_scratch("mp_cloudnc")[...] = cp.float32(0.25)
    booked = tmp_path / "booked.npz"
    restart.write_restart(booked, written, on)

    kept, _ = harness.make_physics_state(on, 4242)
    restart.restore_restart(booked, kept, on)
    assert float(kept.existing_scratch("mp_cloudnc").min()) == 0.25

    capsys.readouterr()
    dropped, _ = harness.make_physics_state(off, 4242)
    restart.restore_restart(booked, dropped, off)
    assert dropped.existing_scratch("mp_cloudnc") is None
    assert dropped.existing_scratch("mp_cloudncv") is None
    assert "does not book settled cloud" in capsys.readouterr().out

    plain_state, _ = harness.make_physics_state(off, 4242)
    plain = tmp_path / "plain.npz"
    restart.write_restart(plain, plain_state, off)
    started, _ = harness.make_physics_state(on, 4242)
    started.existing_scratch("mp_cloudnc")[...] = cp.float32(9.0)
    capsys.readouterr()
    restart.restore_restart(plain, started, on)
    assert float(cp.abs(started.existing_scratch("mp_cloudnc")).max()) == 0.0
    assert "predates the serialized accumulator 'mp_cloudnc'" in capsys.readouterr().out
