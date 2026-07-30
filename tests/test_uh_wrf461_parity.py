"""UP_HELI_MAX transcription gate against the WRF v4.6.1 oracle.

The oracle (tools/uh_wrf461_oracle) runs the UNMODIFIED cal_helicity +
compute_diff_metrics extracted verbatim from the pinned tree
d66e442fccc04111067e29274c9f9eaccc3cef28 at gfortran -O0, over a 3-step
fixture with terrain, map factors, a rotating updraft, a downdraft column
and a mid-band sign-flip column.  Every input is echoed with its exact
FP32 bit pattern; this gate rebuilds gpuwm-shaped arrays from those bits
and holds BOTH implementations (NumPy mirror; CUDA kernel when a GPU is
present) at the pinned ULP baseline -- equality, not an upper bound.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.uh_diag import mirror_up_heli_max_step_np

ORACLE_DIR = Path(__file__).resolve().parents[1] / "gpuwm" / "data" / "uh" / "oracle"

#: Measured transcription distance, pinned exactly (a later change that
#: moves ANY output must show up here).
BASELINE_MAX_ULP = {
    "uh": 0,
    "up_heli_max": 0,
}

# Fixture geometry (tools/uh_wrf461_oracle/run_cal_helicity.F90).
NX, NY, NZ, NSTEPS = 14, 12, 30, 3


def _rows(name):
    with open(ORACLE_DIR / name, newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _bits_to_f32(bits):
    return np.array(bits, dtype=np.int32).view(np.float32)


@pytest.fixture(scope="module")
def oracle():
    inputs: dict[tuple[str, int], dict[tuple[int, int, int], int]] = {}
    for row in _rows("uh-inputs.csv"):
        key = (row["field"], int(row["step"]))
        inputs.setdefault(key, {})[
            (int(row["k"]), int(row["j"]), int(row["i"]))] = int(row["bits"])
    outputs: dict[tuple[str, int], dict[tuple[int, int], int]] = {}
    for row in _rows("uh-outputs.csv"):
        key = (row["field"], int(row["step"]))
        outputs.setdefault(key, {})[
            (int(row["j"]), int(row["i"]))] = int(row["bits"])

    def grid3(field, step, nk, nj, ni, koff=1):
        table = inputs[(field, step)]
        out = np.empty((nk, nj, ni), dtype=np.float32)
        for (k, j, i), bits in table.items():
            out[k - koff, j - 1, i - 1] = _bits_to_f32([bits])[0]
        assert len(table) == nk * nj * ni, (field, step)
        return out

    def grid2(field, step, nj, ni):
        table = inputs[(field, step)]
        out = np.empty((nj, ni), dtype=np.float32)
        for (_k, j, i), bits in table.items():
            out[j - 1, i - 1] = _bits_to_f32([bits])[0]
        assert len(table) == nj * ni, (field, step)
        return out

    def column(field, nk):
        table = inputs[(field, 0)]
        out = np.zeros((nk,), dtype=np.float32)
        for (k, _j, _i), bits in table.items():
            out[k - 1] = _bits_to_f32([bits])[0]
        return out

    def scalar(field):
        (bits,) = inputs[(field, 0)].values()
        return _bits_to_f32([bits])[0]

    def out2(field, step):
        table = outputs[(field, step)]
        out = np.empty((NY, NX), dtype=np.float32)
        for (j, i), bits in table.items():
            out[j - 1, i - 1] = _bits_to_f32([bits])[0]
        assert len(table) == NY * NX, (field, step)
        return out

    static = {
        "phb": grid3("phb", 0, NZ + 1, NY, NX),
        "ht": grid2("ht", 0, NY, NX),
        "msfu": grid2("msfux", 0, NY, NX + 1),
        "msfv": grid2("msfvy", 0, NY + 1, NX),
        "dn": column("dn", NZ),
        "dnw": column("dnw", NZ),
        "fnm": column("fnm", NZ),
        "fnp": column("fnp", NZ),
        "cf1": scalar("cf1"), "cf2": scalar("cf2"), "cf3": scalar("cf3"),
        "rdx": scalar("rdx"), "rdy": scalar("rdy"),
    }
    steps = []
    for s in range(1, NSTEPS + 1):
        steps.append({
            "u": grid3("u", s, NZ, NY, NX + 1),
            "v": grid3("v", s, NZ, NY + 1, NX),
            "w": grid3("w", s, NZ + 1, NY, NX),
            "ph": grid3("ph", s, NZ + 1, NY, NX),
            "uh": out2("uh", s),
            "up_heli_max": out2("up_heli_max", s),
        })
    return static, steps


def test_oracle_fixture_hashes_match_the_build_receipt():
    recorded = {}
    for line in (ORACLE_DIR / "oracle-sha256sums.txt").read_text(
            encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition("  ")
        recorded[Path(name).name] = digest
    for name in ("uh-inputs.csv", "uh-outputs.csv"):
        digest = hashlib.sha256(
            (ORACLE_DIR / name).read_bytes()).hexdigest()
        assert digest == recorded[name], name
    # The build receipt also pins the authority blob and both extractions.
    assert "module_diffusion_em.pinned.F" in recorded
    assert "extract_cal_helicity.f90" in recorded
    assert "extract_compute_diff_metrics.f90" in recorded


def test_oracle_reference_is_pure_arithmetic_and_guard_can_fire():
    report = (ORACLE_DIR / "libmvec-report.txt").read_text(encoding="utf-8")
    reference = report.split("# -O2")[0]
    for symbol in ("expf", "powf", "logf", "sqrtf", "cbrtf", "_ZGV"):
        assert symbol not in reference.split("REFERENCE")[-1], symbol
    assert "_ZGVbN4v_expf" in report.split("positive control")[-1]


def test_fixture_exercises_the_branches(oracle):
    """The fixture must earn its coverage claims, not assume them."""
    _static, steps = oracle
    # Running max: step 2 is the strongest, step 3 must hold step 2's max.
    assert (steps[1]["up_heli_max"] >= steps[0]["up_heli_max"]).all()
    np.testing.assert_array_equal(
        steps[2]["up_heli_max"], steps[1]["up_heli_max"])
    assert float(steps[1]["up_heli_max"].max()) > 1000.0
    # use_column suppression: columns with nonzero |uh| whose max never grew.
    grew = steps[1]["up_heli_max"] > 0
    active = steps[1]["uh"] != 0
    assert (active & ~grew).sum() > 10
    # Anticyclonic lobes exist (negative uh reaches the smoother).
    assert float(steps[1]["uh"].min()) < -10.0
    # WRF edge behaviour: east column / north row exactly zero; west column
    # and south row mirror their inward neighbours.
    for s in range(NSTEPS):
        field = steps[s]["up_heli_max"]
        assert (field[:, NX - 1] == 0).all()
        assert (field[NY - 1, :] == 0).all()
        np.testing.assert_array_equal(field[:, 0], field[:, 1])
        np.testing.assert_array_equal(field[0, :], field[1, :])


def _run_mirror(static, steps):
    up_heli_max = np.zeros((NY, NX), dtype=np.float32)
    got = []
    for step in steps:
        uh, _use = mirror_up_heli_max_step_np(
            step["u"], step["v"], step["w"], step["ph"], static["phb"],
            static["msfu"], static["msfv"], static["ht"],
            static["dn"], static["dnw"], static["fnm"], static["fnp"],
            static["cf1"], static["cf2"], static["cf3"],
            static["rdx"], static["rdy"], up_heli_max)
        got.append({"uh": uh.copy(), "up_heli_max": up_heli_max.copy()})
    return got


def test_numpy_mirror_matches_the_oracle_at_the_pinned_ulp(oracle):
    static, steps = oracle
    got = _run_mirror(static, steps)
    worst = {"uh": 0, "up_heli_max": 0}
    for s, step in enumerate(steps):
        for field in ("uh", "up_heli_max"):
            distance = fp32_ulp_distance(got[s][field], step[field])
            worst[field] = max(worst[field], int(distance.max()))
    assert worst == BASELINE_MAX_ULP


def test_cuda_kernel_matches_the_oracle_at_the_pinned_ulp(oracle):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from gpuwm.core.uh_diag import device_uh_step

    static, steps = oracle
    up_heli_max = cp.zeros((NY, NX), dtype=cp.float32)
    uh = cp.zeros((NY, NX), dtype=cp.float32)
    use = cp.zeros((NY, NX), dtype=cp.float32)
    worst = {"uh": 0, "up_heli_max": 0}
    for s, step in enumerate(steps):
        device_uh_step(
            cp.asarray(step["u"]), cp.asarray(step["v"]),
            cp.asarray(step["w"]), cp.asarray(step["ph"]),
            cp.asarray(static["phb"]),
            cp.asarray(static["msfu"]), cp.asarray(static["msfv"]),
            cp.asarray(static["ht"]),
            cp.asarray(static["dn"]), cp.asarray(static["dnw"]),
            cp.asarray(static["fnm"]), cp.asarray(static["fnp"]),
            static["cf1"], static["cf2"], static["cf3"],
            static["rdx"], static["rdy"], uh, use, up_heli_max)
        for field, device in (("uh", uh), ("up_heli_max", up_heli_max)):
            distance = fp32_ulp_distance(device.get(), steps[s][field])
            worst[field] = max(worst[field], int(distance.max()))
    assert worst == BASELINE_MAX_ULP


def test_cuda_negative_control_actually_fires():
    """The ULP comparison must be able to see a real difference: perturb one
    w value by one ULP and require a nonzero distance somewhere."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from gpuwm.core.uh_diag import device_uh_step

    rng = np.random.default_rng(7)
    # Small synthetic column set with an updraft inside 2-5 km.
    nz, ny, nx = 12, 6, 7
    znw = ((nz - np.arange(nz + 1)) / nz).astype(np.float32)
    z = (1.0 - znw) * np.float32(9000.0)
    phb = (z * np.float32(9.81)).astype(np.float32)
    u = rng.normal(5, 1, (nz, ny, nx + 1)).astype(np.float32)
    v = rng.normal(3, 1, (nz, ny + 1, nx)).astype(np.float32)
    w = np.full((nz + 1, ny, nx), 4.0, dtype=np.float32)
    ph = np.zeros((nz + 1, ny, nx), dtype=np.float32)
    msfu = np.ones((ny, nx + 1), dtype=np.float32)
    msfv = np.ones((ny + 1, nx), dtype=np.float32)
    ht = np.zeros((ny, nx), dtype=np.float32)
    dnw = np.diff(znw).astype(np.float32)
    dn = np.zeros(nz, dtype=np.float32)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    fnp = np.zeros(nz, dtype=np.float32)
    fnm = np.zeros(nz, dtype=np.float32)
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    cof1 = (2 * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
    cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
    args = (msfu, msfv, ht, dn, dnw, fnm, fnp,
            np.float32(fnp[1] + cof1), np.float32(fnm[1] - cof1 - cof2),
            np.float32(cof2), np.float32(1e-3), np.float32(1e-3))

    def run(w_host):
        maxf = cp.zeros((ny, nx), dtype=cp.float32)
        uh = cp.zeros((ny, nx), dtype=cp.float32)
        use = cp.zeros((ny, nx), dtype=cp.float32)
        device_uh_step(cp.asarray(u), cp.asarray(v), cp.asarray(w_host),
                       cp.asarray(ph), cp.asarray(phb),
                       *[cp.asarray(a) if isinstance(a, np.ndarray) else a
                         for a in args], uh, use, maxf)
        return maxf.get()

    base = run(w)
    w_perturbed = w.copy()
    # A single-ULP nudge of one 4.0 legitimately vanishes inside the
    # 8-point 0.125*(sum to 32.0) average (below half an ULP of 32), so the
    # control perturbs by 2**-10 relative -- small, but guaranteed to move
    # wavg -- and requires the harness to see it.
    w_perturbed[nz // 2, ny // 2, nx // 2] *= np.float32(1.0 + 2.0**-10)
    assert base.max() > 0
    assert fp32_ulp_distance(run(w_perturbed), base).max() > 0
