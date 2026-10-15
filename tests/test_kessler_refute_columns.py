"""Kessler (mp_physics=1) on columns the kessler-mp1 lane did not use.

The fixtures (tests/data/oracles/kessler/refute-*/, PROVENANCE.md beside
them) were written by tools/kessler_wrf461_oracle/refute/: the same
byte-unmodified WRF v4.6.1 routine and run_kessler.F90 driver as
tests/test_kessler_wrf461_parity.py, gfortran 13.3.0 -O0, on new inputs:

* refute-setA, 104 columns x 50 levels: model tops near 1 hPa (es >= p
  aloft), Exner above one, 315-330 K and 185-205 K surfaces, large negative
  undershoots, signed zeros in qv/qc/qr, qc/qr log-uniform over the whole
  float32 range (subnormals included), saturation to 1e-6, 1-5 m surface
  layers (split counts in the thousands), density spikes, RAINNC to 1e5 mm,
  qr/qc on every power of two 2**-149..2**-3 and its neighbours; dt 7.2 s x
  10 and 900 s x 2;
* refute-setB, 32 columns x 256 levels (KESS_KMAX): LES and deep grids;
* refute-setC/D, 32 columns x 2 and x 3 levels (the smallest the routine
  accepts);
* refute-setF, 24 columns x 50 levels: realistic high-top columns
  (stratospheric vapour of 2-6 ppmv) and the C-library-variant witnesses.

What these pin:

1. Strict build (GPUWM_WRF_EXACT=1): bitwise against WRF on every word the
   es cap does not touch, free running and per-call replay.
2. The es cap (kessler.cu, WOOF's own guard): WRF's routine "condenses" water
   it does not have where es >= p and heats without bound (4,435 K after four
   60 s calls in refute-setF).  WOOF caps es at 0.99 p and stays bounded.
3. The C-library variant: on an FMA-capable x86 CPU the box's libm dispatches
   powf/expf to an FMA build whose words differ from the generic build in a
   handful of inputs (powf(x, 0.875) at four x, two of them physical rain
   mixing ratios, 2.2729364e-07 and 8.878658e-10 kg/kg).  WOOF's gfk_pow and
   gfk_exp equal the generic build on every float32 input.  refute-setF holds
   the reference under both dispatches: WOOF is bitwise against the generic
   one, and against the FMA one it differs only on the witness columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from gpuwm.verify import kessler_oracle as ko

ROOT = ko.FIXTURE_DIR
SETS = ("refute-setA", "refute-setB", "refute-setC", "refute-setD",
        "refute-setF")
F32 = np.float32
#: Exclusion margin for the es cap.  The cap engages where es > 0.99 p; cells
#: with es > 0.98 p in WRF's own state are left out of the bitwise
#: comparison, so a last-bit difference between numpy's float32 exp and the
#: device's cannot misclassify a capped cell as an uncapped one.
CAP_MARGIN = F32(0.98)
#: Default arithmetic (CuPy's injected -ftz=true) may differ from WRF only by
#: flushed subnormal intermediates.  The breakage this names: any arithmetic
#: difference other than flushing; the largest FTZ residue measured on these
#: sets is 3.5e-34, far below this bound and far below any physical amount.
DEFAULT_FTZ_BOUND = 1e-30


def _load(name, ref="kessler-wrf461.npz"):
    d = ROOT / name
    with np.load(d / "kessler-inputs.npz") as f:
        inputs = {k: f[k] for k in f.files}
    with np.load(d / ref) as f:
        wrf = {k: f[k] for k in f.files}
    return inputs, wrf


def _call_inputs(inputs, wrf):
    """The state entering every call of the WRF run (n, nz, ncol)."""
    out = {k: [] for k in ko.LEVEL_FIELDS}
    n = 0
    for nsteps in inputs["nsteps"]:
        for step in range(int(nsteps)):
            for k in ko.LEVEL_FIELDS:
                out[k].append(inputs[k] if step == 0 else wrf[k][n - 1])
            n += 1
    return {k: np.stack(v) for k, v in out.items()}


def _capped(inputs, state):
    """Cells where es exceeds CAP_MARGIN * p in the WRF state of each call,
    in the Fortran's float32 order."""
    pii = inputs["pii"][None]
    with np.errstate(all="ignore"):
        temp = (pii * state["t"]).astype(F32)
        p = (F32(1.0e5) * np.power(pii, F32(1004.0) / F32(287.0))).astype(F32)
        es = (F32(1000.0) * F32(0.6112) * np.exp(
            F32(17.67) * (temp - F32(273.15)) / (temp - F32(29.65)))
        ).astype(F32)
    return es > CAP_MARGIN * p


def test_fixtures_are_the_documented_sets():
    shapes = {"refute-setA": (50, 104, 12), "refute-setB": (256, 32, 5),
              "refute-setC": (2, 32, 12), "refute-setD": (3, 32, 12),
              "refute-setF": (50, 24, 8)}
    for name, (nz, ncol, calls) in shapes.items():
        inputs, wrf = _load(name)
        assert inputs["t"].shape == (nz, ncol)
        assert int(inputs["nsteps"].sum()) == calls
        for k in ko.LEVEL_FIELDS:
            assert wrf[k].shape == (calls, nz, ncol)
    inputs, _ = _load("refute-setA")
    q = np.stack([inputs[k] for k in ("qv", "qc", "qr")])
    assert np.any((q != 0) & (np.abs(q) < np.finfo(F32).tiny))
    assert np.any((q == 0) & np.signbit(q))
    assert np.any(inputs["pii"] > 1)
    assert np.any(q < -1e-5)


def test_wrf_reference_runs_away_where_es_exceeds_p():
    """The WRF defect the cap exists for, read from WRF's own outputs."""
    inputs, wrf = _load("refute-setF")
    cols = np.nonzero(inputs["regime"] == "high_top_real")[0]
    assert _capped(inputs, {"t": inputs["t"][None]})[:, :, cols].any()
    last = int(inputs["nsteps"].sum()) - 1          # the 4 x 60 s case
    temp = inputs["pii"][:, cols] * wrf["t"][last][:, cols]
    assert temp.max() > 1000.0
    # one call "condenses" more water than the cell holds
    first = 0
    k, c = np.nonzero(_capped(inputs, {"t": inputs["t"][None]})[0])
    hot = np.isin(c, cols)
    k, c = k[hot], c[hot]
    before = inputs["qv"][k, c] + inputs["qc"][k, c] + inputs["qr"][k, c]
    after = (wrf["qv"][first][k, c] + wrf["qc"][first][k, c]
             + wrf["qr"][first][k, c])
    assert np.any(after > 10 * before)


def test_cpu_mirror_caps_es():
    """npref's float64 mirror carries the same cap and stays bounded."""
    from gpuwm.verify import npref
    import inspect
    assert "es = np.minimum(es, 0.99 * pressure)" in inspect.getsource(
        npref.np_kessler_column)
    inputs, _ = _load("refute-setF")
    col = int(np.nonzero(inputs["regime"] == "high_top_real")[0][0])
    state = [inputs[k][:, col].astype(float) for k in ("t", "qv", "qc", "qr")]
    rainnc = float(inputs["rainnc"][col])
    args = [inputs[k][:, col].astype(float) for k in ("rho", "pii", "z",
                                                      "dz8w")]
    water0 = state[1] + state[2] + state[3]
    for _ in range(4):
        t, qv, qc, qr, rainnc, _ = npref.np_kessler_column(
            *state, *args, 60.0, rainnc)
        state = [t, qv, qc, qr]
    assert np.max(inputs["pii"][:, col] * state[0]) < 400.0
    # the top ten layers only lose water (rain falls out of their base)
    w = (inputs["rho"][:, col] * inputs["dz8w"][:, col]).astype(float)[-10:]
    slab0 = np.sum(w * water0[-10:])
    slab = np.sum(w * (state[1] + state[2] + state[3])[-10:])
    assert slab <= slab0 * (1 + 1e-9)


def test_kernel_carries_the_cap():
    from gpuwm.core import kernels
    src = (Path(kernels.__file__).parent / "kessler.cu").read_text(
        encoding="utf-8")
    code = "\n".join(line.split("//")[0] for line in src.splitlines())
    assert "es = fminf(es, 0.99f * pressure);" in code


def _compare_masked(name, mode, ref="kessler-wrf461.npz"):
    inputs, wrf = _load(name, ref)
    got = ko.run_gpu(inputs, mode=mode, wrf=wrf)
    capped = _capped(inputs, _call_inputs(inputs, wrf))      # (n, nz, ncol)
    if mode == "free":
        # a capped cell changes its column's later calls: drop the column
        # from the whole case it appears in
        mask = np.zeros_like(capped)
        n = 0
        for nsteps in inputs["nsteps"]:
            sl = slice(n, n + int(nsteps))
            bad = capped[sl].any(axis=(0, 1))
            mask[sl] = bad[None, None, :]
            n += int(nsteps)
    else:
        mask = capped
    col_mask = mask.any(axis=1)                              # (n, ncol)
    out = {}
    for k in ko.OUTPUT_FIELDS:
        g, w = got[k], wrf[k]
        keep = ~(mask if g.ndim == 3 else col_mask)
        miss = (g.view(np.uint32) != w.view(np.uint32)) & keep
        diff = np.abs(g.astype(np.float64) - w.astype(np.float64))
        out[k] = (int(miss.sum()), float(diff[miss].max()) if miss.any()
                  else 0.0, int(keep.sum()))
    return out, inputs, got, mask


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", ("free", "replay"))
@pytest.mark.parametrize("name", SETS)
def test_device_is_bitwise_wrf_off_the_cap(name, mode):
    from gpuwm import wrf_exact
    ref = ("kessler-wrf461-generic-libm.npz" if name == "refute-setF"
           else "kessler-wrf461.npz")
    table, _, _, _ = _compare_masked(name, mode, ref)
    assert all(kept > 0 for _, _, kept in table.values())
    if wrf_exact.ENABLED:
        assert {k: v for k, v in table.items() if v[0]} == {}, table
    else:
        assert all(v[1] < DEFAULT_FTZ_BOUND for v in table.values()), table


@pytest.mark.gpu
@requires_gpu
def test_fma_dispatch_reference_differs_only_on_the_witnesses():
    from gpuwm import wrf_exact
    if not wrf_exact.ENABLED:
        pytest.skip("the witness words are graded under the strict build")
    inputs, wrf = _load("refute-setF")
    got = ko.run_gpu(inputs, mode="replay", wrf=wrf)
    regime = inputs["regime"]
    capped = _capped(inputs, _call_inputs(inputs, wrf))
    seen = set()
    for k in ko.LEVEL_FIELDS:
        miss = (got[k].view(np.uint32) != wrf[k].view(np.uint32)) & ~capped
        seen |= set(map(str, regime[np.nonzero(miss)[-1]]))
    assert seen == {"witness_30740e00", "witness_34740e00"}, seen


@pytest.mark.gpu
@requires_gpu
def test_device_stays_bounded_where_wrf_runs_away():
    inputs, wrf = _load("refute-setF")
    got = ko.run_gpu(inputs, mode="free", wrf=wrf)
    cols = np.nonzero(inputs["regime"] == "high_top_real")[0]
    last = int(inputs["nsteps"].sum()) - 1
    temp = inputs["pii"][:, cols] * got["t"][last][:, cols]
    assert temp.max() < 400.0
    # mass of water in the top ten layers never grows (rain only leaves)
    w = (inputs["rho"] * inputs["dz8w"]).astype(np.float64)[-10:, cols]
    slab0 = np.sum(w * (inputs["qv"] + inputs["qc"] + inputs["qr"]
                        )[-10:, cols], axis=0)
    slab = np.sum(w * (got["qv"][last] + got["qc"][last] + got["qr"][last]
                       )[-10:, cols], axis=0)
    assert np.all(slab <= slab0 * (1 + 1e-5)), (slab, slab0)
