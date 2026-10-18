"""Grade WOOF's Grell-Freitas kernel against the regime-column WRF oracle.

Runs ``gf_gfdrv_stage`` -- the kernel the engine adapter launches
(gpuwm/core/gf.py) -- on every column build_regime.sh fed to WRF's own
GFDRV, and compares every output word.

The comparison of record is against ``gf-regime-wrf-crgamma``: WRF's
byte-pinned objects with the correctly rounded tgammaf linked in, the gamma
WOOF ships.  The kernel runs its SHIPPED path for gamma (fzu computed, the
three override slots 0), so nothing is pinned from the capture.  Two k22
arms run:

* ``faithful`` (k22_wrf_faithful = 1): WRF's shallow MAXLOC section offset,
  so the kernel is asked to be WRF word for word;
* ``shipped`` (k22_wrf_faithful = 0): the corrected index the engine runs,
  graded against independently compiled corrected WRF, on every column.

Both referees cap the shallow pressure-difference loop at ktf-1. There
is no exemption for columns whose k22 changes.

``gf-regime-wrf-stock`` (WRF as it links, glibc tgammaf) is compared too,
and the difference is reported as the documented gamma divergence.

Strict versus default arithmetic is chosen by the environment BEFORE this
module imports gpuwm (``GPUWM_WRF_EXACT=1`` for strict); the report records
which one ran.

usage:
    python gf_regime_gpu.py ORACLE_DIR OUT.json
ORACLE_DIR holds gf-regime-inputs.bin, gf-regime-wrf-crgamma[-k0].bin and
gf-regime-wrf-stock[-k0].bin (build dir or the committed fixture dir).
"""

from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gf_regime_columns as grc                                  # noqa: E402
from gf_field_lists import (                                     # noqa: E402
    DRV_IN_LEV, DRV_IN_SCA, DRV_ISCA_FIELDS, DRV_LEV_FIELDS, DRV_SCA_FIELDS,
)

WRF_LEV = list(grc.OUT_LEV)
WRF_SCA = list(grc.OUT_SCA)
WRF_INT = list(grc.OUT_INT)


def _first(d: Path, *names: str) -> Path:
    for n in names:
        if (d / n).exists():
            return d / n
    raise FileNotFoundError(f"none of {names} in {d}")


def load_oracle(oracle_dir: Path):
    inp = grc.read_inputs(_first(oracle_dir, "gf-regime-inputs.bin"))
    cr = grc.read_outputs(_first(oracle_dir, "gf-regime-wrf-crgamma-k0.bin",
                                 "gf-regime-wrf-crgamma.bin"))
    st = grc.read_outputs(_first(oracle_dir, "gf-regime-wrf-stock-k0.bin",
                                 "gf-regime-wrf-stock.bin"))
    man = json.loads(_first(oracle_dir, "gf-regime-manifest.json").read_text(
        encoding="utf-8"))
    for o in (cr, st):
        if o["kte_mode"] != 0:
            raise ValueError("grade against kte_mode 0, WRF's own convention")
        if (o["ncol"], o["nz"]) != (inp["ncol"], inp["nz"]):
            raise ValueError("oracle and inputs disagree on shape")
    return inp, cr, st, man


def load_reference(oracle_dir: Path, arm: str, *, arithmetic=None):
    """Load an independent WRF arm, or its hash-bound sparse word receipt.

    The receipt is a compact test fixture, generated from WRF output rather
    than inferred from the GPU or from a changed-k22 column list.
    """
    if arm not in ("safe", "corrected"):
        raise ValueError(arm)
    from gpuwm import wrf_exact
    arithmetic = arithmetic or ("strict" if wrf_exact.ENABLED else "default")
    if arithmetic not in ("strict", "default"):
        raise ValueError(arithmetic)
    reference_arm = "scalar-" + arm if arithmetic == "strict" else arm
    path = oracle_dir / f"gf-regime-wrf-{reference_arm}-k0.bin"
    if path.exists():
        result = grc.read_outputs(path)
    else:
        delta = json.loads((oracle_dir / f"gf-regime-wrf-{reference_arm}-delta.json").read_text())
        base = _first(oracle_dir, "gf-regime-wrf-crgamma-k0.bin",
                      "gf-regime-wrf-crgamma.bin")
        if hashlib.sha256(base.read_bytes()).hexdigest() != delta["base_sha256"]:
            raise ValueError("WRF reference delta has the wrong base hash")
        result = grc.read_outputs(base)
        for key in ("lev", "sca", "isc"):
            words = result[key].view(np.uint32).reshape(-1)
            for index, value in delta["words"][key]:
                words[index] = value
        raw = np.asarray([47012, result["ncol"], result["nz"], result["kte_mode"]],
                         dtype="<i4").tobytes()
        raw += b"".join(result[key].tobytes() for key in ("lev", "sca", "isc"))
        if hashlib.sha256(raw).hexdigest() != delta["output_sha256"]:
            raise ValueError("WRF reference delta does not reconstruct the recorded output hash")
    inp = grc.read_inputs(oracle_dir / "gf-regime-inputs.bin")
    if result["kte_mode"] != 0 or (result["ncol"], result["nz"]) != (inp["ncol"], inp["nz"]):
        raise ValueError("corrected referee and inputs disagree on vertical convention or shape")
    return result


def load_branch_reference():
    """216 pinned-fzu boundary words from the independent corrected WRF run."""
    from gpuwm.verify.gf_oracle import load_gf_oracle, GF_ORACLE_DIR
    fixture = load_gf_oracle()
    delta = json.loads((GF_ORACLE_DIR / "gf-corrected-boundary-delta.json").read_text())
    for name, digest in delta["base_sha256"].items():
        if hashlib.sha256((GF_ORACLE_DIR / name).read_bytes()).hexdigest() != digest:
            raise ValueError("corrected boundary reference has the wrong base hash")
    want = dict(ncol=fixture.ncol, nz=fixture.levels["t"].shape[1],
                lev=np.stack([fixture.levels[f] for f in WRF_LEV], axis=1),
                sca=np.stack([fixture.surface[f] for f in WRF_SCA], axis=1),
                isc=np.stack([fixture.surface[f].astype(np.int32) for f in WRF_INT], axis=1))
    for key in ("lev", "sca", "isc"):
        flat = want[key].view(np.uint32).ravel()
        for index, value in delta["words"][key]:
            flat[index] = value
    return fixture, want


def kernel_inputs(inp):
    n, nz = inp["ncol"], inp["nz"]
    L = grc.LEV_FIELDS
    lvin = np.zeros((n, len(DRV_IN_LEV), nz), dtype=np.float32)
    for j, name in enumerate(DRV_IN_LEV):
        if name == "p8w":
            lvin[:, j, :] = inp["p8w"][:, :nz]
        else:
            lvin[:, j, :] = inp["lev"][:, L.index(name), :]
    scin = np.zeros((n, len(DRV_IN_SCA)), dtype=np.float32)
    S = grc.SCA_FIELDS
    for j, name in enumerate(DRV_IN_SCA):
        if name in S:
            scin[:, j] = inp["sc"][:, S.index(name)]
        # fzu_up/fzu_dn/fzu_sh stay 0: the shipped path COMPUTES fzu
    iin = np.ascontiguousarray(inp["ii"][:, :3].astype(np.int32))
    return (np.ascontiguousarray(lvin), np.ascontiguousarray(scin), iin)


def run_kernel(inp, k22_wrf_faithful: int):
    import cupy as cp

    from gpuwm.core.gf import _gf_module, gf_workspace_floats

    n, nz = inp["ncol"], inp["nz"]
    lvin, scin, iin = kernel_inputs(inp)
    module = _gf_module(nz)
    fn = module.get_function("gf_gfdrv_stage")
    d_lv, d_sc, d_ii = cp.asarray(lvin), cp.asarray(scin), cp.asarray(iin)
    d_lev = cp.zeros((n, len(DRV_LEV_FIELDS), nz), dtype=cp.float32)
    d_sca = cp.zeros((n, len(DRV_SCA_FIELDS)), dtype=cp.float32)
    d_isc = cp.zeros((n, len(DRV_ISCA_FIELDS)), dtype=cp.int32)
    d_ws = cp.empty(gf_workspace_floats(nz, n), dtype=cp.float32)
    fn(((n + 63) // 64,), (64,),
       (d_lv, d_sc, d_ii, d_lev, d_sca, d_isc, d_ws,
        np.int32(k22_wrf_faithful), np.int32(n), np.int32(nz)))
    cp.cuda.Stream.null.synchronize()
    return dict(lev=cp.asnumpy(d_lev), sca=cp.asnumpy(d_sca),
                isc=cp.asnumpy(d_isc))


def _ulp(a, b):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    return fp32_ulp_distance(np.asarray(a, np.float32),
                             np.asarray(b, np.float32))


def compare(got, want):
    """Per-field word comparison of a kernel run against one WRF output."""
    n = want["ncol"]
    out = {}
    col_bad = np.zeros(n, dtype=bool)
    words = 0
    for j, name in enumerate(WRF_LEV):
        g = got["lev"][:, DRV_LEV_FIELDS.index(name), :]
        w = want["lev"][:, j, :]
        neq = g.view(np.uint32) != w.view(np.uint32)
        d = _ulp(g, w)
        finite = d >= 0
        cols = np.flatnonzero(neq.any(axis=1))
        col_bad[cols] = True
        words += g.size
        out[name] = dict(words=int(g.size), words_differ=int(neq.sum()),
                         got_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                         wrf_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                         columns_differ=int(cols.size),
                         max_ulp=int(d[finite].max()) if finite.any() else -1,
                         non_comparable=int((~finite).sum()),
                         max_abs=float(np.max(np.abs(
                             g.astype(np.float64) - w.astype(np.float64)))),
                         first_columns=[int(c) for c in cols[:8]])
    for j, name in enumerate(WRF_SCA):
        g = got["sca"][:, DRV_SCA_FIELDS.index(name)]
        w = want["sca"][:, j]
        neq = g.view(np.uint32) != w.view(np.uint32)
        d = _ulp(g, w)
        finite = d >= 0
        cols = np.flatnonzero(neq)
        col_bad[cols] = True
        words += g.size
        out[name] = dict(words=int(g.size), words_differ=int(neq.sum()),
                         got_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                         wrf_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                         columns_differ=int(cols.size),
                         max_ulp=int(d[finite].max()) if finite.any() else -1,
                         non_comparable=int((~finite).sum()),
                         max_abs=float(np.max(np.abs(
                             g.astype(np.float64) - w.astype(np.float64)))),
                         first_columns=[int(c) for c in cols[:8]])
    for j, name in enumerate(WRF_INT):
        g = got["isc"][:, DRV_ISCA_FIELDS.index(name)].astype(np.int64)
        w = want["isc"][:, j].astype(np.int64)
        cols = np.flatnonzero(g != w)
        col_bad[cols] = True
        words += g.size
        out[name] = dict(words=int(g.size), words_differ=int(cols.size),
                         got_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                         wrf_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                         columns_differ=int(cols.size), max_ulp=None,
                         first_columns=[int(c) for c in cols[:8]])
    total_differ = int(sum(v["words_differ"] for v in out.values()))
    return dict(fields=out, words=int(words), words_differ=total_differ,
                columns_differ=int(col_bad.sum()),
                columns=[int(c) for c in np.flatnonzero(col_bad)])


def coverage(cr, man):
    cols = man["columns"]
    pr = cr["sca"][:, WRF_SCA.index("pratec")]
    xs = cr["sca"][:, WRF_SCA.index("xmb_shallow")]
    ktd = cr["isc"][:, WRF_INT.index("ktop_deep")]
    rqi = cr["lev"][:, WRF_LEV.index("rqicuten")]
    rqc = cr["lev"][:, WRF_LEV.index("rqccuten")]
    gdc2 = cr["lev"][:, WRF_LEV.index("gdc2")]
    return dict(
        columns=len(cols),
        regimes=sorted({c["regime"] for c in cols}),
        precipitating=int((pr > 0).sum()),
        shallow_fired=int((xs > 0).sum()),
        deep_top_reached=int((ktd > 0).sum()),
        ice_tendency_columns=int((np.abs(rqi) > 0).any(axis=1).sum()),
        liquid_tendency_columns=int((np.abs(rqc) > 0).any(axis=1).sum()),
        ice_cloud_columns=int((np.abs(gdc2) > 0).any(axis=1).sum()),
        clos_choice=sorted({c["ichoice"] for c in cols}),
        dx=sorted({c["dx"] for c in cols}),
        ishallow=sorted({c["ishallow"] for c in cols}),
        xland=sorted({c["xland"] for c in cols}),
        ht_range=[min(c["ht"] for c in cols), max(c["ht"] for c in cols)],
        any_nan=bool(np.isnan(cr["lev"]).any() or np.isnan(cr["sca"]).any()),
    )


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    oracle_dir, out = Path(argv[1]), Path(argv[2])
    from gpuwm import wrf_exact

    inp, cr, st, man = load_oracle(oracle_dir)
    safe = load_reference(oracle_dir, "safe")
    corrected = load_reference(oracle_dir, "corrected")
    faithful = run_kernel(inp, 1)
    shipped = run_kernel(inp, 0)
    import cupy as cp
    dev = cp.cuda.Device()
    report = dict(
        arithmetic="strict (GPUWM_WRF_EXACT=1)" if wrf_exact.ENABLED
        else "default",
        strict_options=list(wrf_exact.STRICT_OPTIONS) if wrf_exact.ENABLED
        else [],
        device=cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode(),
        nvrtc=".".join(str(v) for v in cp.cuda.nvrtc.getVersion()),
        cupy=cp.__version__,
        reference_build="official scalar WRF flags" if wrf_exact.ENABLED else "retained default column control",
        nz=inp["nz"],
        coverage=coverage(cr, man),
        faithful_vs_wrf_reference=compare(faithful, safe),
        shipped_vs_wrf_corrected=compare(shipped, corrected),
        shipped_vs_wrf_crgamma=compare(shipped, cr),
        faithful_vs_wrf_stock=compare(faithful, st),
        shipped_vs_faithful_columns=compare(
            shipped, dict(lev=faithful["lev"][:, [DRV_LEV_FIELDS.index(f)
                                                  for f in WRF_LEV], :],
                          sca=faithful["sca"][:, [DRV_SCA_FIELDS.index(f)
                                                  for f in WRF_SCA]],
                          isc=faithful["isc"][:, [DRV_ISCA_FIELDS.index(f)
                                                  for f in WRF_INT]],
                          ncol=inp["ncol"]))["columns"],
        k22_faithful=[int(v) for v in faithful["isc"][:, DRV_ISCA_FIELDS.index(
            "k22_shallow")]],
        k22_shipped=[int(v) for v in shipped["isc"][:, DRV_ISCA_FIELDS.index(
            "k22_shallow")]],
    )
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    fv = report["faithful_vs_wrf_reference"]
    sv = report["shipped_vs_wrf_corrected"]
    gv = report["faithful_vs_wrf_stock"]
    print(f"[{report['arithmetic']}] {report['device']} nvrtc {report['nvrtc']}")
    print(f"faithful k22 vs selected WRF control: {fv['words_differ']} of "
          f"{fv['words']} words differ on {fv['columns_differ']} of "
          f"{inp['ncol']} columns")
    for name, v in fv["fields"].items():
        if v["words_differ"]:
            print(f"   {name}: {v['words_differ']} words, max_ulp "
                  f"{v['max_ulp']}, max_abs {v.get('max_abs')}, cols "
                  f"{v['first_columns']}")
    print(f"shipped k22 vs corrected WRF: {sv['words_differ']} words on "
          f"{sv['columns_differ']} columns {sv['columns'][:12]}")
    print(f"faithful vs WRF (stock glibc gamma): {gv['words_differ']} words "
          f"on {gv['columns_differ']} columns")
    print(json.dumps(report["coverage"]))
    return int(fv["words_differ"] != 0 or sv["words_differ"] != 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
