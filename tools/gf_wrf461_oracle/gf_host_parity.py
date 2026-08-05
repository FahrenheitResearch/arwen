"""Grade gpuwm/core/kernels/gf_deep.cu -- compiled for the HOST -- against
the committed oracle fixtures, no GPU anywhere.

Why this exists: the CUDA source is the artifact, and x86-64 SSE evaluates
the same __fadd_rn/__fmul_rn/... operations with the same IEEE-754
round-to-nearest semantics when built with contraction off.  Compiling the
kernel as C++ (tools/gf_wrf461_oracle/gf_host_harness.cpp) therefore checks
the whole transcription -- scheme, gamma path, glibc libm layer, constant
table -- before a GPU ever sees it, and separates "the transcription is
wrong" from "the device toolchain did something" when the on-node gate
disagrees.  The remaining host/device difference classes are exactly the
ones tests/test_gf_deep_cuda.py measures on the node: sm_120's FP32
subnormal flush and anything ptxas does to the compiled stream.

Build (glibc 2.39 toolchain, same as the oracle):

    g++ -O2 -std=c++17 -ffp-contract=off -fno-unsafe-math-optimizations \
        -I ../../gpuwm/core/kernels -shared -fPIC gf_host_harness.cpp \
        -o gf_host_harness.so -lm

Run from anywhere:  python3 tools/gf_wrf461_oracle/gf_host_parity.py

Measured 2026-08-04 on WSL Ubuntu-24.04 (gcc 13.3.0, glibc 2.39-0ubuntu8.7,
the oracle's own toolchain): 83/83 level fields, 69/69 float scalars and
39/39 integer fields bitwise over all 216 columns with fzu COMPUTED, plus
0 word mismatches on the four gf-libm sweeps (65638/32768/16385/16385
arguments), the 254 powf answer-sheet rows, the 100 pgamma fzu rows and
the 123-word constant table.
"""

import csv
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from gpuwm.core.fp32_ulp import fp32_ulp_distance                # noqa: E402
from gpuwm.verify.gf_oracle import (                             # noqa: E402
    GF_NZ, GF_ORACLE_DIR, load_gf_oracle,
)

from gf_field_lists import (                                     # noqa: E402
    DRV_IN_LEV, DRV_IN_SCA, DRV_ISCA_FIELDS, DRV_LEV_FIELDS, DRV_SCA_FIELDS,
    IN_LEV, IN_SCA, ISCA_FIELDS, LEV_FIELDS, SCA_FIELDS,
    SH_IN_LEV, SH_IN_SCA, SH_ISCA_FIELDS, SH_LEV_FIELDS, SH_SCA_FIELDS,
)

NZ = GF_NZ


def _lib():
    path = os.environ.get(
        "GF_HOST_HARNESS", os.path.join(HERE, "gf_host_harness.so"))
    lib = ctypes.CDLL(path)
    lib.host_gf_deep_stage.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int]
    for fn in (lib.host_gf_shallow_stage, lib.host_gf_gfdrv_stage):
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.host_gf_libm_unary.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int]
    lib.host_gf_libm_pow.argtypes = [
        ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int]
    lib.host_gf_fzu.argtypes = [
        ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int]
    lib.host_gf_const_dump.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    return lib


def _fp(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ip(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _load_word_csv(name):
    xs, ys = [], []
    with (GF_ORACLE_DIR / name).open(encoding="ascii") as fh:
        for line in fh:
            a, b = line.strip().split(",")
            xs.append(int(a, 16))
            ys.append(int(b, 16))
    return (np.array(xs, dtype=np.uint32).view(np.float32),
            np.array(ys, dtype=np.uint32))


def main():
    lib = _lib()
    fixture = load_gf_oracle()
    n = fixture.ncol

    lv = fixture.stage_levels
    sf = fixture.stage_surface
    gs = fixture.surface
    lvin = np.zeros((n, len(IN_LEV), NZ), dtype=np.float32)
    for j, name in enumerate(IN_LEV):
        lvin[:, j, :] = lv[name]
    scin = np.zeros((n, len(IN_SCA)), dtype=np.float32)
    for j, name in enumerate(IN_SCA):
        if name == "dx":
            scin[:, j] = gs["dx"].astype(np.float32)
        elif name in ("fzu_up", "fzu_dn"):
            scin[:, j] = 0.0
        else:
            scin[:, j] = sf[name].astype(np.float32)
    iin = np.ascontiguousarray(sf["kpbli"].astype(np.int32))
    lvin = np.ascontiguousarray(lvin)
    scin = np.ascontiguousarray(scin)
    lev = np.zeros((n, len(LEV_FIELDS), NZ), dtype=np.float32)
    sca = np.zeros((n, len(SCA_FIELDS)), dtype=np.float32)
    isc = np.zeros((n, len(ISCA_FIELDS)), dtype=np.int32)
    lib.host_gf_deep_stage(_fp(lvin), _fp(scin), _ip(iin), _fp(lev),
                           _fp(sca), _ip(isc), n, NZ)

    order = [tuple(int(v) for v in k) for k in fixture.key]
    with (GF_ORACLE_DIR / "gf-deep-surface.csv").open(
            newline="", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    key = {(int(r["case"]), int(r["idx"]), int(r["arm"])): r for r in rows}
    want_s = {k: [] for k in rows[0]}
    for trip in order:
        for k, v in key[trip].items():
            want_s[k].append(v)
    with (GF_ORACLE_DIR / "gf-deep-levels.csv").open(
            newline="", encoding="ascii") as fh:
        lrows = list(csv.DictReader(fh))
    lkey = {(int(r["case"]), int(r["idx"]), int(r["arm"]), int(r["k"])): r
            for r in lrows}
    names_l = [nm for nm in lrows[0] if nm not in ("case", "idx", "arm", "k")]
    want_l = {nm: np.zeros((len(order), NZ), dtype=np.float32)
              for nm in names_l}
    for ci, trip in enumerate(order):
        for k in range(1, NZ + 1):
            r = lkey[trip + (k,)]
            for nm in names_l:
                want_l[nm][ci, k - 1] = np.float32(r[nm])

    def ws(name):
        return np.array([np.float32(v) for v in want_s[name]],
                        dtype=np.float32)

    def wi(name):
        return np.array([int(v) for v in want_s[name]], dtype=np.int64)

    ierr6 = wi("ierr_6") == 0
    k22m = wi("k22_0") > 0
    pminm = (wi("ierr_1") == 0) & ~(
        (ws("frh_kb") >= np.float32(0.97)) & (ws("sig") <= ws("sig_thresh")))
    lev_masks = {nm: ierr6 for nm in
                 ("xhe", "xq", "xt", "xqes", "xhes", "dtempdz")}
    sca_masks = {"hkb0": k22m, "hkbo0": k22m, "xhkb": ierr6, "pr7": ierr6}
    isca_masks = {"pmin_lev": pminm, "start_level": pminm}

    failures = 0
    for j, name in enumerate(LEV_FIELDS):
        got = lev[:, j, :]
        want = (fixture.stage_levels[name] if name in ("cupclw", "cnvwt")
                else want_l[name])
        m = lev_masks.get(name)
        a, b = (got[m], want[m]) if m is not None else (got, want)
        d = fp32_ulp_distance(a, b)
        finite = d >= 0
        w = int(d[finite].max()) if finite.any() else -1
        bad = int(np.count_nonzero(~finite))
        if w != 0 or bad:
            failures += 1
            print(f"LEVEL {name}: max_ulp={w} lanes={int((d != 0).sum())} "
                  f"nonfinite={bad}")
    print(f"level fields bitwise: {len(LEV_FIELDS) - failures}"
          f"/{len(LEV_FIELDS)}")

    fail_s = 0
    for j, name in enumerate(SCA_FIELDS):
        got = sca[:, j]
        want = ws(name)
        m = sca_masks.get(name)
        a, b = (got[m], want[m]) if m is not None else (got, want)
        d = fp32_ulp_distance(a, b)
        finite = d >= 0
        w = int(d[finite].max()) if finite.any() else -1
        if w != 0 or int(np.count_nonzero(~finite)):
            fail_s += 1
            print(f"SCALAR {name}: max_ulp={w} "
                  f"lanes={int((d != 0).sum())}")
    print(f"float scalars bitwise: {len(SCA_FIELDS) - fail_s}"
          f"/{len(SCA_FIELDS)}")

    fail_i = 0
    for j, name in enumerate(ISCA_FIELDS):
        got = isc[:, j].astype(np.int64)
        want = wi(name)
        m = isca_masks.get(name)
        a, b = (got[m], want[m]) if m is not None else (got, want)
        neq = int((a != b).sum())
        if neq:
            fail_i += 1
            print(f"INT {name}: differs on {neq} columns")
    print(f"int fields exact: {len(ISCA_FIELDS) - fail_i}/{len(ISCA_FIELDS)}")

    slot = {"tgammaf": 0, "lgammaf": 2, "expm1f": 3, "exp2f": 4}
    for fn_name, s in sorted(slot.items()):
        x, wantw = _load_word_csv(f"gf-libm-{fn_name}.csv")
        x = np.ascontiguousarray(x)
        out = np.zeros(7 * x.size, dtype=np.float32)
        lib.host_gf_libm_unary(_fp(x), _fp(out), x.size)
        got = out.reshape(-1, 7)[:, s].copy().view(np.uint32)
        nd = int(np.count_nonzero(got != wantw))
        print(f"libm {fn_name}: {x.size} args, {nd} word mismatches")
        failures += (nd != 0)

    # ---- the shallow stage, WRF-faithful k22, fzu computed ----------------
    ncase = 18
    col_of_case = {}
    for ci, (case, idx, arm) in enumerate(fixture.key):
        col_of_case.setdefault(int(case), ci)
    shin = np.zeros((ncase, len(SH_IN_LEV), NZ), dtype=np.float32)
    shsc = np.zeros((ncase, len(SH_IN_SCA)), dtype=np.float32)
    shii = np.zeros(ncase, dtype=np.int32)
    for case in range(1, ncase + 1):
        ci = col_of_case[case]
        for j, name in enumerate(SH_IN_LEV):
            shin[case - 1, j, :] = lv[name][ci]
        for j, name in enumerate(SH_IN_SCA):
            shsc[case - 1, j] = 0.0 if name == "fzu_sh" else sf[name][ci]
        shii[case - 1] = int(sf["kpbli"][ci])
    shin = np.ascontiguousarray(shin)
    shsc = np.ascontiguousarray(shsc)
    slev = np.zeros((ncase, len(SH_LEV_FIELDS), NZ), dtype=np.float32)
    ssca = np.zeros((ncase, len(SH_SCA_FIELDS)), dtype=np.float32)
    sisc = np.zeros((ncase, len(SH_ISCA_FIELDS)), dtype=np.int32)
    lib.host_gf_shallow_stage(_fp(shin), _fp(shsc), _ip(shii), _fp(slev),
                              _fp(ssca), _ip(sisc), 1, ncase, NZ)

    with (GF_ORACLE_DIR / "gf-shallow-surface.csv").open(
            newline="", encoding="ascii") as fh:
        srows = list(csv.DictReader(fh))
    sws = {k: [None] * ncase for k in srows[0]}
    for r in srows:
        for k, v in r.items():
            sws[k][int(r["case"]) - 1] = v

    fail_sh = 0
    for j, name in enumerate(SH_LEV_FIELDS):
        got = slev[:, j, :]
        want = fixture.shallow_levels[name]
        d = fp32_ulp_distance(got, want)
        finite = d >= 0
        w = int(d[finite].max()) if finite.any() else -1
        if w != 0 or int(np.count_nonzero(~finite)):
            fail_sh += 1
            print(f"SH LEVEL {name}: max_ulp={w} "
                  f"lanes={int((d != 0).sum())}")
    for j, name in enumerate(SH_SCA_FIELDS):
        got = ssca[:, j]
        want = np.array([np.float32(v) for v in sws[name]],
                        dtype=np.float32)
        d = fp32_ulp_distance(got, want)
        finite = d >= 0
        w = int(d[finite].max()) if finite.any() else -1
        if w != 0 or int(np.count_nonzero(~finite)):
            fail_sh += 1
            print(f"SH SCALAR {name}: max_ulp={w} "
                  f"lanes={int((d != 0).sum())}")
    for j, name in enumerate(SH_ISCA_FIELDS):
        got = sisc[:, j].astype(np.int64)
        want = np.array([int(v) for v in sws[name]], dtype=np.int64)
        neq = int((got != want).sum())
        if neq:
            fail_sh += 1
            print(f"SH INT {name}: differs on {neq} cases "
                  f"got={got.tolist()} want={want.tolist()}")
    print(f"shallow fields bitwise: "
          f"{len(SH_LEV_FIELDS) + len(SH_SCA_FIELDS) + len(SH_ISCA_FIELDS) - fail_sh}"
          f"/{len(SH_LEV_FIELDS) + len(SH_SCA_FIELDS) + len(SH_ISCA_FIELDS)}")

    # ---- the whole driver, WRF-faithful k22, fzu computed -----------------
    gl = fixture.levels
    gs = fixture.surface
    din = np.zeros((n, len(DRV_IN_LEV), NZ), dtype=np.float32)
    dsc = np.zeros((n, len(DRV_IN_SCA)), dtype=np.float32)
    dii = np.zeros((n, 3), dtype=np.int32)
    for j, name in enumerate(DRV_IN_LEV):
        din[:, j, :] = gl[name]
    for j, name in enumerate(DRV_IN_SCA):
        dsc[:, j] = gs[name].astype(np.float32)
    dii[:, 0] = gs["kpbl"].astype(np.int32)
    dii[:, 1] = gs["ishallow"].astype(np.int32)
    dii[:, 2] = gs["ichoice"].astype(np.int32)
    din = np.ascontiguousarray(din)
    dsc = np.ascontiguousarray(dsc)
    dii = np.ascontiguousarray(dii)
    dlev = np.zeros((n, len(DRV_LEV_FIELDS), NZ), dtype=np.float32)
    dsca = np.zeros((n, len(DRV_SCA_FIELDS)), dtype=np.float32)
    disc = np.zeros((n, len(DRV_ISCA_FIELDS)), dtype=np.int32)
    lib.host_gf_gfdrv_stage(_fp(din), _fp(dsc), _ip(dii), _fp(dlev),
                            _fp(dsca), _ip(disc), 1, n, NZ)

    from gpuwm.verify.gf_oracle import stage_rows_to_distrust
    exact = ~stage_rows_to_distrust(fixture)
    fail_d = 0
    # the pinned GFDRV boundary, on the 208 columns the driver's own
    # decomposition reproduces
    for j, name in enumerate(
            ["rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt_phy",
             "dvdt_phy", "gdc", "gdc2"]):
        got = dlev[:, DRV_LEV_FIELDS.index(name), :][exact]
        want = gl[name][exact]
        d = fp32_ulp_distance(got, want)
        w = int(d[d >= 0].max()) if (d >= 0).any() else -1
        if w != 0:
            fail_d += 1
            print(f"DRV {name}: max_ulp={w} lanes={int((d != 0).sum())}")
    for name in ["raincv", "pratec"]:
        got = dsca[:, DRV_SCA_FIELDS.index(name)][exact]
        want = gs[name][exact]
        d = fp32_ulp_distance(got, want)
        w = int(d[d >= 0].max()) if (d >= 0).any() else -1
        if w != 0:
            fail_d += 1
            print(f"DRV {name}: max_ulp={w}")
    for name in ["htop", "hbot", "xmb_shallow"]:
        got = dsca[:, DRV_SCA_FIELDS.index(name)]
        want = gs[name]
        d = fp32_ulp_distance(got, want)
        w = int(d[d >= 0].max()) if (d >= 0).any() else -1
        if w != 0:
            fail_d += 1
            print(f"DRV {name} (all 216): max_ulp={w}")
    for name in ["ktop_deep", "k22_shallow", "kbcon_shallow",
                 "ktop_shallow"]:
        got = disc[:, DRV_ISCA_FIELDS.index(name)].astype(np.int64)
        want = gs[name].astype(np.int64)
        neq = int((got != want).sum())
        if neq:
            fail_d += 1
            print(f"DRV {name}: differs on {neq} columns")
    # the stage seam on ALL 216 columns
    idx = {}
    ss = fixture.stage_surface
    for i in range(ss["case"].shape[0]):
        idx[(int(ss["case"][i]), int(ss["idx"][i]), int(ss["arm"][i]))] = i
    order = [idx[tuple(int(v) for v in k)] for k in fixture.key]
    for name in ["outt", "outq", "outqc", "outu", "outv", "outts", "outqs",
                 "outqcs"]:
        got = dlev[:, DRV_LEV_FIELDS.index(name), :]
        want = fixture.stage_levels[name]
        d = fp32_ulp_distance(got, want)
        w = int(d[d >= 0].max()) if (d >= 0).any() else -1
        if w != 0:
            fail_d += 1
            print(f"DRV seam {name}: max_ulp={w} "
                  f"lanes={int((d != 0).sum())}")
    for name in ["pret", "prets"]:
        got = dsca[:, DRV_SCA_FIELDS.index(name)]
        want = ss[name][order]
        d = fp32_ulp_distance(got, want)
        w = int(d[d >= 0].max()) if (d >= 0).any() else -1
        if w != 0:
            fail_d += 1
            print(f"DRV seam {name}: max_ulp={w}")
    # the 8 distrust columns: amplitude-only, bounded as the CPU gate bounds
    bad = ~exact
    got = dlev[:, DRV_LEV_FIELDS.index("rthcuten"), :][bad]
    want = gl["rthcuten"][bad]
    d = fp32_ulp_distance(got, want)
    live = np.abs(want) > np.float32(1e-12)
    rel = np.abs(got[live] - want[live]) / np.abs(want[live])
    print(f"driver 8-column residual: max_ulp={int(d[d >= 0].max())} "
          f"max_rel={float(rel.max()):.3e} (bounds: 34, 1e-4)")
    if int(d[d >= 0].max()) > 34 or float(rel.max()) >= 1e-4:
        fail_d += 1
    print(f"driver gate failures: {fail_d}")

    sys.exit(1 if (failures or fail_s or fail_i or fail_sh or fail_d)
             else 0)


if __name__ == "__main__":
    main()
