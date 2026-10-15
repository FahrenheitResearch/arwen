"""Part-by-part DIAGNOSTIC comparison of WOOF against the instrumented WRF copy.

    python probe.py IN.bin my2dump.bin

Runs ONE call of the shipped launcher on the first step's input, snapshotting
the state after each kernel (the kernels split exactly at WRF's part
boundaries), and compares each snapshot with the instrumented Fortran's dump
at the same checkpoint:

    prelim+geometry  <->  tag 400 (end of Part 1)
    cold             <->  tag 450 (end of Part 2)
    warm             <->  tag 500 (end of Part 3)
    sediment         <->  tag 800 (end of Part 4)
    diagnostics      <->  tag 900 (end of scheme, N back to #/kg)

and the constant vector with the Fortran SAVE block.  Diagnostic only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STATE = ("T", "Q", "QC", "QR", "QI", "QN", "QG", "QH",
         "NC", "NR", "NY", "NN", "NG", "NH")
AUX = ("DE", "QSW", "QSI", "pres", "gamfact", "DZ", "iDZ")


def read_dump(path):
    raw = Path(path).read_bytes()
    off = 0
    states, consts = [], {}
    while off < len(raw):
        kind = int(np.frombuffer(raw[off:off + 4], "<i4")[0])
        if kind == 1:
            tag, ni, nk = np.frombuffer(raw[off + 4:off + 16], "<i4")
            off += 16
            rec = {}
            for name in STATE + AUX:
                n = int(ni) * int(nk)
                rec[name] = np.frombuffer(raw[off:off + 4 * n], "<f4").reshape(
                    (ni, nk), order="F").copy()
                off += 4 * n
            states.append((int(tag), rec))
        elif kind == 2:
            which, n = np.frombuffer(raw[off + 4:off + 12], "<i4")
            off += 12
            consts.setdefault(int(which), []).append(
                np.frombuffer(raw[off:off + 4 * int(n)], "<f4").copy())
            off += 4 * int(n)
        else:
            raise SystemExit(f"bad record kind {kind} at {off}")
    return states, consts


def ordered(a):
    i = a.view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def report(label, ref, got, names, mask=None):
    print(f"--- {label}")
    for nm in names:
        a, b = ref[nm], got[nm]
        if mask is not None:
            a, b = a[:, mask], b[:, mask]
        bits = a.view(np.uint32) != b.view(np.uint32)
        if bits.any():
            u = np.abs(ordered(a) - ordered(b))
            i, k = np.argwhere(bits)[0]
            print(f"  {nm:8s} differ {int(bits.sum()):6d}  maxULP {int(u.max()):11d}"
                  f"  first (col {i}, k {k}) wrf={a[i, k]!r} woof={b[i, k]!r}")
    print("  (all other fields bit-identical)")


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cupy as cp
    import gpuwm  # noqa: F401
    import gpuwm.core.milbrandt2 as my2
    from gpuwm.core.milbrandt2_constants import CK_ORDER, ck_vector
    from gen_columns import read_stream

    in_path, dump_path = sys.argv[1], sys.argv[2]
    states, consts = read_dump(dump_path)
    ncol, nz, nsteps, dt, arr, sfc = read_stream(in_path)

    # constants
    late = ("cxr", "cxi", "Gzr", "Gzi", "Gzs", "Gzg", "Gzh")
    early = [n for n in CK_ORDER if n not in late]
    ck = dict(zip(CK_ORDER, ck_vector()))
    fort = dict(zip(early, consts[1][0]))
    fort.update(zip(late, consts[2][0]))
    bad = [(n, fort[n], ck[n]) for n in CK_ORDER
           if np.float32(fort[n]).view(np.uint32) != np.float32(ck[n]).view(np.uint32)]
    print(f"constants: {len(CK_ORDER) - len(bad)} of {len(CK_ORDER)} bit-identical")
    for n, a, b in bad:
        print(f"  {n}: wrf={a!r} woof={b!r}")

    def dev3(a):
        return cp.ascontiguousarray(cp.asarray(np.ascontiguousarray(a.T).reshape(nz, 1, ncol)))

    f = {k: dev3(arr[k]) for k in ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "nc",
                                   "nr", "ni", "ns", "ng", "nh", "th", "pii", "p", "w")}
    psfc = cp.asarray(sfc["p8w"].reshape(1, ncol))
    acc = {k: cp.asarray(sfc[k].reshape(1, ncol)) for k in
           ("rainnc", "snownc", "grplnc", "hailnc")}
    z2 = lambda: cp.zeros((1, ncol), cp.float32)  # noqa: E731
    rainncv, snowncv, grplncv, hailncv, sr = (z2() for _ in range(5))
    zet = cp.zeros((nz, 1, ncol), cp.float32)
    scratch = {k: cp.zeros((nz, 1, ncol), cp.float32) for k in
               ("pres", "de", "ide", "dz", "idz", "gamfact", "qsw", "qsi",
                "qc_in", "qr_in", "nc_in", "nr_in")}
    temperature = f["th"] * f["pii"]

    def host(a):
        return cp.asnumpy(a).reshape(nz, ncol).T

    def snap():
        names = dict(T=temperature, Q=f["qv"], QC=f["qc"], QR=f["qr"], QI=f["qi"],
                     QN=f["qs"], QG=f["qg"], QH=f["qh"], NC=f["nc"], NR=f["nr"],
                     NY=f["ni"], NN=f["ns"], NG=f["ng"], NH=f["nh"],
                     DE=scratch["de"], QSW=scratch["qsw"], QSI=scratch["qsi"],
                     pres=scratch["pres"], gamfact=scratch["gamfact"],
                     DZ=scratch["dz"], iDZ=scratch["idz"])
        return {k: host(v) for k, v in names.items()}

    snaps = {}
    real_get = my2.get_kernel
    order = {"milbrandt2_geometry": 400, "milbrandt2_cold": 450,
             "milbrandt2_warm": 500, "milbrandt2_sediment_64": 800,
             "milbrandt2_sediment_256": 800, "milbrandt2_diagnostics": 900}

    def get_kernel(module, name):
        k = real_get(module, name)
        if name not in order:
            return k

        def run(*a, **kw):
            k(*a, **kw)
            cp.cuda.Device().synchronize()
            snaps[order[name]] = snap()
        return run

    my2.get_kernel = get_kernel
    my2.launch_milbrandt2(
        temperature, f["qv"], f["qc"], f["qr"], f["qi"], f["qs"], f["qg"], f["qh"],
        f["nc"], f["nr"], f["ni"], f["ns"], f["ng"], f["nh"], f["w"], f["p"], psfc,
        float(dt), acc["rainnc"], rainncv, acc["snownc"], snowncv, acc["grplnc"],
        grplncv, acc["hailnc"], hailncv, sr, zet, **scratch)
    my2.get_kernel = real_get

    first = {}
    for tag, rec in states:
        first.setdefault(tag, rec)       # step 1 only
    aux = {400: AUX, 450: (), 500: (), 800: (), 900: ()}
    for tag in (400, 450, 500, 800, 900):
        report(f"tag {tag}", first[tag], snaps[tag], STATE + aux[tag])


if __name__ == "__main__":
    main()
