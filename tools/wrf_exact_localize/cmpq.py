"""cmpq.py WRF_DUMP WOOF_DUMP [STEP]: the qv transport path at each RK stage of one step."""
import sys, glob, re, os
import numpy as np
wd, od = sys.argv[1], sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme = map(int, open(f"{wd}/dims.txt").read().split())
def W(rk, stage, name, like):
    f = f"{wd}/s{step:03d}_rk{rk}_it00_{stage}__{name}.bin"
    if not os.path.exists(f): return None
    raw = open(f, "rb").read()
    i0, i1, k0, k1, j0, j1 = np.frombuffer(raw[:24], "<i4")
    a = np.frombuffer(raw[24:], "<f4").reshape((i1 - i0 + 1, k1 - k0 + 1, j1 - j0 + 1), order="F")
    nk, nj, ni = like.shape
    return np.ascontiguousarray(a[ids - i0: ids - i0 + ni, kds - k0: kds - k0 + nk, jds - j0: jds - j0 + nj].transpose(1, 2, 0))
def Z(rk, tag):
    rks = {1: (1, 10), 2: (2, 11), 3: (3, 12)}[rk]
    f = sorted(x for r in rks for x in glob.glob(f"{od}/*_s{step:03d}_rk{r}_it*_{tag}.npz"))
    return np.load(f[0]) if f else None
def rep(label, a, b):
    if a is None or b is None: print(f"  {label:34s} (missing)"); return
    b = b.astype(np.float32)
    eq = (a.view(np.int32) == b.view(np.int32)) | ((a == 0) & (b == 0))
    if eq.all(): print(f"  {label:34s} IDENTICAL"); return
    bad = np.argwhere(~eq); jj, ii = bad[:, -2], bad[:, -1]; ny, nx = a.shape[-2:]
    edge = np.minimum.reduce([jj, ii, ny - 1 - jj, nx - 1 - ii])
    print(f"  {label:34s} differ {len(bad)}/{a.size} first {tuple(int(x) for x in bad[0])} min edge {edge.min()} edge<5 {np.mean(edge<5)*100:.0f}%")
for rk in (1, 2, 3):
    print("=== RK", rk)
    z0, z1, z2, z9 = Z(rk, "q0_fluxes"), Z(rk, "q1_advect"), Z(rk, "q2_update_in"), Z(rk, "q9_updated")
    if z0 is not None:
        for n in ("ru_m", "rv_m", "ww_m"):
            rep(n, W(rk, "q1_sctend", n, z0["x_" + n]), z0["x_" + n])
    if z1 is not None:
        rep("advect_tend vs flux_div", W(rk, "q1_sctend", "advect_tend", z1["x_advect"]), z1["x_advect"])
    if z2 is not None:
        f = z2["x_fixed"] if "x_fixed" in z2 else None
        if f is not None:
            rep("moist_tend(sc) vs WOOF fixed", W(rk, "q1_sctend", "moist_tend", f), f)
            rep("moist_tend after relax vs fixed", W(rk, "q2_relax", "moist_tend", f), f)
        print("  WOOF update keys:", [k for k in z2.files if k.startswith("x_")])
    if z9 is not None:
        rep("qv after update (WRF i_pphi)", W(rk, "i_pphi", "qv", z9["qv"]), z9["qv"])
