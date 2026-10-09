"""cmp4.py WRF_DUMP WOOF_DUMP [step] [--interior N]: the localize kit's matched points, plus the base4
points (Smagorinsky h/v order, acoustic coefficients, folded relaxation).  Every word compared."""
import glob
import os
import re
import sys

import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
wd, od = args[0], args[1]
step = int(args[2]) if len(args) > 2 else 1
ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme = map(int, open(f"{wd}/dims.txt").read().split())


def wrf(rk, it, stage, name, like):
    f = f"{wd}/s{step:03d}_rk{rk}_it{it:02d}_{stage}__{name}.bin"
    if not os.path.exists(f) or like is None:
        return None
    raw = open(f, "rb").read()
    i0, i1, k0, k1, j0, j1 = np.frombuffer(raw[:24], "<i4")
    a = np.frombuffer(raw[24:], "<f4").reshape((i1 - i0 + 1, k1 - k0 + 1, j1 - j0 + 1), order="F")
    if like.ndim == 2:
        nj, ni = like.shape
        a = a[ids - i0: ids - i0 + ni, 0, jds - j0: jds - j0 + nj].T
    else:
        nk, nj, ni = like.shape
        a = a[ids - i0: ids - i0 + ni, kds - k0: kds - k0 + nk, jds - j0: jds - j0 + nj].transpose(1, 2, 0)
    return np.ascontiguousarray(a)


woof = {}
for f in sorted(glob.glob(f"{od}/*_s{step:03d}_*.npz")):
    m = re.match(r"\d+_s\d+_rk(\d+)_it(\d+)_(.+)\.npz", os.path.basename(f))
    rk = int(m[1])
    # the hook's stage counter runs on across steps: 9 before a step's
    # stages, then 10..12 for steps after the first
    rk = 0 if rk == 9 else (rk - 9 if rk > 9 else rk)
    if m[3] == "z_after_wsurface":
        rk = 9
    woof.setdefault((rk, int(m[2]), m[3]), f)
NDIFF = [0]


def ulp(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, -(ai & 0x7fffffff), ai)
    bi = np.where(bi < 0, -(bi & 0x7fffffff), bi)
    return np.abs(ai - bi)


def report(label, a, b):
    if a is None or b is None:
        return
    b = np.asarray(b, np.float32)
    if a.shape != b.shape:
        print(f"  {label:30s} shape {a.shape} vs {b.shape}")
        return
    eq = a.view(np.int32) == b.view(np.int32)
    eq |= (a == 0) & (b == 0)
    if eq.all():
        print(f"  {label:30s} IDENTICAL ({a.size})")
        return
    NDIFF[0] += 1
    bad = np.argwhere(~eq)
    u = ulp(a, b)
    d = np.abs(a.astype(np.float64) - b)
    jj, ii = bad[:, -2], bad[:, -1]
    ny, nx = a.shape[-2], a.shape[-1]
    edge = np.minimum.reduce([jj, ii, ny - 1 - jj, nx - 1 - ii])
    ks = sorted(set(int(x) for x in bad[:, 0])) if a.ndim == 3 else []
    print(f"  {label:30s} differ {len(bad)}/{a.size}, max|d| {d.max():.3e}, med ulp {int(np.median(u[~eq]))}, "
          f"max ulp {int(u.max())}, first {tuple(int(x) for x in bad[0])}, min edge {int(edge.min())}, "
          f"edge<5 {np.mean(edge < 5)*100:.0f}%" + (f", k {ks[0]}..{ks[-1]} ({len(ks)})" if ks else ""))


def W(key):
    f = woof.get(key)
    return np.load(f) if f else None


STATE = [("u_2", "u"), ("v_2", "v"), ("w_2", "w"), ("t_2", "thp"), ("ph_2", "php"), ("mu_2", "mup")]
TEND = (("ru_tend", "ru_t"), ("rv_tend", "rv_t"), ("rw_tend", "rw_t"), ("t_tend", "rth_t"), ("ph_tend", "rph_t"),
        ("mu_tend", "rmu_t"))
SS = (("u_2", "u_pp"), ("v_2", "v_pp"), ("w_2", "w_pp"), ("t_2", "th_pp"), ("ph_2", "ph_pp"), ("mu_2", "mu_pp"),
      ("ww", "ww_pp"), ("p", "p_pp"), ("al", "al_pp"))

# held tendencies (rk 0 in WOOF = before the stage loop)
z1, z2 = W((0, 0, "c01_hsmag")), W((0, 0, "c02_vsmag"))
if z2 is not None:
    print("=== held tendencies (first_rk_step_part2)")
    for n, s in (("ru_tendf", "ru"), ("rv_tendf", "rv"), ("rw_tendf", "rw"), ("t_tendf", "rth")):
        if f"x_hv_{s}" not in z2:
            continue
        wv = wrf(1, 0, "c1_vdiff", n, z2[f"x_hv_{s}"])
        wh = wrf(1, 0, "c2_hdiff", n, z2[f"x_hv_{s}"])
        report(f"{n} v+h vs WOOF h,v", wh, z2[f"x_hv_{s}"])
        if wv is not None and z1 is not None:
            h = z1[f"x_h_{s}"].astype(np.float32)
            report(f"  fl(WRFv + WOOFh) vs WRF", wh, (wv + h).astype(np.float32))
            report(f"  fl(WOOFh + WRFv) vs WOOF", z2[f"x_hv_{s}"], (h + wv).astype(np.float32))
z = W((0, 0, "c_fixed"))
if z is not None:
    print("=== held tendencies after diff6 (WRF d_rktend *_tendf)")
    for n, s in (("ru_tendf", "smag_ru"), ("rv_tendf", "smag_rv"), ("rw_tendf", "smag_rw"), ("t_tendf", "smag_rth")):
        if "x_" + s in z:
            report(f"{n} vs {s} (WOOF undivided?)", wrf(1, 0, "d_rktend", n, z["x_" + s]), z["x_" + s])
for rk in (1, 2, 3):
    print(f"=== RK stage {rk}")
    z = W((rk, 0, "b_prep"))
    if z is not None:
        for wn, on in STATE + [("al", "al"), ("alt", "alt"), ("ru", "x_ru"), ("rv", "x_rv"), ("ww", "x_ww")]:
            report(f"b {wn}", wrf(rk, 0, "b_prep", wn, z[on]), z[on])
        report("b p", wrf(rk, 0, "b_prep", "p", z["p"]), z["p_perturbation"] if "p_perturbation" in z else z["p"] - z["pb"])
    for tag in ("k4_advt", "k1_advu", "k2_advv", "k3_advw", "k5_rhsph", "k6_hpg", "k7_buoy", "kA_corcurv"):
        zk = W((rk, 0, tag))
        if zk is None:
            continue
        wtag = {"kA_corcurv": "kA_curv"}.get(tag, tag)
        for wn, on in TEND[:5]:
            a = wrf(rk, 0, wtag, wn, zk[on])
            if a is not None:
                report(f"{tag} {wn}", a, zk[on])
    for wst, ost in (("d_rktend", "d_slow"), ("e_addtend", "e_lateral")):
        z = W((rk, 0, ost))
        if z is None:
            continue
        for wn, on in TEND:
            report(f"{wst[:6]} {wn}", wrf(rk, 0, wst, wn, z[on]), z[on])
    z = W((rk, 0, "e0_relax"))
    if z is not None:
        for wn, on in (("u_save", "x_relax_u"), ("v_save", "x_relax_v"), ("t_save", "x_relax_theta"),
                       ("w_save", "x_relax_w")):
            if on in z:
                report(f"relax {wn}", wrf(rk, 0, "e_addtend", wn, z[on]), z[on])
    z = W((rk, 0, "f0_coef"))
    if z is not None and rk == 1:
        for wn, on in (("alpha", "x_alpha"), ("gamma", "x_gam"), ("a", "x_a")):
            report(f"coef {wn}", wrf(rk, 0, "w9_adv", wn, z[on]), z[on])
    z = W((rk, 0, "f_ssinit"))
    if z is not None:
        for wn, on in SS:
            report(f"ssprep {wn}", wrf(rk, 0, "f_ssprep", wn, z[on]), z[on])
    for it in range(1, 9):
        z = W((rk, it, "g_ss"))
        if z is None:
            continue
        for wn, on in SS:
            report(f"ss{it} {wn}", wrf(rk, it, "g_ss", wn, z[on]), z[on])
    z = None
    for it in range(0, 9):
        z = W((rk, it, "h_finish")) or z
    if z is not None:
        for wn, on in STATE:
            report(f"finish {wn}", wrf(rk, 0, "h_finish", wn, z[on]), z[on])
z = None
if z is None:
    cands = [k for k in woof if k[2] == "z_after_wsurface"]
    z = W(cands[-1]) if cands else None
if z is not None:
    print("=== end of step (WRF end of solve_em vs WOOF after the step epilogue)")
    for wn, on in STATE + [("p", "p_perturbation"), ("al", "al"), ("qv", "qv")]:
        if on in z:
            report(f"end {wn}", wrf(9, 0, "z_end", wn, z[on]), z[on])
print(f"DIFFERING POINTS: {NDIFF[0]}")
