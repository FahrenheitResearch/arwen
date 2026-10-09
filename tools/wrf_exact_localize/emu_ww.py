"""emu_ww.py WRF_DUMP WOOF_DUMP: emulate WRF calc_ww_cp in float32 from dumped words, stages 1-3 of step 1."""
import sys, glob, re, os
import numpy as np
f32 = np.float32
wd, od = sys.argv[1], sys.argv[2]
ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme = map(int, open(f"{wd}/dims.txt").read().split())
def wrf_full(rk, stage, name):
    raw = open(f"{wd}/s001_rk{rk}_it00_{stage}__{name}.bin", "rb").read()
    i0, i1, k0, k1, j0, j1 = np.frombuffer(raw[:24], "<i4")
    a = np.frombuffer(raw[24:], "<f4").reshape((i1 - i0 + 1, k1 - k0 + 1, j1 - j0 + 1), order="F")
    return a, (i0, k0, j0)
def crop(a, o, nk, nj, ni, ioff=0, joff=0):
    i0, k0, j0 = o
    return a[ids - i0 + ioff: ids - i0 + ioff + ni, kds - k0: kds - k0 + nk, jds - j0 + joff: jds - j0 + joff + nj].transpose(1, 2, 0)
z0 = np.load(sorted(glob.glob(f"{od}/*_s001_rk1_it00_b_prep.npz"))[0])
nz, ny, nx = z0["p"].shape
mub = z0["mub2d"].astype(f32); dnw = z0["dnw"].astype(f32); c1h = z0["c1h"].astype(f32); c2h = z0["c2h"].astype(f32)
msft = z0["msft"].astype(f32); msfu = z0["msfu"].astype(f32); msfv = z0["msfv"].astype(f32)
msfv_inv = (f32(1) / msfv).astype(f32)
rdx = f32(1) / f32(3000.); rdy = rdx
for rk in (1, 2, 3):
    mu_a, o = wrf_full(rk, "b_prep", "mu_2")
    u_a, ou = wrf_full(rk, "b_prep", "u_2"); v_a, ov = wrf_full(rk, "b_prep", "v_2"); ww_a, ow = wrf_full(rk, "b_prep", "ww")
    mu = crop(mu_a, o, 1, ny, nx)[0]
    # halo column/row (WRF memory i = ids-1) for the edge faces
    muh_w = crop(mu_a, o, 1, ny, 1, ioff=-1)[0][:, 0]; muh_s = crop(mu_a, o, 1, 1, nx, joff=-1)[0][0]
    u = crop(u_a, ou, nz, ny, nx + 1); v = crop(v_a, ov, nz, ny + 1, nx); ww_w = crop(ww_a, ow, nz + 1, ny, nx)
    MUP = np.zeros((ny, nx + 2), f32); MUB = np.zeros_like(MUP)
    MUP[:, 1:-1] = mu; MUP[:, 0] = muh_w; MUP[:, -1] = crop(mu_a, o, 1, ny, 1, ioff=nx)[0][:, 0]
    MUB[:, 1:-1] = mub; MUB[:, 0] = mub[:, 0]; MUB[:, -1] = mub[:, -1]
    muu = (f32(0.5) * (((MUP[:, 1:] + MUB[:, 1:]).astype(f32) + MUP[:, :-1]).astype(f32) + MUB[:, :-1]).astype(f32)).astype(f32)
    MVP = np.zeros((ny + 2, nx), f32); MVB = np.zeros_like(MVP)
    MVP[1:-1] = mu; MVP[0] = muh_s; MVP[-1] = crop(mu_a, o, 1, 1, nx, joff=ny)[0][0]
    MVB[1:-1] = mub; MVB[0] = mub[0]; MVB[-1] = mub[-1]
    muv = (f32(0.5) * (((MVP[1:] + MVB[1:]).astype(f32) + MVP[:-1]).astype(f32) + MVB[:-1]).astype(f32)).astype(f32)
    ww = np.zeros((nz + 1, ny, nx), f32); dmdt = np.zeros((ny, nx), f32); divv = np.zeros((nz, ny, nx), f32)
    for k in range(nz):
        cu = (c1h[k] * muu + c2h[k]).astype(f32); cv = (c1h[k] * muv + c2h[k]).astype(f32)
        fu = ((cu * u[k]).astype(f32) / msfu).astype(f32)
        fv = ((cv * v[k]).astype(f32) * msfv_inv).astype(f32)
        br = (rdx * (fu[:, 1:] - fu[:, :-1]).astype(f32)).astype(f32) + (rdy * (fv[1:] - fv[:-1]).astype(f32)).astype(f32)
        divv[k] = ((msft * dnw[k]).astype(f32) * br.astype(f32)).astype(f32)
        dmdt = (dmdt + divv[k]).astype(f32)
    for k in range(1, nz):
        ww[k] = ((ww[k - 1] - ((dnw[k - 1] * c1h[k - 1]).astype(f32) * dmdt).astype(f32)).astype(f32) - divv[k - 1]).astype(f32)
    bad = ww.view(np.int32) != ww_w.view(np.int32)
    zz = np.load(sorted(glob.glob(f"{od}/*_s001_rk{rk}_it00_b_prep.npz"))[0])
    wo = zz["x_ww"]
    print(f"rk{rk}: emul vs WRF differ {bad.sum()}/{bad.size}; WOOF vs emul {(wo.view(np.int32) != ww.view(np.int32)).sum()}; "
          f"halo mu west == edge col: {np.array_equal(muh_w, mu[:, 0])}, wrf halo-edge max {np.abs(muh_w - mu[:,0]).max():.3e}")
