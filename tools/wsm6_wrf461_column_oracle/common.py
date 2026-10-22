"""Shared input arithmetic and word grading for the WRF WSM column oracles.

The scheme drivers retain their field layouts, regimes and declared divergences.
These helpers keep the copied float32 operations and report ordering identical.
"""

from pathlib import Path
import hashlib
import numpy as np

F32 = np.float32
RD, CP, P0 = 287.0, 1004.5, 1.0e5


def _interfaces(dz0: float = 40.0, growth: float = 1.08,
                dzmax: float = 700.0, nz: int = 45) -> np.ndarray:
    dz = np.empty(nz)
    d = dz0
    for k in range(nz):
        dz[k] = d
        d = min(d * growth, dzmax)
    return dz


def _qsat(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Bolton saturation mixing ratio over liquid (float64; input shaping only)."""
    tc = t - 273.15
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    es = np.minimum(es, 0.5 * p)
    return 0.622 * es / (p - es)


def _bump(z: np.ndarray, z1: float, z2: float, peak: float) -> np.ndarray:
    """A smooth hump of ``peak`` between heights z1 and z2 (zero outside)."""
    if peak == 0.0 or z2 <= z1:
        return np.zeros_like(z)
    x = (z - z1) / (z2 - z1)
    shape = np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 1.5, 0.0)
    return peak * shape


def profile(dz, *, ps, ts, lapse, ztrop, inv_depth, inv_dt,
            warm_nose, rh, seed, strat=1.5e-3):
    """The common hydrostatic, randomly perturbed input column (float64)."""
    zi = np.concatenate(([0.0], np.cumsum(dz)))
    z = 0.5 * (zi[:-1] + zi[1:])
    t = ts - lapse * np.minimum(z, ztrop) + strat * np.maximum(z - ztrop - 2000.0, 0.0)
    if inv_depth > 0.0:
        ramp = np.clip(1.0 - z / inv_depth, 0.0, 1.0)
        t = np.where(z < inv_depth, ts - inv_dt * ramp,
                     ts - lapse * (np.minimum(z, ztrop) - inv_depth)
                     + strat * np.maximum(z - ztrop - 2000.0, 0.0))
    if warm_nose is not None:
        zb, zt, amp = warm_nose
        t = t + _bump(z, zb, zt, amp)
    rng = np.random.default_rng(seed)
    t = t + rng.normal(0.0, 0.15, z.size)
    p = np.empty_like(z)
    lnp = np.log(ps)
    zprev = 0.0
    for k in range(z.size):
        lnp -= 9.81 * (z[k] - zprev) / (RD * t[k])
        p[k] = np.exp(lnp)
        zprev = z[k]
    rh = (0.7 * np.exp(-z / 6000.0) + 0.05) if rh is None else rh(z)
    qv = np.maximum(rh * _qsat(t, p), 1.0e-7)
    return zi, z, t, p, qv


def driver_fields(out, exact_t=None):
    """WRF driver quantities, with the original float32 operation order."""
    pii = np.power(out["p"] / F32(P0), F32(RD / CP), dtype=F32)
    th = (out["t"] / pii).astype(F32)
    if exact_t is not None:
        for k, tk in exact_t.items():
            pii[k] = F32(1.0)
            th[k] = F32(tk)
    tv = out["t"] * (F32(1.0) + F32(0.608) * np.maximum(out["qv"], F32(0.0)))
    den = (out["p"] / (F32(RD) * tv)).astype(F32)
    out.update(th=th, pii=pii.astype(F32), den=den)
    return out


def micron(re_metres):
    return (np.asarray(re_metres, np.float32) * np.float32(1.0e6)).astype(np.float32)


def _ordered(a):
    bits = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def ulp(a, b):
    return np.abs(_ordered(a) - _ordered(b))


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, "<f4").tobytes()).hexdigest()


def _header():
    import cupy as cp
    import gpuwm.wrf_exact as wrf_exact

    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return {"strict_build": bool(wrf_exact.ENABLED),
            "device": name.decode() if isinstance(name, bytes) else str(name),
            "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
            "cupy": cp.__version__}


def read_columns(path: Path, level_fields, surface_fields):
    with open(path, "rb") as fh:
        ncol, nz = (int(v) for v in np.fromfile(fh, "<i4", 2))
        level = {n: np.fromfile(fh, "<f4", nz * ncol).reshape(nz, ncol) for n in level_fields}
        surf = {n: np.fromfile(fh, "<f4", ncol) for n in surface_fields}
    return ncol, nz, level, surf


def read_wrf(path: Path, level_fields, surface_fields, extra_header=()):
    with open(path, "rb") as fh:
        header = [int(v) for v in np.fromfile(fh, "<i4", 3 + len(extra_header))]
        ncol, nz, nsteps = header[:3]
        dt = float(np.fromfile(fh, "<f4", 1)[0])
        steps = []
        for _ in range(nsteps):
            lv = {n: np.fromfile(fh, "<f4", nz * ncol).reshape(nz, ncol) for n in level_fields}
            for n in ("re_cloud", "re_ice", "re_snow"):
                lv[n] = micron(lv[n])
            sf = {n: np.fromfile(fh, "<f4", ncol) for n in surface_fields}
            steps.append((lv, sf))
    # WSM6's hail flag precedes dt in both the binary header and report.
    return dict(ncol=ncol, nz=nz, nsteps=nsteps,
                **dict(zip(extra_header, header[3:])), dt=dt, steps=steps)


def accumulate(fields, istep, step_pairs, names, keep=None, *,
               nonfinite=False, allow_empty=True, worst_labels=("a", "b")):
    """Common word comparison; the callers choose their existing report schema."""
    for n, (w, g) in step_pairs.items():
        w = np.asarray(w, np.float32)
        g = np.asarray(g, np.float32)
        cols = list(names)
        if keep is not None:
            w, g = w[:, keep], g[:, keep]
            cols = [names[i] for i in np.nonzero(keep)[0]]
        diff_bits = w.view(np.uint32) != g.view(np.uint32)
        u = np.where(np.isnan(w) & np.isnan(g), 0, ulp(w, g))
        gap = np.abs(w.astype(np.float64) - g.astype(np.float64))
        gap = np.where(np.isnan(gap), 0.0, gap)
        initial = {"words": 0, "bit_mismatch": 0, "max_ulp": 0,
                   "max_abs": 0.0, "worst": None, "columns": {}}
        if nonfinite:
            initial.update(nonfinite_gpu=0, nonfinite_wrf=0)
        initial["first_step_with_a_mismatch"] = None
        row = fields.setdefault(n, initial)
        row["words"] += int(w.size)
        row["bit_mismatch"] += int(diff_bits.sum())
        if nonfinite:
            row["nonfinite_gpu"] += int((~np.isfinite(g)).sum())
            row["nonfinite_wrf"] += int((~np.isfinite(w)).sum())
        row["max_abs"] = max(row["max_abs"], float(gap.max()) if gap.size or not allow_empty else 0.0)
        if diff_bits.any() and row["first_step_with_a_mismatch"] is None:
            row["first_step_with_a_mismatch"] = istep + 1
        for i in np.nonzero(diff_bits.any(axis=0))[0]:
            row["columns"][cols[i]] = row["columns"].get(cols[i], 0) + int(diff_bits[:, i].sum())
        m = int(u.max()) if u.size or not allow_empty else 0
        if diff_bits.any() and (row["worst"] is None or m > row["max_ulp"]):
            k, i = np.unravel_index(int(np.argmax(u)), u.shape)
            row["max_ulp"] = max(row["max_ulp"], m)
            row["worst"] = {"step": istep + 1, "level": int(k), "column": cols[i],
                            worst_labels[0]: float(w[k, i]), worst_labels[1]: float(g[k, i])}
