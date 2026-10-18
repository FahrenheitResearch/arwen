"""Measure the gwd_opt = 1 (KIM orographic drag) kernel against WRF v4.6.1.

The fixture is written by ``tools/gwdo_kim_wrf461_oracle`` (build.sh drives
``run_gwdo_kim.F90``, which calls SUBROUTINE gwdo of the byte-unmodified
``phys/module_bl_gwdo.F`` and, through it, ``bl_gwdo_run`` of
``phys/physics_mmm/bl_gwdo.F90``).  It lives under
``tests/data/oracles/gwdo_kim/``, one pair per vertical grid G (L50, L90):

``gwdo-kim-inputs-G.npz``
    float32 ``u, v, t, qv, p, pi, z, dz, rublten0, rvblten0`` of shape
    ``(nz, ncol)``, ``p_interface (nz + 1, ncol)``, ``var, oc1, sina, cosa
    (ncol,)``, ``oa, ol (4, ncol)``, the ``dx`` and ``dt`` of every case and
    the ``regime`` name of every column.
``gwdo-kim-wrf461-G.npz``
    WRF's words after one gwdo call per (dx, dt) case: ``rublten, rvblten,
    dtaux3d, dtauy3d`` of shape ``(ncase, nz, ncol)`` and ``dusfcg, dvsfcg``
    of shape ``(ncase, ncol)``.

:func:`measure` runs :func:`gpuwm.core.terrain_drag.launch_gwdo` (the
production launcher and its production compile route) on the same inputs
and returns, per output field, the number of words that are not
bit-identical, the FP32 ULP maximum and the largest absolute difference.

:func:`sector_probe` and :func:`atanf_probe` evaluate the kernel's own
``td_atan2f`` wind-sector statements and ``td_atanf`` on the device, for
``tools/gwdo_kim_wrf461_oracle/sector_sweep.py`` and the stored sample of
the reference platform's atan2f words (``sector-sample.npz``).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "gwdo_kim")
GRIDS = ("L50", "L90")
LEVEL_INPUTS = ("u", "v", "t", "qv", "p", "pi", "z", "dz", "rublten0",
                "rvblten0")
COLUMN_INPUTS = ("var", "oc1", "sina", "cosa")
LEVEL_FIELDS = ("rublten", "rvblten", "dtaux3d", "dtauy3d")
SURFACE_FIELDS = ("dusfcg", "dvsfcg")
OUTPUT_FIELDS = LEVEL_FIELDS + SURFACE_FIELDS


def read_inputs_bin(path) -> dict[str, np.ndarray]:
    """make_columns.py's stream file -> the input arrays."""
    raw = Path(path).read_bytes()
    ncol, nz, ncase = (int(x) for x in np.frombuffer(raw[:12], "<i4"))
    off = 12

    def take(count, shape=None):
        nonlocal off
        a = np.frombuffer(raw, "<f4", count, off).copy()
        off += 4 * count
        return a if shape is None else a.reshape(shape)

    out = {"dx": take(ncase), "dt": take(ncase)}
    for name in LEVEL_INPUTS:
        out[name] = take(nz * ncol, (nz, ncol))
    out["p_interface"] = take((nz + 1) * ncol, (nz + 1, ncol))
    for name in COLUMN_INPUTS:
        out[name] = take(ncol)
    out["oa"] = take(4 * ncol, (4, ncol))
    out["ol"] = take(4 * ncol, (4, ncol))
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return out


def read_outputs_bin(path, nz: int, ncol: int, ncase: int):
    """run_gwdo_kim's output stream -> ``{field: (ncase, ...)}``."""
    raw = np.fromfile(path, "<f4")
    per_case = len(LEVEL_FIELDS) * nz * ncol + len(SURFACE_FIELDS) * ncol
    if raw.size != per_case * ncase:
        raise ValueError(f"{path}: {raw.size} words, expected "
                         f"{per_case * ncase}")
    raw = raw.reshape(ncase, per_case)
    out, off = {}, 0
    for name in LEVEL_FIELDS:
        out[name] = raw[:, off:off + nz * ncol].reshape(ncase, nz, ncol)
        off += nz * ncol
    for name in SURFACE_FIELDS:
        out[name] = raw[:, off:off + ncol]
        off += ncol
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


def load_fixture(grid: str, directory=FIXTURE_DIR):
    directory = Path(directory)
    with np.load(directory / f"gwdo-kim-inputs-{grid}.npz") as f:
        inputs = {k: f[k] for k in f.files}
    with np.load(directory / f"gwdo-kim-wrf461-{grid}.npz") as f:
        wrf = {k: f[k] for k in f.files}
    return inputs, wrf


def run_gpu(inputs) -> dict[str, np.ndarray]:
    """``launch_gwdo`` once per (dx, dt) case, laid out as the fixture."""
    import cupy as cp
    from gpuwm.core.terrain_drag import launch_gwdo

    nz, ncol = inputs["u"].shape

    def lev(a):
        return cp.asarray(np.ascontiguousarray(a, np.float32)
                          .reshape(a.shape[0], 1, ncol))

    def row(a):
        return np.ascontiguousarray(a, np.float32).reshape(1, ncol)

    atm = {"u": lev(inputs["u"]), "v": lev(inputs["v"]),
           "temperature": lev(inputs["t"]), "qv": lev(inputs["qv"]),
           "pressure": lev(inputs["p"]), "exner": lev(inputs["pi"]),
           "p_interface": lev(inputs["p_interface"]), "z": lev(inputs["z"]),
           "dz": lev(inputs["dz"])}
    oa = [row(inputs["oa"][m]) for m in range(4)]
    ol = [row(inputs["ol"][m]) for m in range(4)]
    out = {k: [] for k in OUTPUT_FIELDS}
    for dx, dt in zip(inputs["dx"], inputs["dt"]):
        du = lev(inputs["rublten0"])
        dv = lev(inputs["rvblten0"])
        diag = {}
        launch_gwdo(atm, du, dv, var=row(inputs["var"]),
                    con=row(inputs["oc1"]), oa=oa, ol=ol,
                    sina=row(inputs["sina"]), cosa=row(inputs["cosa"]),
                    dx=float(dx), dt=float(dt), diagnostics=diag)
        out["rublten"].append(cp.asnumpy(du)[:, 0, :])
        out["rvblten"].append(cp.asnumpy(dv)[:, 0, :])
        out["dtaux3d"].append(cp.asnumpy(diag["DTAUX3D"])[:, 0, :])
        out["dtauy3d"].append(cp.asnumpy(diag["DTAUY3D"])[:, 0, :])
        out["dusfcg"].append(cp.asnumpy(diag["DUSFCG"])[0])
        out["dvsfcg"].append(cp.asnumpy(diag["DVSFCG"])[0])
    return {k: np.ascontiguousarray(np.stack(v)) for k, v in out.items()}


def _field_row(g, w, regime):
    g = np.asarray(g, np.float32)
    w = np.asarray(w, np.float32)
    miss = g.view(np.uint32) != w.view(np.uint32)
    ulp = np.asarray(fp32_ulp_distance(g, w), np.int64)
    diff = np.abs(g.astype(np.float64) - w.astype(np.float64))
    diff = np.where(np.isnan(diff), np.inf, diff)
    by_regime = {}
    col_miss = miss.reshape(-1, miss.shape[-1]).any(axis=0)
    for name in sorted(set(map(str, regime))):
        sel = np.asarray(regime) == name
        by_regime[name] = int(col_miss[sel].sum())
    return {"words": int(g.size),
            "not_bit_identical": int(miss.sum()),
            "max_ulp": int(ulp.max()) if ulp.size else 0,
            "max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "columns_missed_by_regime": by_regime}


def compare(got, wrf, regime) -> dict:
    return {name: _field_row(got[name], wrf[name], regime)
            for name in OUTPUT_FIELDS}


def measure(grid: str, directory=FIXTURE_DIR) -> dict:
    inputs, wrf = load_fixture(grid, directory)
    return compare(run_gpu(inputs), wrf, inputs["regime"])


def activity(inputs, wrf) -> dict:
    """Which columns WRF actually dragged, per regime and case.

    A column is active in a case when WRF changed any of its RUBLTEN words.
    The grade covers inactive columns too (their words must stay the PBL
    tendencies, bit for bit); this only shows the drag is not vacuous.
    """
    regime = np.asarray(inputs["regime"]).astype(str)
    moved = (wrf["rublten"].view(np.uint32)
             != inputs["rublten0"][None].view(np.uint32)).any(axis=1)
    out = {}
    for name in sorted(set(regime)):
        sel = regime == name
        out[name] = {"columns": int(sel.sum()),
                     "active_per_case": [int(x) for x in moved[:, sel].sum(1)]}
    return out


#: The statements of kernels/terrain_drag.cu the sector probe repeats.
SECTOR_KERNEL_LINES = (
    "const real fdir = (real)mdir / (2.0f * pi_);",
    "real wdir = td_atan2f(ubar, vbar) + pi_;",
    "int idir = td_nint(fdir * wdir) % mdir + 1;",
)
#: module_pbl_driver.F's literal PI=3.141592653, as float32.
PBL_DRIVER_PI = np.float32(3.141592653)
SECTOR_PROBE = r"""
extern "C" __global__
void gwdo_sector_probe(const real *ubar_in, const real *vbar_in, real *a_out,
                       int *idir_out, real pi_, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int mdir = 8;
    real ubar = ubar_in[i], vbar = vbar_in[i];
    real a = td_atan2f(ubar, vbar);
    a_out[i] = a;
    if (!isfinite(a)) { idir_out[i] = 0; return; }
    const real fdir = (real)mdir / (2.0f * pi_);
    real wdir = td_atan2f(ubar, vbar) + pi_;
    int idir = td_nint(fdir * wdir) % mdir + 1;
    idir_out[i] = idir;
}

extern "C" __global__
void gwdo_atanf_probe(unsigned int start, real *out, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = td_atanf(__uint_as_float(start + (unsigned int)i));
}
"""


def sector_probe_source(source: str | None = None) -> str:
    """The production terrain-drag source plus the two probe kernels."""
    from gpuwm.core import terrain_drag

    source = terrain_drag.module_source() if source is None else source
    for line in SECTOR_KERNEL_LINES:
        if line not in source:
            raise RuntimeError(f"kernels/terrain_drag.cu no longer says "
                               f"{line!r}; the sector probe must follow it")
    return source + SECTOR_PROBE


@lru_cache(maxsize=None)
def _probe_module(source: str, device: int):
    import cupy as cp
    from gpuwm import nvrtc_ptx_cache as compiler
    from gpuwm.core import terrain_drag

    with cp.cuda.Device(device):
        ptx, _ = compiler.compile_using_nvrtc(
            source, terrain_drag.MODULE_OPTIONS, None,
            "terrain_drag_sector_probe.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    return module


def _probe(source=None):
    import cupy as cp
    return _probe_module(sector_probe_source(source), cp.cuda.Device().id)


def sector_probe(pairs, source=None):
    """(td_atan2f words, sectors) for float32 (ubar, vbar) rows."""
    import cupy as cp

    pairs = np.ascontiguousarray(pairs, np.float32)
    n = pairs.shape[0]
    u = cp.asarray(np.ascontiguousarray(pairs[:, 0]))
    v = cp.asarray(np.ascontiguousarray(pairs[:, 1]))
    a = cp.empty(n, cp.float32)
    s = cp.empty(n, cp.int32)
    _probe(source).get_function("gwdo_sector_probe")(
        ((n + 255) // 256,), (256,), (u, v, a, s, PBL_DRIVER_PI, np.int64(n)))
    return cp.asnumpy(a), cp.asnumpy(s)


def atanf_probe(start: int, n: int, source=None):
    """td_atanf on the float32 bit patterns start .. start + n - 1."""
    import cupy as cp

    out = cp.empty(n, cp.float32)
    _probe(source).get_function("gwdo_atanf_probe")(
        ((n + 255) // 256,), (256,), (np.uint32(start), out, np.int64(n)))
    return cp.asnumpy(out)


def load_sector_sample(directory=FIXTURE_DIR):
    with np.load(Path(directory) / "sector-sample.npz") as f:
        return {k: f[k] for k in f.files}


#: Branch controls: (name, statement in gwdo_column, replacement, what the
#: fixture must reach for the replacement to move a WRF word).  Each one is a
#: one-statement change to the production kernel; the oracle catching it is
#: the evidence that the columns reach that branch, measured on the real
#: kernel rather than on a mirror of it.
BRANCH_MUTATIONS = (
    ("wind_sector_atan2",
     "else z = td_atanf(fabsf(y / x));",
     "else z = (float)atan((double)fabsf(y / x));",
     "a sector boundary decided by atan2f's last bit"),
    ("flow_blocking",
     "real taufb = 0.5f * rhobar * coefm / (dxmeter * dxmeter) * fbdcd",
     "real taufb = 0.51f * rhobar * coefm / (dxmeter * dxmeter) * fbdcd",
     "flow-blocking stress (kblk > 0)"),
    ("wave_breaking",
     "real temc = 2.0f + 1.0f / tem2;",
     "real temc = 2.01f + 1.0f / tem2;",
     "saturation hypothesis (rim <= ric)"),
    ("no_breaking_carry",
     "taup[kp1 - 1] = taup[k - 1];",
     "taup[kp1 - 1] = 0.99f * taup[k - 1];",
     "stress carried up unbroken (rim > ric)"),
    ("dtfac_limiter",
     "dtfac = td_min(dtfac, fabsf(velco_at(k) / (deltim * td)));",
     "dtfac = dtfac;",
     "the critical-line time-step limiter (dtfac < 1)"),
    ("velco_clamp",
     "if (vc < veleps && vc > 0.0f) vc = veleps;",
     "if (vc < veleps && vc > 0.0f) vc = vc;",
     "velco in (0, veleps) clamped to veleps"),
    ("frmax_cap",
     "fr = td_min(fr, frmax);",
     "fr = fr;",
     "Froude number capped at frmax"),
    ("odmax_clamp",
     "od = td_min(od, odmax);",
     "od = od;",
     "orographic aspect ratio capped at odmax"),
    ("odmin_clamp",
     "od = td_max(od, odmin);",
     "od = od;",
     "orographic aspect ratio floored at odmin"),
    ("efmax_clamp",
     "efact = td_min(td_max(efact, efmin), efmax);",
     "efact = td_max(efact, efmin);",
     "enhancement factor capped at efmax"),
    ("critical_level",
     "icrilv = icrilv || (usqj < ric) || (velco_at(k) <= 0.0f);",
     "icrilv = icrilv || (usqj < ric);",
     "a critical level aloft (velco <= 0 above kbl)"),
    ("low_level_reversal",
     "if (k < kbl) ldrag = ldrag || velco_at(k) <= 0.0f;",
     "if (k < kbl) ldrag = ldrag;",
     "reversed flow inside the low layer switches the drag off"),
    ("kbl_floor",
     "int kbl = klowtop;",
     "int kbl = klowtop > 0 ? klowtop : kte;",
     "2*var above the model top (klowtop = 0, kbl = kpblmin)"),
    ("light_wind",
     "ldrag = ldrag || ulow == 1.0f;",
     "ldrag = ldrag;",
     "ulow clamped to 1 m/s switches the drag off"),
    ("zero_variance",
     "ldrag = ldrag || varc <= 0.0f;",
     "ldrag = ldrag;",
     "VAR = 0 switches the drag off"),
    ("subgrid_rho_weights",
     "rhobar = rhobar + rdelks * RHO(k);",
     "rhobar = rhobar + rdelks * RHO(k) * 1.0001f;",
     "low-layer mean density in the base stress"),
)


def mutated_source(name: str) -> str:
    """The production terrain-drag source with one gwdo statement changed.

    The statement must occur exactly once inside gwdo_column (and, for the
    atan2 control, once inside td_atan2f).
    """
    from gpuwm.core import terrain_drag

    row = {m[0]: m for m in BRANCH_MUTATIONS}[name]
    _, old, new, _ = row
    source = terrain_drag.module_source()
    if name == "wind_sector_atan2":
        a = source.index("__device__ float td_atan2f(")
        b = source.index("\n}\n", a)
    else:
        a = source.index("void gwdo_column(")
        b = source.index("void gwdo_gsl_column(")
    region = source[a:b]
    if region.count(old) != 1:
        raise RuntimeError(f"{name}: {old!r} occurs {region.count(old)} "
                           "times in its function")
    return source[:a] + region.replace(old, new) + source[b:]


class kernel_source:
    """Context manager: compile the terrain-drag unit from ``source``.

    Keeps the trial compile out of the process's kernel manifest and puts
    the production unit back on exit.
    """

    def __init__(self, source: str):
        self.source = source

    def __enter__(self):
        from importlib import import_module
        from gpuwm.core import terrain_drag

        self.td = terrain_drag
        self.km = import_module("gpuwm.certify.kernel_manifest")
        self.saved = (terrain_drag.module_source, self.km._MANIFEST)
        terrain_drag._module.cache_clear()
        self.km._MANIFEST = {}
        terrain_drag.module_source = lambda: self.source
        return self

    def __exit__(self, *exc):
        self.td.module_source, self.km._MANIFEST = self.saved
        self.td._module.cache_clear()
        return False


def measure_mutation(name: str) -> dict:
    """Words each grid's oracle misses with one gwdo statement changed."""
    with kernel_source(mutated_source(name)):
        out = {}
        for grid in GRIDS:
            table = measure(grid)
            out[grid] = sum(v["not_bit_identical"] for v in table.values())
    return out
