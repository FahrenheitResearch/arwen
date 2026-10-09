"""Grade WOOF's damp_opt=3 damper and w_damp against compiled WRF 4.6.1.

    GPUWM_WRF_EXACT=1 python tools/upper_damping_wrf461_oracle/run_oracle.py \
        --oracle OUT --output receipt.json [--fixture tests/data/upper_damping_wrf461.npz]

``OUT`` is build.py's output (``OUT/O0`` and ``OUT/stock``).  Run once with
GPUWM_WRF_EXACT=1 (the strict build: the 0 ULP standard) and once without
(default arithmetic).  Every output word of every routine is compared.

advance_w (damp_opt=3 lives inside it) runs through the small-step vertical
oracle machinery (gpuwm/verify/smallstep_vertical_oracle.py, with the
strict-mode adapters of tools/smallstep_wrf_exact/compare.py under
GPUWM_WRF_EXACT=1): the native launch is the one the production launcher
``prepare_acoustic_substep_launch`` assembles, and WRF's calc_coef_w,
advance_w and calc_p_rho run with their complete argument lists.  The
fields graded are a, alpha, gamma, w, ph, t_2ave, muts, muave, p and al.

w_damp runs through the production launcher ``dycore.apply_w_damping`` and
the ``w_cfl_stat`` reduction (WRF's max_vert_cfl / max_horiz_cfl).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.upper_damping_wrf461_oracle import columns  # noqa: E402

F32 = np.float32
DT_HRRR = 18.0                       # 3 km, time_step 18 s
DTAU_HRRR = DT_HRRR / 4.0            # time_step_sound 4


def advance_w_cases(edge_zdamp: float):
    base = dict(dx=3000.0, dy=3000.0, specified=False, periodic=True, ztop=20000.0)
    rows = (
        ("hrrr_damp", dict(damp_opt=3, dampcoef=0.2, zdamp=5000.0, dtau=DTAU_HRRR)),
        ("hrrr_control", dict(damp_opt=0, dampcoef=0.2, zdamp=5000.0, dtau=DTAU_HRRR)),
        ("deep_weak", dict(damp_opt=3, dampcoef=0.05, zdamp=8000.0, dtau=2.0)),
        ("shallow_strong", dict(damp_opt=3, dampcoef=1.0, zdamp=2500.0, dtau=1.0)),
        ("whole_column", dict(damp_opt=3, dampcoef=0.2, zdamp=25000.0, dtau=DTAU_HRRR)),
        ("hbot_on_level", dict(damp_opt=3, dampcoef=0.2, zdamp=edge_zdamp, dtau=DTAU_HRRR)),
        ("rigid_lid", dict(damp_opt=3, dampcoef=0.2, zdamp=5000.0, dtau=DTAU_HRRR, top_lid=True)),
        ("dry_loading", dict(damp_opt=3, dampcoef=0.2, zdamp=5000.0, dtau=DTAU_HRRR, moist_cq=False)),
    )
    for name, extra in rows:
        yield name, dict(base, **extra)


def _vertical_module():
    from gpuwm.wrf_exact import ENABLED
    if ENABLED:
        from tools.smallstep_wrf_exact.compare import exact_adapters, exact_workspace_source
        _, module = exact_adapters()
        return module, exact_workspace_source
    from gpuwm.verify import smallstep_vertical_oracle as module
    return module, None


def run_advance_w(libraries, fixture):
    from tools.smallstep_wrf471_oracle import vertical_workspace
    module, workspace = _vertical_module()
    out = {}
    for map_name, map_factors in (("unity_map", False), ("map_factors", True)):
        raw = columns.build_raw(ROOT, map_factors=map_factors)
        for name, value in raw.items():
            fixture[f"raw/{map_name}/{name}"] = value
        edge, edge_level = columns.edge_zdamp(raw)
        for case, metadata in advance_w_cases(edge):
            key = f"{map_name}/{case}"
            record = {"metadata": metadata, "columns": int(raw["MU"].size),
                      "kernel": "advance_w_phi_msf" if map_factors else "advance_w_phi"}
            if metadata["damp_opt"] == 3:
                levels = columns.damping_layer_levels(raw, metadata["zdamp"])
                record["damped_levels_per_column"] = [int(levels.min()), int(levels.max())]
                record["damping_layer_w_words"] = int(levels.sum())
            if case == "hbot_on_level":
                record["hbot_equals_level"] = {"column_j_i": list(columns.EDGE_COLUMN),
                                               "level": edge_level}
            for flavour, library in libraries.items():
                arrays = {}
                ctx = (patch.object(vertical_workspace, "workspace_source", workspace)
                       if workspace else _null())
                with ctx:
                    record[flavour] = module.measure_vertical_case(raw, metadata, str(library),
                                                                   output_arrays=arrays)
                if flavour == "O0":
                    for name in ("w", "ph"):
                        fixture[f"{key}/{name}_wrf"] = arrays[name + "_wrf"]
                    record["_arrays"] = arrays
                else:
                    record["wrf_O0_vs_stock_words_differing"] = int(sum(
                        (arrays[n + "_wrf"].view(np.uint32)
                         != record["_arrays"][n + "_wrf"].view(np.uint32)).sum()
                        for n in ("a", "alpha", "gamma", "w", "ph", "p", "al", "t2")))
            out[key] = record
        # hrrr_damp and hrrr_control differ in damp_opt alone, so the WRF w
        # words that differ between them are the words WRF's damper changed.
        control = out[f"{map_name}/hrrr_control"]["_arrays"]
        rec = out[f"{map_name}/hrrr_damp"]
        rec["wrf_w_words_changed_by_damping_vs_control"] = int(
            (rec["_arrays"]["w_wrf"].view(np.uint32)
             != control["w_wrf"].view(np.uint32)).sum())
    for rec in out.values():
        rec.pop("_arrays", None)
    return out


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ----------------------------------------------------------------- w_damp
W_DAMP_CASES = (("crit1_ieva0", 1.0, 0), ("crit1_ieva1", 1.0, 1),
                ("crit2_ieva0", 2.0, 0), ("crit2_ieva1", 2.0, 1),
                ("crit1p5_ieva1", 1.5, 1))


def _cfl(ww, m, rdnw, dt):
    return np.abs(((ww / m).astype(F32) * rdnw).astype(F32) * F32(dt)).astype(F32)


def w_damp_inputs(raw, dt, seed=2601):
    """Omega from the regime w, scaled so convective cores pass CFL 2."""
    rng = np.random.default_rng(seed)
    nzp1, ny, nx = raw["W"].shape
    nz = nzp1 - 1
    c1f, c2f, rdnw = raw["C1F"], raw["C2F"], raw["RDNW"]
    mut = (raw["MUB"] + raw["MU"]).astype(F32)
    m = (c1f[:, None, None] * mut[None]).astype(F32) + c2f[:, None, None]
    h = columns.column_heights(raw).astype(np.float64)
    dz = np.empty_like(h)
    dz[1:-1] = 0.5 * (h[2:] - h[:-2])
    dz[0], dz[-1] = h[1] - h[0], h[-1] - h[-2]
    w64 = raw["W"].astype(np.float64)
    target = 1.6 * w64 * dt / dz                      # signed Courant number
    scale = np.ones(nzp1)
    scale[1:nz] = 1.0 / (np.abs(rdnw[1:nz].astype(np.float64)) * dt)
    ww = (-target * m.astype(np.float64) * scale[:, None, None]).astype(F32)
    ww[0] = 0.0
    ww[nz] = 0.0
    # Exact onset words: CFL == 1.0, 2.0 and 1.5 to the last bit on edge
    # columns (WRF's test is strict, so these cells must not be damped).
    for i, onset in ((10, 1.0), (11, 2.0), (9, 1.5)):
        found = 0
        for k in range(2, nz - 1):
            j = 7
            guess = F32(onset * m[k, j, i] / (abs(float(rdnw[k])) * dt))
            bits = guess.view(np.int32) + np.arange(-64, 65, dtype=np.int32)
            cand = bits.view(F32)
            hit = cand[_cfl(cand, m[k, j, i], rdnw[k], dt) == F32(onset)]
            if hit.size:
                ww[k, j, i] = hit[0] if k % 2 else -hit[0]
                found += 1
        if found < 3:
            raise RuntimeError(f"only {found} exact onset words for CFL {onset}")
    rw_t = (m * rng.normal(0.0, 0.02, m.shape)).astype(F32)
    rw_t[:, 7, 0] = F32(0.0)
    return dict(ww=ww, w=raw["W"].copy(), mut=mut, mub=raw["MUB"], mup=raw["MU"],
                rw_t=rw_t, u=raw["U"], v=raw["V"], msfu=raw["MAPFAC_U"],
                msfv=raw["MAPFAC_V"], c1f=c1f, c2f=c2f, rdnw=rdnw,
                cfl=_cfl(ww, m, np.concatenate((rdnw, rdnw[-1:]))[:, None, None], dt))


def _wrf_w_damp(library, inp, dt, dx, dy, crit, ieva):
    nzp1, ny, nx = inp["w"].shape
    nz = nzp1 - 1

    def mem3(a):                                    # (k, j, i) -> (i, k, j) on (nx+1, nz+1, ny+1)
        out = np.zeros((nx + 1, nz + 1, ny + 1), dtype=F32, order="F")
        kk, jj, ii = a.shape
        out[:ii, :kk, :jj] = a.transpose(2, 0, 1)
        return out

    def mem2(a):
        out = np.ones((nx + 1, ny + 1), dtype=F32, order="F")
        jj, ii = a.shape
        out[:ii, :jj] = a.T
        return out

    def vec(a):
        out = np.zeros(nz + 1, dtype=F32)
        out[:a.size] = a
        return out

    rw = mem3(inp["rw_t"])
    msfu = mem2(inp["msfu"])
    msfv = mem2(inp["msfv"])
    args = [rw, np.zeros(1, F32), np.zeros(1, F32), mem3(inp["u"]), mem3(inp["v"]),
            mem3(inp["ww"]), mem3(inp["w"]), mem2(inp["mut"]), vec(inp["c1f"]), vec(inp["c2f"]),
            vec(inp["rdnw"]), np.array([1.0 / dx], F32), np.array([1.0 / dy], F32),
            msfu, msfu, msfv, msfv, np.array([dt], F32), np.array([crit], F32),
            np.array([ieva], np.int32), np.array([nx], np.int32), np.array([ny], np.int32),
            np.array([nz], np.int32)]
    fn = ctypes.CDLL(str(library)).oracle_w_damp
    fn.argtypes = [ctypes.c_void_p] * len(args)
    fn.restype = None
    fn(*[ctypes.c_void_p(a.ctypes.data) for a in args])
    return (np.ascontiguousarray(rw[:nx, :nz + 1, :ny].transpose(1, 2, 0)),
            F32(args[1][0]), F32(args[2][0]))


def _woof_w_damp(inp, dt, dx, dy, crit, ieva):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core import dycore
    from gpuwm.core.kernels import get_kernel
    nzp1, ny, nx = inp["w"].shape
    nz = nzp1 - 1
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, ztop=20000.0, dt=dt, run_seconds=dt,
                    w_damping=1, zadvect_implicit=ieva, w_crit_cfl=crit)
    dev = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=F32))
    state = SimpleNamespace(rw_t=dev(inp["rw_t"]), w=dev(inp["w"]), mup=dev(inp["mup"]),
                            mub2d=dev(inp["mub"]), c1f=dev(inp["c1f"]), c2f=dev(inp["c2f"]),
                            rdnw=dev(inp["rdnw"]))
    ww = dev(inp["ww"])
    dycore.apply_w_damping(state, cfg, ww)
    out = cp.zeros(4 + 32, dtype=cp.uint32)
    blocks = ((nz - 1) * ny * nx + 255) // 256
    get_kernel("openbc", "w_cfl_stat")(
        (blocks,), (256,),
        (ww, state.mup, state.mub2d, state.c1f, state.c2f, state.rdnw, dev(inp["u"]),
         dev(inp["v"]), dev(inp["msfu"]), dev(inp["msfv"]), out, F32(dt), F32(1.0 / dx),
         F32(1.0 / dy), F32(dycore.w_damp_onset(cfg)), np.int32(nz), np.int32(ny), np.int32(nx)))
    stats = cp.asnumpy(out)
    return (cp.asnumpy(state.rw_t), stats[0:1].view(F32)[0], stats[3:4].view(F32)[0],
            int(stats[1]))


def run_w_damp(libraries, fixture):
    from gpuwm.verify.smallstep_vertical_oracle import _measure
    out = {}
    dx = dy = 3000.0
    for map_name, map_factors in (("unity_map", False), ("map_factors", True)):
        raw = columns.build_raw(ROOT, map_factors=map_factors)
        for dt in (DT_HRRR, 30.0):
            inp = w_damp_inputs(raw, dt)
            for name in ("ww", "rw_t"):
                fixture[f"w_damp/{map_name}/dt{int(dt)}/{name}"] = inp[name]
            interior = inp["cfl"][1:-1]
            for case, crit, ieva in W_DAMP_CASES:
                key = f"{map_name}/dt{int(dt)}/{case}"
                woof_rw, woof_vmax, woof_hmax, woof_hits = _woof_w_damp(inp, dt, dx, dy, crit, ieva)
                onset = crit if ieva else 1.0
                record = {"columns": int(raw["MU"].size), "dt": dt, "w_crit_cfl": crit,
                          "zadvect_implicit": ieva,
                          "cells_above_onset": int((interior > F32(onset)).sum()),
                          "cells_exactly_at_onset": int((interior == F32(onset)).sum()),
                          "max_input_cfl": float(interior.max())}
                for flavour, library in libraries.items():
                    wrf_rw, vmax, hmax = _wrf_w_damp(library, inp, dt, dx, dy, crit, ieva)
                    record[flavour] = {
                        "rw_tend": _measure(woof_rw, wrf_rw),
                        "max_vert_cfl": _measure(np.array([woof_vmax]), np.array([vmax])),
                        "max_horiz_cfl": _measure(np.array([woof_hmax]), np.array([hmax])),
                        "rw_words_changed_by_wrf": int((wrf_rw.view(np.uint32)
                                                        != inp["rw_t"].view(np.uint32)).sum())}
                    if flavour == "O0":
                        fixture[f"w_damp/{key}/rw_wrf"] = wrf_rw
                        fixture[f"w_damp/{key}/max_cfl_wrf"] = np.array([vmax, hmax], F32)
                record["woof_cells_above_onset"] = woof_hits
                out[key] = record
    return out


#: Harness self-checks in the small-step measurement, not WOOF-vs-WRF
#: outputs: ``ph_unchanged`` compares WOOF's zero-length diagnosis launch
#: (dtau = 0, used only to isolate the EOS) with its own input phi, and the
#: two ``reference_pm1_*`` entries compare WRF with WRF.
SELF_CHECKS = ("ph_unchanged", "reference_pm1_copy", "reference_pm1_history_copy")


def worst(result, *, self_checks=False):
    """Largest max_ulp and total differing words over the graded fields."""
    max_ulp, words = 0, 0
    stack = [(None, result)]
    while stack:
        name, node = stack.pop()
        if isinstance(node, dict):
            if "max_ulp" in node and "differing_words" in node:
                if (name in SELF_CHECKS) == self_checks:
                    max_ulp = max(max_ulp, int(node["max_ulp"]))
                    words += int(node["differing_words"])
            else:
                stack.extend(node.items())
    return max_ulp, words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    from gpuwm.wrf_exact import ENABLED
    libraries = {name: args.oracle / name / "libsmallstep_oracle.so" for name in ("O0", "stock")}
    damp_libraries = {name: args.oracle / name / "libw_damp_oracle.so" for name in ("O0", "stock")}
    fixture = {}
    result = {"mode": "strict GPUWM_WRF_EXACT=1" if ENABLED else "default arithmetic",
              "advance_w": run_advance_w(libraries, fixture),
              "w_damp": run_w_damp(damp_libraries, fixture)}
    summary = {}
    for part in ("advance_w", "w_damp"):
        for flavour in ("O0", "stock"):
            graded = {k: v[flavour] for k, v in result[part].items()}
            summary[f"{part}/{flavour}"] = dict(zip(("max_ulp", "differing_words"), worst(graded)))
            if part == "advance_w":
                summary[f"{part}/{flavour}/harness_self_checks"] = dict(
                    zip(("max_ulp", "differing_words"), worst(graded, self_checks=True)))
    result["summary"] = summary
    import cupy as cp
    result["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    result["nvrtc"] = ".".join(map(str, cp.cuda.nvrtc.getVersion()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.fixture:
        np.savez_compressed(args.fixture, **fixture)
    print(json.dumps({"mode": result["mode"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
