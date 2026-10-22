#!/usr/bin/env python3
"""Classic Thompson (mp_physics=8): WOOF's kernel against WRF v4.6.1's
Fortran on regime columns, every output word, every step.

Two phases, so the Fortran runs on CPU cores and only the WOOF phase needs a
card:

``oracle.py wrf BUILD_DIR OUT_DIR``
    For every set in ``SETS``: forms the float32 inputs exactly as
    ``tools/thompson_real_column_parity/real_column_parity.prepare(mp=8)``
    forms them (``pii`` with the host's powf, the whole geopotential over a
    zero base, ``dz`` and ``hgt`` from it), runs every build of
    ``run_columns_classic_steps`` that ``build.sh`` made (``VARIANTS``:
    unmodified WRF at the oracle flags, at WRF's stock flags, and the
    repaired controls) for the set's step count, and saves inputs and every
    step's outputs to ``OUT_DIR/wrf-<set>.npz``.

``oracle.py woof OUT_DIR [--backend gpu|host] [--json PATH] [--reference V]``
    Runs the production adapter ``gpuwm.core.microphysics._apply_thompson``
    (the classic unit, ``kernels/thompson_classic.cu``) on the same inputs,
    step by step, and compares every output word of every step with the
    reference build's (default ``repaired``: WRF v4.6.1 with its two
    classic-Thompson defects repaired, repair_wrf461.py), and reports every
    build against unmodified WRF beside it, with the footprint check (the
    words where WOOF differs from unmodified WRF must be exactly the words
    the repaired build differs in).  Between steps the harness hands the unit WRF's own
    state back: the species are the unit's in-place state, and ``thp`` is
    set to the unit's microphysics theta (``mp_th``), as mp_gt_driver's
    INTENT(INOUT) ``th`` carries it into its next call.  Surface
    accumulators carry over in the unit's own scratch.  Reflectivity is
    computed every step on both sides (``diagflag``).

Comparison: a word matches when its 32 bits are equal.  The report gives,
per set and field, the mismatching words, the largest total-order ULP
distance (``gpuwm.core.fp32_ulp``) and the largest absolute difference,
plus the worst cell.  Effective radii: WRF writes metres, the unit stores
microns as its last operation ``* 1.0e6f``; WRF's word goes through the same
float32 multiply before the comparison.

The ``nan`` set (columns with one NaN cell) is graded separately: it is not
a forecast state.  Its report also says whether the NaN pattern agrees.

``--backend host`` compiles the same kernel source as C++ on the CPU
(tools/thompson_real_column_parity/host_backend.py), which separates the
transcription from the device toolchain.

``oracle.py fixture OUT_DIR FIXTURE.npz`` cuts the committed GPU gate's
fixture (``FIXTURE_SET`` at ``FIXTURE_STEPS``) from a ``wrf`` run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARITY = ROOT / "tools" / "thompson_real_column_parity"
for _p in (str(ROOT), str(PARITY), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.thompson_aerosol_column_oracle.column_state import (
    ColumnState as OracleColumnState,
)

f32 = np.float32
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")
OUT3 = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "refl",
        "re_cloud", "re_ice", "re_snow")
OUT2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv",
        "sr")
REAL_FIXTURE = ROOT / "tests" / "data" / "thompson_real_columns_wrf461_mp8.npz"

#: (name, column source, dt seconds, steps).  Column sources: ``real`` is the
#: committed 42-column real-data fixture; the others are files
#: make_columns.py writes into OUT_DIR.
SETS = (
    ("real-1step", "real", None, 1),
    ("real-20step", "real", None, 20),
    ("regimes-dt20-30step", "regimes.npz", 20.0, 30),
    ("regimes-dt60-10step", "regimes.npz", 60.0, 10),
    ("regimes-dt6-5step", "regimes.npz", 6.0, 5),
    ("stress-dt20-3step", "stress.npz", 20.0, 3),
    ("nan-dt20-1step", "nan.npz", 20.0, 1),
)


def _columns(source, out_dir):
    if source == "real":
        z = np.load(REAL_FIXTURE)
        cols = {k: np.asarray(z[f"col_{k}"], f32)
                for k in ("p", "th", "geop", "w") + SPECIES}
        cols["labels"] = np.asarray(z["labels"])
        return cols, float(z["dt"])
    z = np.load(Path(out_dir) / source)
    cols = {k: np.asarray(z[k], f32) for k in ("p", "th", "geop", "w")
            + SPECIES}
    cols["labels"] = np.asarray(z["labels"])
    return cols, None


def _prepare(cols):
    import real_column_parity as rcp
    return rcp.prepare(cols, mp=8)


def _read_steps(path, ncol, nz, steps):
    raw = np.fromfile(path, dtype="<f4")
    per = len(OUT3) * ncol * nz + len(OUT2) * ncol
    if raw.size != per * steps:
        raise RuntimeError(f"{path}: {raw.size} words, want {per * steps}")
    out = {}
    for s in range(steps):
        off = s * per
        for name in OUT3:
            n = ncol * nz
            # Fortran buf(ncol, nz): column index fastest.
            out.setdefault(name, []).append(
                raw[off:off + n].reshape(nz, ncol).T.copy())
            off += n
        for name in OUT2:
            out.setdefault(name, []).append(raw[off:off + ncol].copy())
            off += ncol
    return {k: np.stack(v) for k, v in out.items()}


def run_wrf(build, out_dir):
    import real_column_parity as rcp
    build = Path(build)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = build / "run"
    receipt = (build / "BUILD-RECEIPT.txt").read_text()
    for name, source, dt, steps in SETS:
        cols, fixture_dt = _columns(source, out_dir)
        dt = fixture_dt if dt is None else dt
        inp = _prepare(cols)
        ncol, nz = inp["p"].shape
        in_path = out_dir / f"in-{name}.bin"
        rcp.write_wrf_input(in_path, inp, dt, mp=8)
        result = {"dt": np.float32(dt), "steps": np.int32(steps),
                  "labels": cols["labels"], "receipt": np.array(receipt)}
        for key in ("th", "p", "pii", "geop", "w", "dz", "hgt") + SPECIES:
            result[f"in_{key}"] = inp[key]
        for variant in VARIANTS:
            exe = build / variant / "run_columns_classic_steps"
            if not exe.exists():
                continue
            out_path = out_dir / f"out-{name}-{variant}.bin"
            t0 = time.time()
            done = subprocess.run([str(exe), str(in_path), str(out_path),
                                   str(steps)], cwd=str(run_dir),
                                  capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"{exe} failed on {name}:\n"
                                   f"{done.stdout}\n{done.stderr}")
            outs = _read_steps(out_path, ncol, nz, steps)
            for k, v in outs.items():
                result[f"{variant}_{k}"] = v
            out_path.unlink()
            print(f"{name}: {variant} {ncol} columns x {nz} levels x "
                  f"{steps} steps, dt {dt} s, {time.time() - t0:.1f} s")
        in_path.unlink()
        np.savez_compressed(out_dir / f"wrf-{name}.npz", **result)


class _State(OracleColumnState):
    _scratch_error = " reused"

    def __init__(self, z, xp):
        self._xp = xp
        cols = {name: np.asarray(z[f"in_{name}"], f32)
                for name in ("p", "th", "geop", "w", *SPECIES)}
        super().__init__(cols, xp, SPECIES, copy_inputs=True, host_zeros=True)


def _host(a):
    a = a.get() if hasattr(a, "get") else np.asarray(a)
    return np.asarray(a, dtype=f32)


def _cols(a):
    a = _host(a)
    return np.ascontiguousarray(a.reshape(a.shape[0], -1).T)


def run_woof(z, xp):
    """Every step's outputs of the classic unit, keyed as WRF's."""
    from gpuwm.core.microphysics import _apply_thompson
    from gpuwm.core.refl import consume_refl_10cm
    state = _State(z, xp)
    dt = float(z["dt"])
    steps = int(z["steps"])
    cfg = SimpleNamespace(mp_physics=8, no_mp_heating=0, mp_tend_lim=10.0,
                          thompson_version="wrf_461",
                          thompson_fork_snow_fall="blend")
    out = {k: [] for k in OUT3 + OUT2}
    for _ in range(steps):
        diag = _apply_thompson(state, cfg, dt, refl_10cm_due=True)
        if hasattr(xp, "cuda"):
            xp.cuda.get_current_stream().synchronize()
        for s in SPECIES:
            out[s].append(_cols(getattr(state, s)))
        th = state._scratch["mp_th"]
        out["th"].append(_cols(th))
        # The output frame's consume, as the forecast's history write does
        # (a second unconsumed stash is refused by refl.stash_refl_10cm).
        out["refl"].append(_cols(consume_refl_10cm(state)))
        out["re_cloud"].append(_cols(state.effc))
        out["re_ice"].append(_cols(state.effi))
        out["re_snow"].append(_cols(state.effs))
        for name in OUT2:
            out[name].append(_host(getattr(diag, name)).ravel().copy())
        # mp_gt_driver's th is INTENT(INOUT): its next call starts from
        # the theta this one wrote.
        state.thp[...] = th
    return {k: np.stack(v) for k, v in out.items()}


def compare(have, want, labels):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    fields = {}
    for name in OUT3 + OUT2:
        w = np.asarray(want[name], f32)
        if name.startswith("re_"):
            w = (w * f32(1.0e6)).astype(f32)
        h = np.asarray(have[name], f32).reshape(w.shape)
        hb, wb = h.view(np.uint32), w.view(np.uint32)
        diff = hb != wb
        dist = fp32_ulp_distance(h, w)
        both_nan = np.isnan(h) & np.isnan(w)
        finite = np.isfinite(h) & np.isfinite(w)
        absd = np.where(finite, np.abs(h.astype(np.float64)
                                       - w.astype(np.float64)), 0.0)
        e = {"words": int(w.size), "mismatches": int(diff.sum()),
             "max_ulp": int(dist[~both_nan].max()) if (~both_nan).any()
             else 0,
             "max_abs": float(absd.max()),
             "nan_pattern_equal": bool((np.isnan(h) == np.isnan(w)).all()),
             "nan_words": int(np.isnan(w).sum())}
        if diff.any():
            idx = np.unravel_index(int(np.argmax(np.where(diff, dist, -1))),
                                   w.shape)
            col = int(idx[1])
            e["worst"] = {"step": int(idx[0]) + 1, "column": col,
                          "label": str(labels[col]),
                          "level": int(idx[2]) if w.ndim == 3 else None,
                          "woof": float(h[idx]), "wrf": float(w[idx]),
                          "woof_bits": f"0x{int(hb[idx]):08x}",
                          "wrf_bits": f"0x{int(wb[idx]):08x}"}
            cols = np.unique(np.nonzero(diff)[1])
            e["columns"] = [str(labels[c]) for c in cols[:40]]
            first = np.nonzero(diff.reshape(diff.shape[0], -1).any(1))[0]
            e["first_step"] = int(first[0]) + 1
        fields[name] = e
    return fields


def _environment(backend, xp):
    env = {"backend": backend, "python": platform.python_version(),
           "numpy": np.__version__, "host": platform.node(),
           "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", "")}
    if backend == "gpu":
        import cupy
        dev = cupy.cuda.Device()
        env["cupy"] = cupy.__version__
        env["device"] = cupy.cuda.runtime.getDeviceProperties(
            dev.id)["name"].decode()
        env["cc"] = dev.compute_capability
        env["nvrtc"] = ".".join(map(str, cupy.cuda.nvrtc.getVersion()))
        env["driver"] = cupy.cuda.runtime.driverGetVersion()
        from gpuwm import wrf_exact
        rec = wrf_exact.compile_receipt()
        env["strict_options"] = rec["strict_options"] if rec["enabled"] \
            else None
        env["strict_compiles"] = sum(1 for r in rec["records"]
                                     if r["kind"] == "nvrtc_compile")
    from gpuwm.core.kernels import module_source, module_options
    import hashlib
    src = module_source("thompson_classic")
    env["thompson_classic_source_sha256"] = hashlib.sha256(
        src.encode()).hexdigest()
    env["thompson_classic_options"] = list(module_options("thompson_classic"))
    return env


#: The WRF builds ``oracle.py wrf`` runs (build.sh's variants).  ``oracle``
#: is unmodified WRF v4.6.1 at the oracle flags; ``repaired`` the same with
#: the two v4.6.1 defects repaired (repair_wrf461.py), ``repaired_rcg`` and
#: ``repaired_lqg`` with one each, for attribution; ``stock`` unmodified WRF
#: at WRF's own GNU flags.
VARIANTS = ("oracle", "stock", "repaired", "repaired_rcg", "repaired_lqg")


def _variant(z, variant):
    return {k: z[f"{variant}_{k}"] for k in OUT3 + OUT2}


def _microns(out):
    return {k: (v * f32(1.0e6)).astype(f32) if k.startswith("re_") else v
            for k, v in out.items()}


def _diff_mask(have, want):
    """Per field, the words whose 32 bits differ (WRF radii in microns)."""
    masks = {}
    for name in OUT3 + OUT2:
        w = np.asarray(want[name], f32)
        if name.startswith("re_"):
            w = (w * f32(1.0e6)).astype(f32)
        h = np.asarray(have[name], f32).reshape(w.shape)
        masks[name] = h.view(np.uint32) != w.view(np.uint32)
    return masks


def _summary(fields):
    return {"words": sum(e["words"] for e in fields.values()),
            "mismatches": sum(e["mismatches"] for e in fields.values()),
            "max_ulp": max(e["max_ulp"] for e in fields.values()),
            "max_abs": max(e["max_abs"] for e in fields.values())}


def run_compare(out_dir, backend="gpu", sets=None, json_path=None,
                reference="repaired"):
    """Grade WOOF against ``reference``; report every other variant beside.

    The footprint check: the words where WOOF differs from unmodified WRF
    must be exactly the words where the repaired WRF differs from unmodified
    WRF, field by field and step by step, so every difference from
    unmodified WRF is one the two documented defects make.
    """
    if backend == "host":
        import host_backend
        host_backend.install()
    import cupy as xp
    out_dir = Path(out_dir)
    report = {"reference": reference, "sets": {}}
    for name, _source, _dt, _steps in SETS:
        if sets and name not in sets:
            continue
        z = np.load(out_dir / f"wrf-{name}.npz")
        t0 = time.time()
        have = run_woof(z, xp)
        labels = z["labels"]
        ref = _variant(z, reference)
        fields = compare(have, ref, labels)
        entry = {"columns": int(z["in_p"].shape[0]),
                 "levels": int(z["in_p"].shape[1]),
                 "steps": int(z["steps"]), "dt": float(z["dt"]),
                 "seconds": round(time.time() - t0, 2)}
        entry.update(_summary(fields))
        entry["fields"] = fields
        # Everything else, each against unmodified WRF.
        pristine = _variant(z, "oracle")
        beside = {"woof": compare(have, pristine, labels)}
        for v in VARIANTS[1:]:
            if f"{v}_qv" in z.files:
                beside[v] = compare(_microns(_variant(z, v)), pristine,
                                    labels)
        entry["vs_unmodified_wrf"] = {
            k: dict(_summary(f), fields={
                n: {kk: e[kk] for kk in ("mismatches", "max_ulp", "max_abs")}
                for n, e in f.items() if e["mismatches"]})
            for k, f in beside.items()}
        if f"{reference}_qv" in z.files and reference != "oracle":
            woof_m = _diff_mask(have, pristine)
            ref_m = _diff_mask(_microns(ref), pristine)
            same = all(np.array_equal(woof_m[n], ref_m[n]) for n in woof_m)
            entry["footprint_equal"] = bool(same)
            entry["footprint_columns"] = sorted({
                str(labels[c]) for n in ref_m
                for c in np.unique(np.nonzero(ref_m[n])[1])})
        report["sets"][name] = entry
        vs = entry["vs_unmodified_wrf"]
        print(f"{name}: {entry['columns']} cols x {entry['levels']} lev x "
              f"{entry['steps']} steps, dt {entry['dt']:g}: vs {reference} "
              f"WRF {entry['mismatches']} of {entry['words']} words differ, "
              f"max {entry['max_ulp']} ULP | vs unmodified WRF: woof "
              f"{vs['woof']['mismatches']}, "
              + ", ".join(f"{k} {v['mismatches']}" for k, v in vs.items()
                          if k != "woof")
              + (f" | footprint_equal={entry.get('footprint_equal')}"
                 if "footprint_equal" in entry else ""), flush=True)
        for fname, e in fields.items():
            if e["mismatches"]:
                print(f"   {fname}: {e['mismatches']} words, max "
                      f"{e['max_ulp']} ULP, first step {e['first_step']}, "
                      f"worst {e['worst']}", flush=True)
    graded = [n for n in report["sets"] if not n.startswith("nan")]
    report["overall"] = {
        "graded_sets": graded,
        "words": sum(report["sets"][n]["words"] for n in graded),
        "mismatches": sum(report["sets"][n]["mismatches"] for n in graded),
        "max_ulp": max([report["sets"][n]["max_ulp"] for n in graded],
                       default=0),
        "max_abs": max([report["sets"][n]["max_abs"] for n in graded],
                       default=0.0),
        "footprint_equal": all(report["sets"][n].get("footprint_equal", True)
                               for n in graded),
    }
    report["overall"]["bitwise"] = report["overall"]["mismatches"] == 0
    report["environment"] = _environment(backend, xp)
    o = report["overall"]
    print(f"OVERALL graded vs {reference} WRF: {o['mismatches']} of "
          f"{o['words']} words differ, max {o['max_ulp']} ULP, max abs "
          f"{o['max_abs']:.3e}, bitwise={o['bitwise']}, footprint_equal="
          f"{o['footprint_equal']}")
    if json_path:
        Path(json_path).write_text(json.dumps(report, indent=1))
    return report


#: The committed GPU gate's fixture (tests/test_thompson_mp8_regime_columns_
#: gpu.py): the regime set at dt 20 s over 30 steps, with the repaired and
#: the unmodified WRF answers at steps 1 and 30.
FIXTURE_SET = "regimes-dt20-30step"
FIXTURE_STEPS = (1, 30)


def cut_fixture(out_dir, path):
    z = np.load(Path(out_dir) / f"wrf-{FIXTURE_SET}.npz")
    keep = [s - 1 for s in FIXTURE_STEPS]
    arrays = {k: z[k] for k in z.files if k.startswith("in_")}
    arrays.update(dt=z["dt"], steps=z["steps"], labels=z["labels"],
                  receipt=z["receipt"],
                  kept_steps=np.asarray(FIXTURE_STEPS, np.int32))
    for variant in ("repaired", "oracle"):
        for k in OUT3 + OUT2:
            arrays[f"{variant}_{k}"] = z[f"{variant}_{k}"][keep]
    np.savez_compressed(path, **arrays)
    print(f"{path}: {FIXTURE_SET}, steps {FIXTURE_STEPS}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("wrf")
    a.add_argument("build", type=Path)
    a.add_argument("out", type=Path)
    b = sub.add_parser("woof")
    b.add_argument("out", type=Path)
    b.add_argument("--backend", choices=("gpu", "host"), default="gpu")
    b.add_argument("--sets", nargs="*")
    b.add_argument("--json", type=Path)
    b.add_argument("--reference", choices=VARIANTS, default="repaired",
                   help="the WRF build graded against (default: repaired)")
    c = sub.add_parser("fixture")
    c.add_argument("out", type=Path)
    c.add_argument("path", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "fixture":
        cut_fixture(args.out, args.path)
        return 0
    if args.cmd == "wrf":
        run_wrf(args.build, args.out)
        return 0
    report = run_compare(args.out, args.backend, args.sets, args.json,
                         args.reference)
    return 0 if report["overall"]["bitwise"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
