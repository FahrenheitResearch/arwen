"""Run gpuwm's CUDA WSM3 over the oracle column set and grade it bit for bit.

Reads the SAME ``columns.bin`` the WRF Fortran read (make_columns.py), runs
the production launcher (``gpuwm.core.wsm3.launch_wsm3``) step by step
exactly as WRF's wrapper runs (microphysics, then the radiation effective
radii), and compares every output word.

Two WRF references. O0 is unmodified WRF v4.6.1. O0fix has the one
surface-accounting edit D1. Strict arithmetic must equal O0 on every word
at every step, with no excluded columns. Default arithmetic differences
against both references are counted, including the default upper-cell
remap correction D3. O0fix is diagnostic evidence, not the strict referee.

Effective radii: WRF stores metres; gpuwm's radiation contract is microns,
formed as one float32 multiply ``re * 1.0e6`` -- the multiply WRF's RRTMG
does when it reads re_cloud/re_ice/re_snow.  WRF's words are put through
that multiply before comparison.

Subcommands::

    python run_gpu.py grade ORACLE_DIR [--flavour O0|O0fix|O2vec] [--out JSON]
        every word of every step against run_wsm3_columns.F90's output
    python run_gpu.py make-fixture ORACLE_DIR OUT_DIR       (CPU only)
        the committed fixture: the column set, a SHA-256 of every WRF
        output field at every step for both references and all field words
        at every step (the compact fixture is checked against those hashes)
    python run_gpu.py check-fixture FIXTURE_DIR [--out JSON] [--tier N]
        [--exclude COLUMN ...] [--plant]
        grade against the committed fixture (what the pytest gate runs);
        --plant flips the last bit of every stored theta word in the
        compiled kernel source first, which the check must catch

GPU work.  On a shared box run only through its GPU mutex.
GPUWM_WRF_EXACT=1 selects the strict build (no FMA, no FTZ); unset is the
default build.  Every report states which.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

LEVEL_OUT = ("th", "qv", "qc", "qr", "re_cloud", "re_ice", "re_snow", "t")
SURFACE_OUT = ("rain", "rainncv", "snow", "snowncv", "sr")
#: The only words D1 may move: the rain total, this call's rain and the
#: frozen fraction formed from it.
D1_FIELDS = ("rain", "rainncv", "sr")
#: (dt, nsteps): dt 30 and 72 s are one minor loop; 150 s one loop with
#: dtcld > dtcldcr; 180 s nint(1.5) = 2 loops; 300 s nint(2.5) = 3 loops.
RUNS = ((30.0, 12), (72.0, 8), (150.0, 3), (180.0, 3), (300.0, 4))
#: The planted fault: theta's store, last bit flipped (kernels/wsm3.cu).
PLANT_FROM = "        theta[id] = th;\n"
PLANT_TO = "        theta[id] = __uint_as_float(__float_as_uint(th) ^ 1u);\n"


def tag_of(dt: float, nsteps: int) -> str:
    return f"dt{int(dt)}-n{nsteps}"


def read_columns(path: Path):
    with open(path, "rb") as fh:
        ncol, nz = (int(v) for v in np.fromfile(fh, "<i4", 2))
        names = ("th", "qv", "qc", "qr", "den", "pii", "p", "dz")
        level = {n: np.fromfile(fh, "<f4", nz * ncol).reshape(nz, ncol) for n in names}
        level["w"] = np.fromfile(fh, "<f4", (nz + 1) * ncol).reshape(nz + 1, ncol)
        surf = {n: np.fromfile(fh, "<f4", ncol) for n in ("rain", "snow")}
    return ncol, nz, level, surf


def micron(re_metres):
    return (np.asarray(re_metres, np.float32) * np.float32(1.0e6)).astype(np.float32)


def read_wrf(path: Path):
    """run_wsm3_columns.F90's output, effective radii already in microns."""
    with open(path, "rb") as fh:
        ncol, nz, nsteps = (int(v) for v in np.fromfile(fh, "<i4", 3))
        dt = float(np.fromfile(fh, "<f4", 1)[0])
        steps = []
        for _ in range(nsteps):
            lv = {n: np.fromfile(fh, "<f4", nz * ncol).reshape(nz, ncol) for n in LEVEL_OUT}
            for n in ("re_cloud", "re_ice", "re_snow"):
                lv[n] = micron(lv[n])
            sf = {n: np.fromfile(fh, "<f4", ncol) for n in SURFACE_OUT}
            steps.append((lv, sf))
    return dict(ncol=ncol, nz=nz, nsteps=nsteps, dt=dt, steps=steps)


def run_gpu(columns: Path, dt: float, nsteps: int):
    """The production launcher, step by step, as WRF's wrapper runs."""
    import cupy as cp
    from gpuwm.core.wsm3 import launch_wsm3

    ncol, nz, level, surf = read_columns(columns)
    f32 = np.float32
    dev = {n: cp.asarray(v.reshape(v.shape[0], 1, ncol), dtype=cp.float32)
           for n, v in level.items()}
    w = cp.ascontiguousarray(dev["w"][:nz])
    s = {n: cp.zeros((1, ncol), dtype=cp.float32) for n in
         ("rainnc", "rainncv", "snownc", "snowncv", "sr")}
    s["rainnc"][...] = cp.asarray(surf["rain"].reshape(1, ncol))
    s["snownc"][...] = cp.asarray(surf["snow"].reshape(1, ncol))
    effc = cp.full((nz, 1, ncol), f32(2.49), cp.float32)
    effi = cp.full((nz, 1, ncol), f32(4.99), cp.float32)
    effs = cp.full((nz, 1, ncol), f32(9.99), cp.float32)
    t = cp.empty((nz, 1, ncol), cp.float32)
    steps = []
    for _ in range(nsteps):
        launch_wsm3(dev["th"], dev["qv"], dev["qc"], dev["qr"], w,
                    dev["den"], dev["pii"], dev["p"], dev["dz"],
                    s["rainnc"], s["rainncv"], s["snownc"], s["snowncv"],
                    s["sr"], dt, effc=effc, effi=effi, effs=effs,
                    # The WRF oracle is exact mode (finding WB5 off).
                    conservative_surface=False)
        cp.multiply(dev["th"], dev["pii"], out=t)
        cp.cuda.runtime.deviceSynchronize()
        g = lambda a: a.get().reshape(nz, ncol)
        lv = {"th": g(dev["th"]), "qv": g(dev["qv"]), "qc": g(dev["qc"]),
              "qr": g(dev["qr"]), "re_cloud": g(effc), "re_ice": g(effi),
              "re_snow": g(effs), "t": g(t)}
        sf = {"rain": s["rainnc"], "rainncv": s["rainncv"], "snow": s["snownc"],
              "snowncv": s["snowncv"], "sr": s["sr"]}
        sf = {k: v.get().reshape(ncol) for k, v in sf.items()}
        steps.append((lv, sf))
    return steps


def _ordered(a):
    bits = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def ulp(a, b):
    return np.abs(_ordered(a) - _ordered(b))


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, "<f4").tobytes()).hexdigest()


def _pairs(wrf_step, gpu_step):
    (wl, ws), (gl, gs) = wrf_step, gpu_step
    out = {n: (wl[n], gl[n]) for n in LEVEL_OUT}
    out.update({n: (ws[n][None, :], gs[n][None, :]) for n in SURFACE_OUT})
    return out


def _accumulate(fields, istep, pairs, regimes, keep=None):
    for n, (w, g) in pairs.items():
        w = np.asarray(w, np.float32)
        g = np.asarray(g, np.float32)
        if keep is not None:
            w, g = w[:, keep], g[:, keep]
            names = [regimes[i]["name"] for i in np.nonzero(keep)[0]]
        else:
            names = [r["name"] for r in regimes]
        diff_bits = w.view(np.uint32) != g.view(np.uint32)
        u = np.where(np.isnan(w) & np.isnan(g), 0, ulp(w, g))
        gap = np.abs(w.astype(np.float64) - g.astype(np.float64))
        gap = np.where(np.isnan(gap), 0.0, gap)
        row = fields.setdefault(n, {"words": 0, "bit_mismatch": 0, "max_ulp": 0,
                                    "max_abs": 0.0, "worst": None, "columns": {},
                                    "nonfinite_gpu": 0, "nonfinite_wrf": 0,
                                    "first_step_with_a_mismatch": None})
        row["words"] += int(w.size)
        row["bit_mismatch"] += int(diff_bits.sum())
        row["nonfinite_gpu"] += int((~np.isfinite(g)).sum())
        row["nonfinite_wrf"] += int((~np.isfinite(w)).sum())
        row["max_abs"] = max(row["max_abs"], float(gap.max()))
        if diff_bits.any() and row["first_step_with_a_mismatch"] is None:
            row["first_step_with_a_mismatch"] = istep + 1
        for i in np.nonzero(diff_bits.any(axis=0))[0]:
            row["columns"][names[i]] = row["columns"].get(names[i], 0) + int(diff_bits[:, i].sum())
        m = int(u.max())
        if diff_bits.any() and (row["worst"] is None or m > row["max_ulp"]):
            k, i = np.unravel_index(int(np.argmax(u)), u.shape)
            row["max_ulp"] = max(row["max_ulp"], m)
            row["worst"] = {"step": istep + 1, "level": int(k), "column": names[i],
                            "wrf": float(w[k, i]), "gpu": float(g[k, i])}


def grade(wrf, gpu, regimes, keep=None):
    fields = {}
    for istep, (ws, gs) in enumerate(zip(wrf["steps"], gpu)):
        _accumulate(fields, istep, _pairs(ws, gs), regimes, keep=keep)
    return fields


def _header():
    import cupy as cp
    import gpuwm.wrf_exact as wrf_exact

    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return {"strict_build": bool(wrf_exact.ENABLED),
            "device": name.decode() if isinstance(name, bytes) else str(name),
            "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
            "cupy": cp.__version__}


def cmd_grade(oracle_dir: Path, flavour: str, exclude=()):
    from gpuwm.core.wsm3 import launch_wsm3  # noqa: F401  the scheme under test
    regimes = json.loads((oracle_dir / "columns.json").read_text())["columns"]
    keep = (None if not exclude else
            np.asarray([r["name"] not in exclude for r in regimes]))
    report = {**_header(), "flavour": flavour, "excluded_columns": list(exclude),
              "runs": {}}
    total = 0
    for dt, nsteps in RUNS:
        tag = tag_of(dt, nsteps)
        wrf = read_wrf(oracle_dir / f"wrf-{flavour}-{tag}.bin")
        fields = grade(wrf, run_gpu(oracle_dir / "columns.bin", dt, nsteps),
                       regimes, keep=keep)
        report["runs"][tag] = fields
        bad = sum(r["bit_mismatch"] for r in fields.values())
        total += bad
        print(f"{tag}: {bad} differing words; " + ", ".join(
            f"{n} {r['bit_mismatch']}/{r['max_ulp']}u" for n, r in fields.items()
            if r["bit_mismatch"]), flush=True)
    report["total_bit_mismatch"] = total
    return report


def _d1_census(oracle_dir: Path, regimes):
    """Where unmodified WRF and the D1-patched WRF differ, word by word."""
    census = {}
    for dt, nsteps in RUNS:
        tag = tag_of(dt, nsteps)
        a = read_wrf(oracle_dir / f"wrf-O0-{tag}.bin")
        b = read_wrf(oracle_dir / f"wrf-O0fix-{tag}.bin")
        fields = grade(a, b["steps"], regimes)
        census[tag] = {n: {"bit_mismatch": r["bit_mismatch"],
                           "max_abs": r["max_abs"], "columns": r["columns"]}
                       for n, r in fields.items() if r["bit_mismatch"]}
    return census


def cmd_make_fixture(oracle_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("columns.bin", "columns.json", "wsm3-init-f32.txt",
                 "libmvec-report.txt", "compiler.txt", "oracle-sha256sums.txt",
                 "d1-fix.diff"):
        shutil.copyfile(oracle_dir / name, out / name)
    regimes = json.loads((oracle_dir / "columns.json").read_text())["columns"]
    digests = {"columns_bin_sha256": hashlib.sha256(
        (oracle_dir / "columns.bin").read_bytes()).hexdigest(),
        "O0": {}, "O0fix": {}}
    final = {}
    for dt, nsteps in RUNS:
        tag = tag_of(dt, nsteps)
        for flavour in ("O0", "O0fix"):
            wrf = read_wrf(oracle_dir / f"wrf-{flavour}-{tag}.bin")
            for istep, (lv, sf) in enumerate(wrf["steps"]):
                for n, value in {**lv, **sf}.items():
                    final[f"{flavour}/{tag}/{istep + 1}/{n}"] = value
            digests[flavour][tag] = [
                {**{n: digest(lv[n]) for n in LEVEL_OUT},
                 **{n: digest(sf[n]) for n in SURFACE_OUT}}
                for lv, sf in wrf["steps"]]
            if flavour == "O0fix":
                lv, sf = wrf["steps"][-1]
                for n in LEVEL_OUT:
                    final[f"{tag}/{n}"] = lv[n]
                for n in SURFACE_OUT:
                    final[f"{tag}/{n}"] = sf[n]
            else:
                for istep, (lv, sf) in enumerate(wrf["steps"]):
                    for n in SURFACE_OUT:
                        final[f"O0/{tag}/{istep + 1}/{n}"] = sf[n]
    census = _d1_census(oracle_dir, regimes)
    digests["d1_census"] = census
    digests["d1_columns"] = sorted({col for run in census.values()
                                    for row in run.values()
                                    for col in row["columns"]})
    (out / "wrf461-step-digests.json").write_text(json.dumps(digests, indent=1) + "\n")
    np.savez_compressed(out / "wrf461-final-step.npz", **final)
    return {"written": sorted(p.name for p in out.iterdir()),
            "d1_columns": digests["d1_columns"]}


def force_tier(tier: int) -> None:
    """Compile the WSM3 kernel at WSM3_KMAX = ``tier`` whatever nz is."""
    import gpuwm.core.wsm3 as wsm3

    wsm3._kernel_capacity = lambda nz, _tier=int(tier): _tier


def plant() -> None:
    """Flip the last bit of every stored theta word, through the loader."""
    import gpuwm.core.kernels as kernels

    original = kernels._unit_text

    def planted(name, kernel_dir=kernels._KDIR):
        text = original(name, kernel_dir)
        if name == "wsm3":
            if text.count(PLANT_FROM) != 1:
                raise RuntimeError("plant site not found exactly once")
            text = text.replace(PLANT_FROM, PLANT_TO)
        return text

    kernels._unit_text = planted


def check_fixture(fixture: Path, *, exclude_columns=(), runs=RUNS):
    """Count every word at every step against unmodified WRF, in both modes.

    Whole-field hashes also pin operation order. The D1 reference is a
    diagnostic, never a substitute for the strict stock-WRF acceptance gate.
    """
    from gpuwm.core.wsm3 import launch_wsm3  # noqa: F401
    regimes = json.loads((fixture / "columns.json").read_text())["columns"]
    digests = json.loads((fixture / "wrf461-step-digests.json").read_text())
    final = np.load(fixture / "wrf461-final-step.npz")
    names = [r["name"] for r in regimes]
    keep = None if not exclude_columns else np.asarray(
        [n not in exclude_columns for n in names])
    report = {**_header(), "reference": "O0", "excluded_columns": list(exclude_columns),
              "runs": {}}
    total = 0
    first_th_ulp = []
    for dt, nsteps in runs:
        tag = tag_of(dt, nsteps)
        gpu = run_gpu(fixture / "columns.bin", dt, nsteps)
        fields, d1_fields, hashes = {}, {}, []
        for istep, step in enumerate(gpu):
            def reference(flavour):
                return ({n: final[f"{flavour}/{tag}/{istep + 1}/{n}"] for n in LEVEL_OUT},
                        {n: final[f"{flavour}/{tag}/{istep + 1}/{n}"] for n in SURFACE_OUT})
            raw, fixed = reference("O0"), reference("O0fix")
            _accumulate(fields, istep, _pairs(raw, step), regimes, keep=keep)
            _accumulate(d1_fields, istep, _pairs(fixed, step), regimes, keep=keep)
            gl, gs = step
            got = {n: digest(v) for n, v in {**gl, **gs}.items()}
            hashes.append({"step": istep + 1, "gpu": got,
                           "wrf": digests["O0"][tag][istep]})
            if istep == 0:
                first_th_ulp.append(int(ulp(raw[0]["th"], gl["th"]).max()))
        bad = sum(r["bit_mismatch"] for r in fields.values())
        total += bad
        report["runs"][tag] = {"fields": fields, "vs_D1_reference": d1_fields,
                               "field_hashes": hashes}
    report["total_mismatch"] = total
    report["surface_rain_difference_words_vs_O0"] = sum(
        r["fields"][n]["bit_mismatch"] for r in report["runs"].values() for n in D1_FIELDS)
    report["first_step_th_max_ulp"] = first_th_ulp
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade")
    g.add_argument("oracle_dir", type=Path)
    g.add_argument("--flavour", default="O0", choices=("O0", "O0fix", "O2vec"))
    g.add_argument("--exclude", action="append", default=[])
    g.add_argument("--out", type=Path)
    m = sub.add_parser("make-fixture")
    m.add_argument("oracle_dir", type=Path)
    m.add_argument("out_dir", type=Path)
    c = sub.add_parser("check-fixture")
    c.add_argument("fixture_dir", type=Path)
    c.add_argument("--exclude", action="append", default=[])
    c.add_argument("--tier", type=int)
    c.add_argument("--plant", action="store_true")
    c.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "make-fixture":
        print(json.dumps(cmd_make_fixture(args.oracle_dir, args.out_dir)))
        return 0
    if args.cmd == "grade":
        report = cmd_grade(args.oracle_dir, args.flavour, tuple(args.exclude))
        bad = report["total_bit_mismatch"]
        summary = {tag: {n: [f["bit_mismatch"], f["max_ulp"]]
                         for n, f in r.items() if f["bit_mismatch"]}
                   for tag, r in report["runs"].items()}
    else:
        if args.tier:
            force_tier(args.tier)
        if args.plant:
            plant()
        report = check_fixture(args.fixture_dir,
                               exclude_columns=tuple(args.exclude))
        report["tier"] = args.tier
        report["planted"] = bool(args.plant)
        bad = report["total_mismatch"]
        summary = {tag: {n: [f["bit_mismatch"], f["max_ulp"]]
                         for n, f in r["fields"].items() if f["bit_mismatch"]}
                   for tag, r in report["runs"].items()}

    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({"total_mismatch": bad, "strict_build": report["strict_build"],
                      "reference": report.get("reference"),
                      "surface_rain_difference_words_vs_O0": report.get("surface_rain_difference_words_vs_O0"),
                      "summary": summary,
                      "first_step_th_max_ulp": report.get("first_step_th_max_ulp")}))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
