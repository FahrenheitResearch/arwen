"""Run WOOF's ra=90 adapter on the oracle columns and grade it.

Three subcommands.

  python compare.py run COLUMNS_DIR OUT.npz
      Evaluates the production adapter (``AnalyticClearSkyRadiation``
      through its radiation-slot ``__call__``) and the lw-only / sw-only
      compositions (``ComposedRadiation``, which is what ra_lw=90 with
      ra_sw=0, or the reverse, constructs) on the GPU for every calendar
      case.  The arithmetic mode is whatever the process was started with:
      GPUWM_WRF_EXACT=1 for the strict build, unset for default.

  python compare.py grade COLUMNS_DIR ORACLE_DIR STRICT.npz DEFAULT.npz OUT.json
      COSZEN against WRF 4.6.1's radconst + calc_coszen (cz-out-O0.bin,
      the scalar-libm reference, and cz-out-O2vec.bin, WRF's own flags),
      word for word.  SWDOWN, GLW and GSW have no WRF definition (WRF 4.6.1
      has no ra=90 scheme), so they are graded against a float64
      evaluation of WOOF's own formula on the same float32 inputs, as an
      accuracy figure, and strict against default.  RTHRATENLW/RTHRATENSW
      must be exact zeros.

  python compare.py fixture COLUMNS_DIR ORACLE_DIR OUT.npz
      Writes the committed test fixture
      (tests/data/oracles/analytic_clear_sky/wrf461_coszen.npz): the column
      inputs, the calendar cases and WRF's O0 output words.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

F = np.float32
OUTPUTS = ("swdown", "glw", "gsw", "coszen", "rthratenlw", "rthratensw")


def _load_columns(columns_dir):
    return dict(np.load(Path(columns_dir) / "columns.npz"))


def _cases(c):
    for k, label in enumerate(c["case_labels"]):
        yield (str(label), datetime(*[int(v) for v in c["starts"][k]]),
               int(c["elapsed"][k]), float(c["radt"][k]), float(c["dt"][k]))


def run(columns_dir, out_path):
    import gpuwm  # noqa: F401  (installs the strict compiler hook when selected)
    from gpuwm import wrf_exact
    import cupy as cp
    from gpuwm.core.analytic_radiation import AnalyticClearSkyRadiation
    from gpuwm.core.radiation_composition import ComposedRadiation

    c = _load_columns(columns_dir)
    ncol = c["lat"].size
    lat = cp.asarray(c["lat"].reshape(1, ncol))
    lon = cp.asarray(c["lon"].reshape(1, ncol))

    def level(name):
        values = c[name].reshape(1, 1, ncol)
        return cp.asarray(np.concatenate([values, values], axis=0))

    atmosphere = {"theta": level("theta"), "pressure": level("pressure"),
                  "qv": level("qv")}
    fields = {"albedo": cp.asarray(c["albedo"].reshape(1, ncol)),
              "glw": cp.zeros((1, ncol), cp.float32)}
    saved = {}
    dtypes = {}
    for k, (label, start, elapsed, radt, dt) in enumerate(_cases(c)):
        state = SimpleNamespace(elapsed_seconds=float(elapsed))
        cfg = SimpleNamespace(radt=radt, radt_minutes=radt, dt=dt,
                              clock_dt=0.0, use_adaptive_time_step=False,
                              swint_opt=0)
        args = dict(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        paths = {
            "adapter": AnalyticClearSkyRadiation(start, lat, lon),
            "lw-only": ComposedRadiation(
                start, lat, lon,
                longwave_adapter=AnalyticClearSkyRadiation(start, lat, lon)),
            "sw-only": ComposedRadiation(
                start, lat, lon,
                shortwave_adapter=AnalyticClearSkyRadiation(start, lat, lon)),
        }
        for path, callable_ in paths.items():
            result = callable_(**args)
            for name in OUTPUTS:
                value = getattr(result, name)
                if path == "lw-only" and name in ("swdown", "gsw", "rthratensw"):
                    continue  # no shortwave spectrum selected
                if path == "sw-only" and name in ("glw", "rthratenlw"):
                    continue  # no longwave spectrum selected
                dtypes[f"{path}/{name}"] = str(value.dtype)
                host = cp.asnumpy(value)
                if name.startswith("rthraten"):
                    saved[f"{path}/{name}/{k}/nonzero"] = np.array(
                        int(np.count_nonzero(host)))
                    continue
                saved[f"{path}/{name}/{k}"] = host.reshape(-1)
    saved["meta"] = np.array(json.dumps({
        "strict": bool(wrf_exact.ENABLED), "dtypes": dtypes,
        "cupy": cp.__version__,
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}))
    np.savez(out_path, **saved)
    print(f"wrote {out_path} strict={wrf_exact.ENABLED}")


# ---------------------------------------------------------------------------

def _ordered(values):
    bits = np.asarray(values, np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def ulp(a, b):
    return np.abs(_ordered(a) - _ordered(b))


def _read_oracle(path, ncol, ncase):
    raw = np.fromfile(path, dtype="<f4")
    per = 5 + 2 * ncol
    if raw.size != per * ncase:
        raise SystemExit(f"{path}: {raw.size} words, expected {per * ncase}")
    raw = raw.reshape(ncase, per)
    return {"julian": raw[:, 0], "xtime": raw[:, 1], "gmt": raw[:, 2],
            "declin": raw[:, 3], "solcon": raw[:, 4],
            "coszen": raw[:, 5:5 + ncol], "hrang": raw[:, 5 + ncol:]}


def _stats(got, want):
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    same = got.view(np.uint32) == want.view(np.uint32)
    u = ulp(got, want)
    return {"words": int(got.size), "bitwise_equal": int(same.sum()),
            "max_ulp": int(u.max()) if u.size else 0,
            "max_abs": float(np.max(np.abs(got.astype(np.float64)
                                           - want.astype(np.float64))))}


def _merge(total, part):
    if not total:
        return dict(part)
    return {"words": total["words"] + part["words"],
            "bitwise_equal": total["bitwise_equal"] + part["bitwise_equal"],
            "max_ulp": max(total["max_ulp"], part["max_ulp"]),
            "max_abs": max(total["max_abs"], part["max_abs"])}


def _reference64(c, start, elapsed):
    """WOOF's own formula, evaluated in float64 on the float32 inputs."""
    from gpuwm.core.analytic_radiation import analytic_clear_sky_forcing
    valid = start + timedelta(seconds=float(elapsed))
    theta = c["theta"].astype(np.float64)
    p = c["pressure"].astype(np.float64)
    t = theta * (p / 100000.0) ** (287.0 / 1004.0)
    sw, lw = analytic_clear_sky_forcing(
        valid, c["lat"].astype(np.float64), c["lon"].astype(np.float64),
        t, c["qv"].astype(np.float64), p)
    gsw = sw * (1.0 - c["albedo"].astype(np.float64))
    return {"swdown": sw, "glw": lw, "gsw": gsw}


def grade(columns_dir, oracle_dir, strict_path, default_path, out_path):
    c = _load_columns(columns_dir)
    ncol = c["lat"].size
    cases = list(_cases(c))
    ncase = len(cases)
    oracle = {tag: _read_oracle(Path(oracle_dir) / f"cz-out-{tag}.bin", ncol, ncase)
              for tag in ("O0", "O2vec")}
    runs = {"strict": dict(np.load(strict_path)),
            "default": dict(np.load(default_path))}
    report = {"ncol": int(ncol), "ncase": ncase,
              "column_evaluations": int(ncol * ncase),
              "regimes": {r: int((c["regimes"] == r).sum())
                          for r in dict.fromkeys(c["regimes"].tolist())},
              "cases": [case[0] for case in cases],
              "meta": {mode: json.loads(str(runs[mode]["meta"]))
                       for mode in runs}}

    report["wrf_O0_vs_O2vec_coszen"] = _stats(oracle["O2vec"]["coszen"],
                                              oracle["O0"]["coszen"])
    coszen = {}
    per_case = []
    for mode, data in runs.items():
        for path in ("adapter", "lw-only", "sw-only"):
            total = {}
            for k in range(ncase):
                part = _stats(data[f"{path}/coszen/{k}"], oracle["O0"]["coszen"][k])
                total = _merge(total, part)
                if mode == "strict" and path == "adapter":
                    per_case.append({"case": cases[k][0], **part,
                                     "wrf_julian": float(oracle["O0"]["julian"][k]),
                                     "wrf_xtime": float(oracle["O0"]["xtime"][k]),
                                     "wrf_gmt": float(oracle["O0"]["gmt"][k])})
            coszen[f"{mode}/{path}"] = total
    report["coszen_vs_wrf461_O0"] = coszen
    report["coszen_per_case_strict_adapter"] = per_case

    proxy = {}
    for mode, data in runs.items():
        for name in ("swdown", "glw", "gsw"):
            total = {}
            for k, (_label, start, elapsed, _radt, _dt) in enumerate(cases):
                ref = _reference64(c, start, elapsed)[name].astype(np.float32)
                total = _merge(total, _stats(data[f"adapter/{name}/{k}"], ref))
            proxy[f"{mode}/{name}"] = total
    report["proxy_vs_float64_same_formula"] = proxy

    strict_vs_default = {}
    for key in runs["strict"]:
        if key == "meta" or key.endswith("/nonzero"):
            continue
        name = key.split("/")[1]
        path = key.split("/")[0]
        strict_vs_default[f"{path}/{name}"] = _merge(
            strict_vs_default.get(f"{path}/{name}", {}),
            _stats(runs["default"][key], runs["strict"][key]))
    report["default_vs_strict"] = strict_vs_default
    report["heating_nonzero_words"] = {
        mode: int(sum(int(v) for k, v in data.items() if k.endswith("/nonzero")))
        for mode, data in runs.items()}
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"coszen_vs_wrf461_O0": coszen,
                      "wrf_O0_vs_O2vec": report["wrf_O0_vs_O2vec_coszen"],
                      "heating_nonzero": report["heating_nonzero_words"]},
                     indent=1))


def fixture(columns_dir, oracle_dir, out_path):
    """The committed test fixture: inputs plus WRF's O0 words."""
    c = _load_columns(columns_dir)
    ncol = c["lat"].size
    ncase = c["case_labels"].size
    o = _read_oracle(Path(oracle_dir) / "cz-out-O0.bin", ncol, ncase)
    np.savez_compressed(
        out_path, lat=c["lat"], lon=c["lon"], labels=c["labels"],
        regimes=c["regimes"], case_labels=c["case_labels"],
        starts=c["starts"], elapsed=c["elapsed"], radt=c["radt"], dt=c["dt"],
        wrf_julian=o["julian"], wrf_xtime=o["xtime"], wrf_gmt=o["gmt"],
        wrf_declin=o["declin"], wrf_solcon=o["solcon"],
        wrf_coszen=o["coszen"], wrf_hrang=o["hrang"])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "fixture" and len(sys.argv) == 5:
        fixture(*sys.argv[2:])
    elif len(sys.argv) >= 2 and sys.argv[1] == "run" and len(sys.argv) == 4:
        run(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "grade" and len(sys.argv) == 7:
        grade(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
