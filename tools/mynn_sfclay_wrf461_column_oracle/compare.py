"""Grade WOOF's MYNN surface kernel against the WRF 4.6.1 column oracle.

Reads the oracle file ``run_columns.F90`` writes and runs WOOF's
``mynn_surface_column`` kernel (through ``gpuwm.core.mynn_sfclay``, the
production launch path) on the identical float32 inputs, two ways:

* replay  -- every (config, step) is entered with the state WRF entered
  that call with, so each step is graded on its own;
* free    -- the kernel carries its own UST/MOL/QSFC/ZNT/USTM/HFX/QFX from
  step to step, and a seeded config's first step runs WOOF's own
  ``seed_mynn_surface_first_step`` (the SFCLAY_mynn :330-337 block), whose
  result is graded against the state WRF entered step 1 with.

Every one of the 35 outputs is compared bit for bit.  A NaN equals only the
identical NaN bit pattern.  The arithmetic mode is whatever the process was
started with: set ``GPUWM_WRF_EXACT=1`` before importing gpuwm for the
strict build, leave it unset for default arithmetic.

    python -m tools.mynn_sfclay_wrf461_column_oracle.compare \
        ORACLE.bin INPUT_DIR --receipt receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

from tools.mynn_sfclay_wrf461_column_oracle.columns import (
    CONFIGS, FIELDS, OUTPUT_FIELDS, STATE_FIELDS, STATIC_FIELDS, spp_pattern,
)


def read_columns(path: Path):
    blob = Path(path).read_bytes()
    ncol, nfield = struct.unpack("<ii", blob[:8])
    if nfield != len(FIELDS):
        raise ValueError(f"{path}: {nfield} fields, expected {len(FIELDS)}")
    table = np.frombuffer(blob[8:], dtype="<f4").reshape(ncol, nfield)
    return {name: table[:, k].copy() for k, name in enumerate(FIELDS)}


def read_oracle(path: Path):
    """[(cfg, step, itimestep, state_in{}, out{})] in file order."""
    blob = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        import gzip
        blob = gzip.decompress(blob)
    ncfg, ncol, nstate, nout = struct.unpack("<iiii", blob[:16])
    if (nstate, nout) != (len(STATE_FIELDS), len(OUTPUT_FIELDS)):
        raise ValueError("oracle record shape does not match columns.py")
    if ncfg != len(CONFIGS):
        raise ValueError(f"oracle has {ncfg} configs, columns.py {len(CONFIGS)}")
    records, pos = [], 16
    for _ in range(sum(c[6] for c in CONFIGS)):
        cfg, step, itimestep = struct.unpack("<iii", blob[pos:pos + 12])
        pos += 12
        state = np.frombuffer(blob, "<f4", nstate * ncol, pos).reshape(
            nstate, ncol)
        pos += 4 * nstate * ncol
        out = np.frombuffer(blob, "<f4", nout * ncol, pos).reshape(nout, ncol)
        pos += 4 * nout * ncol
        records.append((cfg, step, itimestep,
                        dict(zip(STATE_FIELDS, state.copy())),
                        dict(zip(OUTPUT_FIELDS, out.copy()))))
    if pos != len(blob):
        raise ValueError(f"{path}: {len(blob) - pos} trailing bytes")
    return ncol, records


def _ulp(got, want):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    return fp32_ulp_distance(got, want)


def _grade(got, want):
    """Per-field {mismatched, max_ulp, max_abs} for one call's outputs."""
    rows = {}
    for name, ref in want.items():
        g = np.asarray(got[name], dtype=np.float32).reshape(ref.shape)
        same = g.view(np.uint32) == ref.view(np.uint32)
        bad = int((~same).sum())
        if bad:
            ulp = _ulp(g, ref)
            with np.errstate(invalid="ignore"):
                diff = np.abs(g.astype(np.float64) - ref.astype(np.float64))
            rows[name] = {"mismatched": bad, "max_ulp": int(np.max(ulp)),
                          "max_abs": float(np.nanmax(diff)) if
                          np.isfinite(diff).any() else float("nan"),
                          "columns": [int(c) for c in np.nonzero(~same)[0]]}
        else:
            rows[name] = {"mismatched": 0, "max_ulp": 0, "max_abs": 0.0,
                          "columns": []}
    return rows


def run(oracle_path, input_dir, *, variant="wrf_461"):
    import cupy as cp
    from gpuwm.core import mynn_sfclay
    from gpuwm.wrf_exact import ENABLED as strict

    columns = read_columns(Path(input_dir) / "columns.bin")
    ncol, records = read_oracle(oracle_path)
    shape = (1, ncol)
    dev = {f: cp.asarray(columns[f].reshape(shape)) for f in STATIC_FIELDS}
    pattern = cp.asarray(spp_pattern(ncol).reshape(shape))

    def call(cfg, itimestep, state):
        _, isftcflx, isfflx, dx, _, _, _, spp = CONFIGS[cfg]
        values = dict(dev)
        for f in ("hfx", "qfx", "znt", "qsfc", "ust"):
            values[f] = cp.asarray(np.asarray(state[f], np.float32)
                                   .reshape(shape))
        result = mynn_sfclay.mynn_surface_layer(
            values, dx=dx, itimestep=itimestep, isfflx=isfflx,
            isftcflx=isftcflx,
            mol=cp.asarray(np.asarray(state["mol"], np.float32).reshape(shape)),
            ustm=cp.asarray(np.asarray(state["ustm"], np.float32)
                            .reshape(shape)),
            spp_pbl=spp, pattern_spp_pbl=pattern if spp else None,
            variant=variant)
        cp.cuda.runtime.deviceSynchronize()
        return {name: cp.asnumpy(getattr(result, name)).reshape(ncol)
                for name in OUTPUT_FIELDS}

    receipt = {"strict": bool(strict), "ncol": ncol, "variant": variant,
               "replay": [], "free": [], "seed": []}
    # Replay: each call entered with WRF's own entering state.
    for cfg, step, itimestep, state, want in records:
        got = call(cfg, itimestep, state)
        receipt["replay"].append({"config": CONFIGS[cfg][0], "step": step,
                                  "itimestep": itimestep,
                                  "fields": _grade(got, want)})
    # Free-running: WOOF carries its own state.
    by_cfg = {}
    for rec in records:
        by_cfg.setdefault(rec[0], []).append(rec)
    for cfg, recs in sorted(by_cfg.items()):
        name, _, _, _, _, seeded, _, _ = CONFIGS[cfg]
        state = {f: columns[f].copy() for f in STATE_FIELDS}
        for _, step, itimestep, wrf_state, want in recs:
            if seeded and itimestep == 1:
                arrays = {f: cp.asarray(state[f].reshape(shape))
                          for f in ("ust", "mol", "qsfc")}
                qstar = cp.zeros(shape, dtype=cp.float32)
                mynn_sfclay.seed_mynn_surface_first_step(
                    dev["u1"], dev["v1"], dev["qv1"], ust=arrays["ust"],
                    mol=arrays["mol"], qsfc=arrays["qsfc"], qstar=qstar)
                for f, a in arrays.items():
                    state[f] = cp.asnumpy(a).reshape(ncol)
                receipt["seed"].append({"config": name, "fields": _grade(
                    state, {f: wrf_state[f] for f in STATE_FIELDS})})
            got = call(cfg, itimestep, state)
            receipt["free"].append({"config": name, "step": step,
                                    "itimestep": itimestep,
                                    "fields": _grade(got, want)})
            for f in STATE_FIELDS:
                state[f] = got[f].copy()
    return receipt


def summarize(receipt):
    """Overall and per-field maxima over every graded call."""
    per_field = {}
    total = mism = 0
    for kind in ("replay", "free", "seed"):
        for entry in receipt[kind]:
            for name, row in entry["fields"].items():
                key = per_field.setdefault(name, {"max_ulp": 0,
                                                  "max_abs": 0.0,
                                                  "mismatched": 0})
                key["max_ulp"] = max(key["max_ulp"], row["max_ulp"])
                if np.isfinite(row["max_abs"]):
                    key["max_abs"] = max(key["max_abs"], row["max_abs"])
                key["mismatched"] += row["mismatched"]
                mism += row["mismatched"]
                total += receipt["ncol"]
    return {"compared_values": total, "mismatched_values": mism,
            "max_ulp": max((v["max_ulp"] for v in per_field.values()),
                           default=0),
            "max_abs": max((v["max_abs"] for v in per_field.values()),
                           default=0.0),
            "per_field": per_field}


def coverage(oracle_path, input_dir):
    """Which branches the WRF answers exercised, read from the outputs."""
    columns = read_columns(Path(input_dir) / "columns.bin")
    _, records = read_oracle(oracle_path)
    za = 0.5 * columns["dz1"]
    za2 = columns["dz1"] + 0.5 * columns["dz2"]
    out = {"regime": {}, "zol_abs_gt_10": 0, "snow_andreas": 0,
           "water": 0, "xland_1p5": 0, "za_le_7_level2": 0,
           "za_le_7_log": 0, "za_7_13": 0, "za_ge_13": 0, "hfx_floor": 0,
           "calls": len(records)}
    for _, _, _, _, want in records:
        for r in (1, 2, 3, 4):
            out["regime"][str(r)] = out["regime"].get(str(r), 0) + int(
                (want["regime"] == r).sum())
        out["zol_abs_gt_10"] += int((np.abs(want["zol"]) > 10).sum())
        out["hfx_floor"] += int((want["hfx"] == np.float32(-250.0)).sum())
    out["snow_andreas"] = int(((columns["xland"] < 1.5)
                               & (columns["snowh"] >= 0.1)).sum())
    out["water"] = int((columns["xland"] >= 1.5).sum())
    out["xland_1p5"] = int((columns["xland"] == 1.5).sum())
    out["za_le_7_level2"] = int(((za <= 7) & (za2 > 7) & (za2 < 13)).sum())
    out["za_le_7_log"] = int(((za <= 7) & ~((za2 > 7) & (za2 < 13))).sum())
    out["za_7_13"] = int(((za > 7) & (za < 13)).sum())
    out["za_ge_13"] = int((za >= 13).sum())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("oracle")
    p.add_argument("input_dir")
    p.add_argument("--receipt", required=True)
    a = p.parse_args(argv)
    receipt = run(a.oracle, a.input_dir)
    receipt["summary"] = summarize(receipt)
    receipt["coverage"] = coverage(a.oracle, a.input_dir)
    receipt["oracle_file_sha256"] = hashlib.sha256(
        Path(a.oracle).read_bytes()).hexdigest()
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        receipt["device"] = props["name"].decode()
        receipt["nvrtc"] = ".".join(map(str, cp.cuda.nvrtc.getVersion()))
    except Exception as exc:  # receipt detail only
        receipt["device"] = f"unknown ({exc})"
    receipt["env_GPUWM_WRF_EXACT"] = os.environ.get("GPUWM_WRF_EXACT", "")
    Path(a.receipt).write_text(json.dumps(receipt, indent=1) + "\n")
    s = receipt["summary"]
    print(f"strict={receipt['strict']} compared={s['compared_values']} "
          f"mismatched={s['mismatched_values']} max_ulp={s['max_ulp']} "
          f"max_abs={s['max_abs']:.3e}")
    for name, row in s["per_field"].items():
        if row["mismatched"]:
            print(f"  {name:7s} mismatched={row['mismatched']:5d} "
                  f"max_ulp={row['max_ulp']} max_abs={row['max_abs']:.3e}")
    return 0 if s["mismatched_values"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
