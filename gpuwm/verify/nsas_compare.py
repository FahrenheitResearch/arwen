"""Grade the NSAS kernels (cu_physics = 96) against the WRF v4.6.1 fixtures.

    python -m gpuwm.verify.nsas_compare [--fixtures DIR] [--plant NAME=HEX]

Runs, on the current device and in the current arithmetic arm (strict when
the process carries ``GPUWM_WRF_EXACT=1``), every single-column oracle case
through

* ``nsas_column`` -- the whole cu_nsas call, against the WRF driver entry's
  tendencies, RAINCV, PRATEC, HBOT and HTOP, and its column prep;
* ``nsas_deep_stage`` -- nsas2d alone, from the oracle's prepared inputs;
* ``nsas_shallow_stage`` -- nscv2d alone, from the oracle's nsas2d state;

and prints one JSON document: per field, the words compared, the words that
differ and the largest distance in units in the last place.  ``--plant``
compiles a copy of nsas.cu with one ``#define`` replaced by the float whose
bits are HEX (the one-bit mutation checks).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from gpuwm.verify import nsas_oracle as no

FIXTURES = (Path(__file__).resolve().parents[2]
            / "tests" / "data" / "oracles" / "nsas")


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|ordered(a) - ordered(b)| on float32 bit patterns (NaN == NaN)."""
    from gpuwm.core.fp32_ulp import monotone_fp32_key
    d = np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))
    both_nan = np.isnan(a) & np.isnan(b)
    return np.where(both_nan, 0, d)


def load(nz: int, fixtures: Path = FIXTURES) -> dict[str, np.ndarray]:
    with np.load(fixtures / f"nsas-nz{nz}.npz") as z:
        return {k: z[k] for k in z.files}


def _metric(got, want, mask=None):
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    if mask is not None:
        got, want = got[mask], want[mask]
    d = ulp_distance(got, want)
    different = got.view(np.uint32) != want.view(np.uint32)
    bad = np.flatnonzero(different)
    samples = [{"index": int(i), "got": float(got.flat[i]),
                "want": float(want.flat[i]),
                "got_bits": f"{got.view(np.uint32).flat[i]:08x}",
                "want_bits": f"{want.view(np.uint32).flat[i]:08x}",
                "ulp": int(d.flat[i])} for i in bad[:3]]
    return {"words": int(d.size), "different_words": int(different.sum()),
            "max_ulp": int(d.max()) if d.size else 0,
            "max_abs": float(np.max(np.abs(got-want))) if d.size else 0.,
            "samples": samples}


def planted_module(plant: str | None):
    """The nsas unit as load_module compiles it, optionally one-bit planted."""
    from gpuwm.core.kernels import load_module, module_options, module_source
    if not plant:
        return load_module("nsas")
    from gpuwm.core.kernels import compile_module_no_ftz
    name, hexbits = plant.split("=")
    src = module_source("nsas")
    pattern = re.compile(rf"^#define {re.escape(name)}\s+\S+$", re.M)
    if len(pattern.findall(src)) != 1:
        raise ValueError(f"no single #define {name} to plant")
    src = pattern.sub(f"#define {name} __int_as_float({hexbits})", src)
    return compile_module_no_ftz("nsas", src, module_options("nsas"))


def stage_preparation(fx, idx):
    """The level-major kernel contract requires C-contiguous stage inputs."""
    return np.ascontiguousarray(np.concatenate(
        [fx[f"out.prep.{n}"][idx].T for n in no.PREP_LEV]
        + [fx["out.prep.prsi"][idx].T], axis=0), dtype=np.float32)


def run(nz: int, fixtures: Path = FIXTURES, plant: str | None = None, *, rows: bool = False) -> dict:
    import cupy as cp

    from gpuwm.core import column_cumulus as cc

    fx = load(nz, fixtures)
    hdr_i, hdr_f = fx["hdr_i"], fx["hdr_f"]
    single = hdr_i[:, 2] == 1
    idx = np.nonzero(~single if rows else single)[0]
    ncol = int(idx.size)
    mp = hdr_i[idx, 4]
    ncloud = np.array([cc.wrf_sas_ncloud(m) for m in mp], dtype=np.int32)

    def lev(name):
        return cp.asarray(np.ascontiguousarray(fx[f"in.{name}"][idx].T))

    mass = {k: lev(src) for k, src in zip(cc.MASS_INPUTS, (
        "t", "qv", "qc", "qi", "p", "pi", "rho", "dz8w", "u", "v"))}
    iface = {"p8w": lev("p8w"), "w": lev("w")}
    sc_h = np.zeros((ncol, len(cc.SC_FIELDS)), dtype=np.float32)
    sc_h[:, cc.SC_FIELDS.index("dt")] = hdr_f[idx, 0]
    sc_h[:, cc.SC_FIELDS.index("dx")] = hdr_f[idx, 1]
    sc_h[:, cc.SC_FIELDS.index("pgcon")] = hdr_f[idx, 2]
    sc_h[:, cc.SC_FIELDS.index("xland")] = hdr_f[idx, 3]
    sc_h[:, cc.SC_FIELDS.index("hpbl")] = hdr_f[idx, 4]
    sc_h[:, cc.SC_FIELDS.index("hfx")] = hdr_f[idx, 5]
    sc_h[:, cc.SC_FIELDS.index("qfx")] = hdr_f[idx, 6]
    ic_h = np.zeros((ncol, len(cc.IC_FIELDS)), dtype=np.int32)
    ic_h[:, cc.IC_FIELDS.index("stepcu")] = hdr_i[idx, 6]
    ic_h[:, cc.IC_FIELDS.index("ncloud")] = ncloud
    ic_h[:, cc.IC_FIELDS.index("dx_factor")] = hdr_i[idx, 5]
    sc, ic = cp.asarray(sc_h), cp.asarray(ic_h)

    module = planted_module(plant)
    scheme = cc.COLUMN_CUMULUS_SCHEMES[96]
    # Every output slot exists; F_QC/F_QI masks are applied when grading,
    # exactly where WRF would not have formed the tendency.
    outs = {n: cp.full((nz, ncol), np.nan, dtype=np.float32)
            for n in cc.LEVEL_OUTPUTS}
    sfc = cp.full((4, ncol), np.nan, dtype=np.float32)
    prep = cp.full((4 * nz + nz + 1, ncol), np.nan, dtype=np.float32)
    cc.launch_columns(scheme, mass=mass, iface=iface, sc=sc, ic=ic,
                      outputs=outs, sfc=sfc, nz=nz, ncol=ncol, prep=prep,
                      module=module, row_width=2 if rows else 1)
    cp.cuda.Device().synchronize()

    report: dict[str, dict] = {}
    f_qc = (ncloud >= 1)[:, None]
    f_qi = (ncloud == 2)[:, None]
    for name in cc.LEVEL_OUTPUTS:
        got = cp.asnumpy(outs[name]).T
        want = fx[f"out.driver.{name}"][idx]
        mask = None
        if name == "rqccuten":
            mask = np.broadcast_to(f_qc, got.shape)
        elif name == "rqicuten":
            mask = np.broadcast_to(f_qi, got.shape)
        report[f"driver.{name}"] = _metric(got, want, mask)
    sfc_h = cp.asnumpy(sfc)
    for j, name in enumerate(cc.SURFACE_OUTPUTS):
        report[f"driver.{name}"] = _metric(sfc_h[j], fx[f"out.driver.{name}"][idx, 0])
    prep_h = cp.asnumpy(prep)
    for j, name in enumerate(no.PREP_LEV):
        report[f"prep.{name}"] = _metric(prep_h[j * nz:(j + 1) * nz].T,
                                         fx[f"out.prep.{name}"][idx])
    report["prep.prsi"] = _metric(prep_h[4 * nz:].T, fx["out.prep.prsi"][idx])

    # The stage oracles, each from the oracle's own inputs to that stage.
    prep_in = cp.asarray(stage_preparation(fx, idx))
    ws = cp.empty(scheme.workspace_slots * (nz + 2) * ncol, dtype=np.float32)
    block = cc.BLOCK
    grid = ((ncol + block - 1) // block,)
    pi_in = mass["pi"]
    for stage, entry, source in (
            ("deep", "nsas_deep_stage", None),
            ("shallow", "nsas_shallow_stage", "deep")):
        if source is None:
            state_h = np.concatenate(
                [fx[f"in.{n}"][idx].T for n in ("t", "qv", "u", "v", "qc", "qi")],
                axis=0)
            ki_h = np.zeros((ncol, 3), dtype=np.int32)
        else:
            state_h = np.concatenate(
                [fx[f"out.{source}.{n}"][idx].T for n in no.STAGE_LEV], axis=0)
            ki_h = np.stack([fx[f"out.{source}.{n}"][idx, 0]
                             for n in ("kbot", "ktop", "icps")],
                            axis=1).astype(np.int32)
        state = cp.asarray(np.ascontiguousarray(state_h, dtype=np.float32))
        ki = cp.asarray(ki_h)
        rain = cp.full(ncol, np.nan, dtype=np.float32)
        module.get_function(entry)(grid, (block,), (
            prep_in, state, pi_in, sc, ic, rain, ki, ws,
            np.int32(ncol), np.int32(nz)))
        cp.cuda.Device().synchronize()
        st = cp.asnumpy(state)
        for j, name in enumerate(no.STAGE_LEV):
            report[f"{stage}.{name}"] = _metric(st[j * nz:(j + 1) * nz].T,
                                                fx[f"out.{stage}.{name}"][idx])
        report[f"{stage}.rain"] = _metric(cp.asnumpy(rain),
                                          fx[f"out.{stage}.rain"][idx, 0])
        ki_out = cp.asnumpy(ki).astype(np.float32)
        for j, name in enumerate(("kbot", "ktop", "icps")):
            report[f"{stage}.{name}"] = _metric(ki_out[:, j],
                                                fx[f"out.{stage}.{name}"][idx, 0])
    return {"nz": nz, "columns": ncol, "row_width": 2 if rows else 1, "fields": report}


def row_coupling(nz: int, fixtures: Path = FIXTURES) -> dict:
    """Columns whose WRF answer moves when a neighbour shares the i-row."""
    fx = load(nz, fixtures)
    hdr_i = fx["hdr_i"]
    rows = np.nonzero(hdr_i[:, 3] == no.FAMILIES["row"])[0]
    solos = np.nonzero(hdr_i[:, 3] == no.FAMILIES["row-solo"])[0]
    if rows.size != solos.size:
        raise ValueError("row and row-solo members do not pair")
    moved = 0
    for r, s in zip(rows, solos):
        if any((fx[f"out.driver.{n}"][r].view(np.int32)
                != fx[f"out.driver.{n}"][s].view(np.int32)).any()
               for n in no.DRIVER_LEV + no.DRIVER_SFC):
            moved += 1
    return {"row_columns": int(rows.size), "columns_moved_by_row": moved}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    ap.add_argument("--plant", default=None)
    ap.add_argument("--nz", type=int, nargs="*", default=[40, 60])
    args = ap.parse_args(argv)
    from gpuwm import wrf_exact
    doc = {"strict": bool(wrf_exact.ENABLED), "plant": args.plant,
           "runs": [run(nz, args.fixtures, args.plant, rows=rows)
                    for rows in (False, True) for nz in args.nz],
           "row_coupling": [row_coupling(nz, args.fixtures) for nz in args.nz]}
    json.dump(doc, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
