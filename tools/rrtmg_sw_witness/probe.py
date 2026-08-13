"""Per-site subnormal census for the legacy RRTMG shortwave CUDA unit.

WHAT IT ANSWERS

``gpuwm/core/kernels/rrtmg_sw.cu`` writes its 679 FP32 add/sub/mul sites as
inline PTX without the ``.ftz`` modifier so that subnormals survive whatever
compile route the module is built through.  That armor is only worth its
keep if subnormals actually occur, and it is only safe if the PTX computes
what the FP64 emulation it replaced computed.  This probe measures both, on
the card, on real columns:

  * per SOURCE LINE, how many times the site executed, how many of those
    calls saw a subnormal operand (what DAZ would destroy), and how many
    produced a subnormal result (what FTZ would destroy);
  * per source line, how many times the PTX arm and the FP64-emulation arm
    disagreed on a bit.  This must be zero.  It is the on-card half of the
    identity proof whose CPU half is tests/test_rrtmg_sw_rn_identity.py.

Both arms run for every operation, so the census costs a large constant
factor and is not a performance configuration.  It is enabled only by
compiling the unit with ``-DRSW_WITNESS``, which this tool does and nothing
on the shipped path does.

WHY A DIURNAL SWEEP

The shortwave subnormals live in the direct-beam transmittance chain
``ztdbt[J+1] = MU(zdbt[J], ztdbt[J])``, whose depth is set by tau/mu0.  A
fixture recorded near local noon never reaches the tail.  A real forecast
spends every morning and evening near the terminator, where mu0 -> 0 drives
the optical depth past the exp_tbl 1e-20 floor and two clamped layers
multiply to 1e-40.  So the probe replays the real fixture columns at the
mu0 ladder a day actually walks.

USAGE

    python tools/rrtmg_sw_witness/probe.py --out evidence/sw-witness
    python tools/rrtmg_sw_witness/probe.py --out DIR --cols 130

Requires a CUDA device and the SW oracle fixtures under
tools/rrtmg_wrf461_oracle/sw_fixtures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
FIX = REPO / "tools" / "rrtmg_wrf461_oracle" / "sw_fixtures"
SRC = REPO / "gpuwm" / "core" / "kernels" / "rrtmg_sw.cu"

#: Slots per source line in the device counter array, in kernel order.
SLOTS = ("calls", "subnormal_operand", "subnormal_result",
         "ptx_vs_fp64_mismatch")
WITNESS_LINES = 4096          # must equal RSW_WITNESS_LINES in the kernel

#: The mu0 ladder.  cos(solar zenith) sampled across a mid-latitude day,
#: floored at the smallest mu0 for which the WRF driver still calls SW.
DIURNAL_MU0 = (0.02, 0.05, 0.10, 0.18, 0.28, 0.40, 0.52, 0.64, 0.75, 0.84,
               0.91, 0.96, 0.99)

#: Source functions, by first line, for aggregating the per-line census into
#: the physical quantity each site computes.
FUNCTIONS = (
    ("rsw_log", "glibc logf transcription (layer pressure)"),
    ("rsw_exp2_core", "glibc exp2f core"),
    ("rsw_exp", "glibc expf (co2mult Boltzmann factor)"),
    ("rsw_etbl", "exp_tbl transmittance lookup / small-od polynomial"),
    ("rsw_setcoef_body", "p/T interpolation weights, column gas amounts"),
    ("rsw_spec", "binary-species mixing fractions"),
    ("rsw_acc8", "8-term k-table interpolation"),
    ("rsw_acc4", "4-term k-table interpolation"),
    ("rsw_selffor", "H2O self + foreign continuum"),
    ("rsw_forr", "H2O foreign continuum"),
    ("rsw_taumol_body", "gas optical depth + Rayleigh, 14 bands"),
    ("rsw_sfluxzen_body", "TOA solar flux per g-point"),
    ("rsw_cldprmc_body", "cloud/ice/snow optics: tau, ssa, asy"),
    ("rsw_reftra", "two-stream layer reflectance/transmittance"),
    ("rsw_vrtqdr", "vertical adding-doubling flux sweep"),
    ("rsw_spcvmc_body", "g-point pipeline, direct-beam product chains"),
    ("rsw_spc_accum_body", "ordered g-point flux accumulation"),
    ("rsw_inatm_layers_body", "layer coldry / molecular column amounts"),
    ("rsw_post_body", "flux differences -> heating rates"),
)


def function_starts(source: str) -> list[tuple[int, str, str]]:
    """Locate each FUNCTIONS entry's definition line in the kernel source."""
    lines = source.splitlines()
    want = {n: q for n, q in FUNCTIONS}
    found = []
    for i, text in enumerate(lines, start=1):
        if not text.startswith("__device__"):
            continue
        for name in want:
            if f" {name}(" in text:
                found.append((i, name, want[name]))
                break
    found.sort()
    return found


def owner(line: int, starts) -> tuple[str, str]:
    hit = ("(file scope)", "")
    for start, name, quantity in starts:
        if line >= start:
            hit = (name, quantity)
        else:
            break
    return hit


def load_day_batch(ncopy: int):
    """Every real fixture day column sharing a flag tuple, replayed across
    the diurnal mu0 ladder."""
    d = dict(np.load(FIX / "fixtures_real.npz"))
    per_col = ("play", "plev", "tlay", "tlev", "h2ovmr", "o3vmr", "co2vmr",
               "ch4vmr", "n2ovmr", "o2vmr", "reicmcl", "relqmcl", "resnmcl")
    mcica = ("cldfmcl", "taucmcl", "ssacmcl", "asmcmcl", "fsfcmcl",
             "ciwpmcl", "clwpmcl", "cswpmcl")
    scal = ("tsfc", "asdir", "asdif", "aldir", "aldif", "coszen", "adjes",
            "scon")
    flagk = ("icld", "inflgsw", "iceflgsw", "liqflgsw", "dyofyr", "nlay")
    need = per_col + mcica + scal + flagk

    cases = sorted({k.split("/")[0] for k in d if k.startswith("c")})
    cases = [c for c in cases if all(f"{c}/entry/{k}" in d for k in need)]
    flags, keep = None, []
    for c in cases:
        if float(d[f"{c}/entry/coszen"]) <= 0:
            continue                      # night: the WRF driver skips SW
        f = tuple(int(d[f"{c}/entry/{k}"]) for k in flagk)
        flags = flags or f
        if f == flags:
            keep.append(c)
    if not keep:
        raise SystemExit("no usable day columns in the real fixture set")

    ladder = np.asarray(DIURNAL_MU0[:max(1, min(ncopy, len(DIURNAL_MU0)))],
                        dtype=np.float32)
    reps = max(1, -(-ncopy // len(DIURNAL_MU0)))
    tiles = int(len(ladder) * reps)

    batch = {}
    for k in per_col:
        v = np.stack([d[f"{c}/entry/{k}"] for c in keep]).astype(np.float32)
        batch[k] = np.concatenate([v] * tiles, axis=0)
    for k in mcica:
        v = np.stack([d[f"{c}/entry/{k}"] for c in keep],
                     axis=1).astype(np.float32)
        batch[k] = np.concatenate([v] * tiles, axis=1)
    for k in scal:
        v = np.asarray([d[f"{c}/entry/{k}"] for c in keep], dtype=np.float32)
        batch[k] = np.concatenate([v] * tiles, axis=0)
    batch["coszen"] = np.concatenate(
        [np.full(len(keep), m, np.float32) for m in np.tile(ladder, reps)])
    return batch, int(batch["coszen"].size), flags, len(keep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--cols", type=int, default=len(DIURNAL_MU0),
                    help="ladder repetitions; total columns is a multiple "
                         "of the usable fixture day columns")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the instrument: flush the cross-check's "
                         "reference arm, so mismatches MUST equal the "
                         "subnormal-result count instead of zero")
    args = ap.parse_args()

    import cupy as cp
    from cupy.cuda import compiler as cc
    from gpuwm.core import rrtmg_sw as sw

    tab = sw.tables_from_dump(dict(np.load(FIX / "sw_tables.npz")))
    batch, ncol, flags, ncase = load_day_batch(args.cols)
    icld, inflg, iceflg, liqflg, dyofyr, nlay = flags
    print(f"columns {ncol} ({ncase} fixture day columns x "
          f"{ncol // ncase} mu0 steps), nlay {nlay}, "
          f"coszen {batch['coszen'].min():.2f}..{batch['coszen'].max():.2f}")

    csw = sw.CudaSW(tab)

    # Recompile the SAME source with the witness arm on, and swap the module
    # in.  The driver prepends a generated #define preamble, so __LINE__
    # carries that offset; subtract it to get source lines back.
    source = SRC.read_text(encoding="ascii")
    _packed, defines = sw._pack_cuda_tables(tab)
    code = defines + source
    preamble = defines.count("\n")
    opts = ["-std=c++17", "--ftz=false", "-DRSW_WITNESS"]
    if args.selftest:
        opts.append("-DRSW_WITNESS_SELFTEST")
    ptx, _ = cc.compile_using_nvrtc(code, tuple(opts), None, "rrtmg_sw.cu")
    module = cp.cuda.function.Module()
    module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    csw.module = module

    counters = cp.zeros(WITNESS_LINES * len(SLOTS), dtype=cp.uint64)
    module.get_function("rsw_witness_bind")((1,), (1,), (counters,))
    cp.cuda.runtime.deviceSynchronize()

    res = csw.rrtmg_sw_batched_device(
        ncol, nlay, icld,
        batch["play"], batch["plev"], batch["tlay"], batch["tlev"],
        batch["tsfc"], batch["h2ovmr"], batch["o3vmr"], batch["co2vmr"],
        batch["ch4vmr"], batch["n2ovmr"], batch["o2vmr"],
        batch["asdir"], batch["asdif"], batch["aldir"], batch["aldif"],
        batch["coszen"], batch["adjes"], dyofyr, batch["scon"],
        inflg, iceflg, liqflg,
        batch["cldfmcl"], batch["taucmcl"], batch["ssacmcl"],
        batch["asmcmcl"], batch["fsfcmcl"], batch["ciwpmcl"],
        batch["clwpmcl"], batch["cswpmcl"],
        batch["reicmcl"], batch["relqmcl"], batch["resnmcl"], aer_opt=0)
    cp.cuda.runtime.deviceSynchronize()
    for v in res.values():
        if isinstance(v, cp.ndarray) and not bool(cp.isfinite(v).all()):
            raise SystemExit("witness run produced a non-finite SW output")

    w = cp.asnumpy(counters).reshape(WITNESS_LINES, len(SLOTS))
    starts = function_starts(source)
    srclines = source.splitlines()

    rows, agg = [], {}
    for raw in range(WITNESS_LINES):
        if w[raw, 0] == 0:
            continue
        line = raw - preamble
        name, quantity = owner(line, starts)
        text = srclines[line - 1].strip() if 1 <= line <= len(srclines) else "?"
        row = {"line": line, "function": name, "quantity": quantity,
               "src": text}
        row.update({s: int(w[raw, i]) for i, s in enumerate(SLOTS)})
        rows.append(row)
        a = agg.setdefault(name, {"quantity": quantity, "sites": 0,
                                  **{s: 0 for s in SLOTS}})
        a["sites"] += 1
        for s in SLOTS:
            a[s] += row[s]

    totals = {s: int(w[:, i].sum()) for i, s in enumerate(SLOTS)}
    if args.selftest:
        # The damaged reference flushes exactly the subnormal results, so the
        # mismatch count must equal the subnormal-result count -- not merely
        # be nonzero.  Anything else means the counter is not measuring what
        # it claims.
        ok = (totals["ptx_vs_fp64_mismatch"] == totals["subnormal_result"]
              and totals["subnormal_result"] > 0)
        verdict = "SELFTEST-OK" if ok else "SELFTEST-FAILED"
    else:
        verdict = ("IDENTICAL" if totals["ptx_vs_fp64_mismatch"] == 0
                   else "DIVERGED")
    print(f"\nsites executed {len(rows)}   calls {totals['calls']:,}")
    print(f"subnormal operand {totals['subnormal_operand']:,}   "
          f"subnormal result {totals['subnormal_result']:,}")
    print(f"PTX vs FP64-emulation mismatches {totals['ptx_vs_fp64_mismatch']:,}"
          f"  -> {verdict}\n")

    hdr = f"{'function':<24}{'sites':>6}{'calls':>15}{'sub_operand':>13}"
    print(hdr + f"{'sub_result':>12}   verdict")
    print("-" * 96)
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["calls"]):
        carries = a["subnormal_operand"] or a["subnormal_result"]
        print(f"{name:<24}{a['sites']:>6}{a['calls']:>15,}"
              f"{a['subnormal_operand']:>13,}{a['subnormal_result']:>12,}"
              f"   {'SUBNORMALS WITNESSED' if carries else 'clean'}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "gpuwm.rrtmg-sw-witness/v1",
               "columns": ncol, "nlay": int(nlay),
               "mu0_ladder": list(DIURNAL_MU0[:args.cols]),
               "preamble_lines": preamble,
               "totals": totals, "verdict": verdict,
               "by_function": agg, "by_line": rows}
    (args.out / "sw-witness.json").write_text(json.dumps(payload, indent=1))
    print("\nwrote", args.out / "sw-witness.json")
    if args.selftest:
        return 0 if verdict == "SELFTEST-OK" else 1
    return 0 if totals["ptx_vs_fp64_mismatch"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
