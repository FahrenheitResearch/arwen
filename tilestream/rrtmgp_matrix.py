"""Drive the RRTMGP reclamation matrix: one fresh subprocess per row.

Prints the two peaks, the workspace size, the residency between firings, the
two step costs, the amortised cost at the production cadence, and the state
digest for every (mode, column_chunk) pair -- and then evaluates the peak
model of :mod:`tilestream.rrtmgp_bench` on the measured numbers so the
saving is arithmetic on this run's data rather than a remembered claim.

Every row prints its own radiation fire count.  A row whose count is zero is
not a radiation measurement and is marked as such rather than averaged in.
"""

from __future__ import annotations

import argparse
import json
import sys

from tilestream.rrtmgp_bench import run_trial_subprocess

MIB = 1 << 20


def _row(res: dict) -> str:
    if res.get("error"):
        return f"  ERROR rc={res['returncode']}  {res['stderr'][-300:]}"
    w = res["workspace_bytes"] / MIB
    return (f"  {res['mode']:<11}{str(res['column_chunk'] or '-'):>7}"
            f"{w:>10.1f}"
            f"{(res['workspace_resident_between_firings'] or 0) / MIB:>10.1f}"
            f"{res['peak_ordinary_bytes'] / MIB:>11.1f}"
            f"{res['peak_radiation_bytes'] / MIB:>11.1f}"
            f"{res['ordinary_step_ms']:>10.2f}"
            f"{res['radiation_step_ms']:>11.1f}"
            f"{res['radiation_over_ordinary']:>8.1f}x"
            f"{res['amortised_ms_per_step']:>10.3f}"
            f"{res['radiation_firings']:>6}"
            f"  {res['digest'][:12]}")


HEADER = (f"  {'mode':<11}{'chunk':>7}{'W MiB':>10}{'resident':>10}"
          f"{'peak_ord':>11}{'peak_rad':>11}{'ord ms':>10}{'rad ms':>11}"
          f"{'ratio':>9}{'amort ms':>10}{'fires':>6}  digest")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rung", default="full")
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=49)
    p.add_argument("--chunks", type=int, nargs="+",
                   default=[3125, 1024, 512, 256])
    p.add_argument("--modes", nargs="+",
                   default=["persistent", "lazy", "tight", "tight+lazy"])
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    plan = [("none", None)]
    for chunk in args.chunks:
        for mode in args.modes:
            plan.append((mode, chunk))

    print(f"RRTMGP reclamation matrix   rung={args.rung}  "
          f"{args.nx}x{args.ny}x{args.nz} = "
          f"{args.nx * args.ny * args.nz:,} cells, "
          f"{args.nx * args.ny:,} columns")
    print()
    print(HEADER)
    print("  " + "-" * (len(HEADER) + 8))
    rows = []
    for mode, chunk in plan:
        res = run_trial_subprocess(
            rung=args.rung, nx=args.nx, ny=args.ny, nz=args.nz, mode=mode,
            column_chunk=chunk, fire=True, steps=args.steps,
            cycles=args.cycles)
        rows.append(res)
        print(_row(res))
        sys.stdout.flush()

    print()
    print("PEAK MODEL, evaluated on the rows above")
    print("  peak    = max(peak_ordinary, peak_radiation), measured")
    print("  saving  = shipped-persistent peak at this chunk MINUS this peak")
    print("  A saving of 0.0 means the reclamation moved bytes the device's")
    print("  high-water mark never saw -- radiation was the busiest step")
    print("  either way.  That is a result, not a rounding error.")
    print()
    print(f"  {'chunk':>7}  {'mode':<11}{'W MiB':>10}{'shipped peak':>13}"
          f"{'this peak':>13}{'saving MiB':>13}  digest agrees")
    by_chunk: dict[int, dict] = {}
    for res in rows:
        if res.get("error") or res["mode"] == "none":
            continue
        by_chunk.setdefault(res["column_chunk"], {})[res["mode"]] = res
    for chunk in sorted(by_chunk, reverse=True):
        pair = by_chunk[chunk]
        base = pair.get("persistent")
        if base is None:
            continue
        base_peak = max(base["peak_ordinary_bytes"],
                        base["peak_radiation_bytes"])
        for mode in ("lazy", "tight", "tight+lazy"):
            res = pair.get(mode)
            if res is None:
                continue
            peak = max(res["peak_ordinary_bytes"],
                       res["peak_radiation_bytes"])
            agree = base["digest"] == res["digest"]
            print(f"  {chunk:>7}  {mode:<11}"
                  f"{res['workspace_bytes'] / MIB:>10.1f}"
                  f"{base_peak / MIB:>13.1f}{peak / MIB:>13.1f}"
                  f"{(base_peak - peak) / MIB:>13.1f}"
                  f"  {'YES' if agree else 'NO -- ANSWERS CHANGED'}")

    digests = {r["digest"] for r in rows if not r.get("error")}
    print()
    print(f"  {len(digests)} distinct digest(s) over {len(rows)} rows: "
          + ("ALL BIT-EXACT" if len(digests) == 1
             else "DIVERGENT -- " + ", ".join(d[:12] for d in digests)))
    zero = [r for r in rows
            if not r.get("error") and not r["radiation_firings"]]
    if zero:
        print(f"  !! {len(zero)} row(s) never fired radiation; not a "
              "radiation measurement")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
