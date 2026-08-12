"""The part of RRTMGP's fixed cost that no pool counter can see.

At 128x128x49 full physics the CuPy pool holds ~570 MiB and the process holds
2.6 GiB of the card.  The two-gigabyte difference is the CUDA context plus
every NVRTC module image the run has compiled, and it is charged to the
DEVICE, once per process, whether or not a single tile buffer exists.  A
least-squares fit of peak against domain size books all of it as intercept,
which is exactly how ``1.90 GiB FIXED + 677 B/cell`` came to be measured.

So "where does the 1.63 GiB go" cannot be answered from the workspace
inventory alone.  This module walks the physics ladder in FRESH SUBPROCESSES
-- one rung per process, because module images are never unloaded -- and
reports the non-pool device bytes each rung leaves behind after one complete
step.  The marginal between the rung with radiation off and the same rung
with radiation on is RRTMGP's own share, and it is compiled code, not
scratch.

The measurement is ``nvidia-smi``'s per-pid used_memory minus the pool total
when the driver reports pid rows, and device_used minus pool_total otherwise;
the row says which, because the fallback is only this process's share if the
card is idle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

MIB = 1 << 20

#: One rung per row.  ``full-norad`` is the control that isolates radiation:
#: identical to ``full`` in every other selector, so the marginal between
#: them is RRTMGP and nothing else.
_M = dict(moist=True, mp_physics=10, ztop=20000.0)
LADDER = {
    "import-only": None,
    "dry": dict(),
    "mp10": dict(_M),
    "+km_opt4": dict(_M, km_opt=4),
    "+sfclay": dict(_M, km_opt=4, sf_sfclay_physics=91),
    "+YSU": dict(_M, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
                 bldt=0.0),
    "+Noah": dict(_M, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
                  bldt=0.0, sf_surface_physics=2),
    "+RRTMGP": dict(_M, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
                    bldt=0.0, sf_surface_physics=2, ra_sw_physics=4,
                    ra_lw_physics=4, radt_minutes=12.0),
    "full": dict(_M, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
                 bldt=0.0, sf_surface_physics=2, ra_sw_physics=4,
                 ra_lw_physics=4, radt_minutes=12.0, cu_physics=1,
                 cudt_minutes=5.0),
    #: The radiation-off twin of ``full``: identical in every other
    #: selector, so ``full`` minus this is RRTMGP and nothing else.  It is
    #: the control for the one-at-a-time ladder, which can only attribute a
    #: jump to whatever rung happened to add it FIRST.
    "full-norad": dict(_M, km_opt=4, sf_sfclay_physics=91,
                       bl_pbl_physics=1, bldt=0.0, sf_surface_physics=2,
                       ra_sw_physics=0, ra_lw_physics=0, cu_physics=1,
                       cudt_minutes=5.0),
}


def _nonpool_now() -> tuple[int, str]:
    import os

    import cupy as cp

    pid = os.getpid()
    pool_total = int(cp.get_default_memory_pool().total_bytes())
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:                                          # noqa: BLE001
        out = ""
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(float(parts[1]) * MIB) - pool_total, "per-pid"
    free, total = cp.cuda.runtime.memGetInfo()
    return int(total - free) - pool_total, "device-minus-pool (idle card)"


def rung(name: str, nx: int, ny: int, nz: int) -> dict:
    import cupy as cp

    selectors = LADDER[name]
    # The CUDA context is created by the first device touch, not by import.
    cp.zeros(1)
    cp.cuda.runtime.deviceSynchronize()
    ctx_bytes, source = _nonpool_now()
    if selectors is None:
        return {"rung": name, "nonpool_bytes": ctx_bytes, "source": source,
                "pool_bytes": int(cp.get_default_memory_pool().total_bytes()),
                "radiation_firings": 0, "digest": None}

    from tilestream import harness, physics_inventory as physinv

    cfg = harness.make_config(nx, ny, nz, **selectors)
    state, driver = physinv.default_builder(cfg)
    before = dict(getattr(driver, "call_counts", {}) or {})
    # Two steps: step 1 compiles and fires every due scheme, step 2 proves
    # the run continues.  Radiation is due at t=0, so a rung with radiation
    # on fires it here -- and the count is reported, never assumed.
    harness.run_steps(state, cfg, 2)
    cp.cuda.runtime.deviceSynchronize()
    counts = {k: int(v) - int(before.get(k, 0))
              for k, v in (getattr(driver, "call_counts", {}) or {}).items()
              if int(v) - int(before.get(k, 0))}
    nonpool, source = _nonpool_now()
    return {
        "rung": name, "nx": nx, "ny": ny, "nz": nz,
        "nonpool_bytes": nonpool,
        "context_bytes": ctx_bytes,
        "modules_bytes": nonpool - ctx_bytes,
        "source": source,
        "pool_bytes": int(cp.get_default_memory_pool().total_bytes()),
        "radiation_firings": int(counts.get("radiation", 0)),
        "cadence_firings": counts,
        "digest": harness.hash_state(state),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("rung")
    one.add_argument("name", choices=sorted(LADDER))
    one.add_argument("--nx", type=int, default=96)
    one.add_argument("--ny", type=int, default=80)
    one.add_argument("--nz", type=int, default=49)
    allp = sub.add_parser("ladder")
    allp.add_argument("--nx", type=int, default=96)
    allp.add_argument("--ny", type=int, default=80)
    allp.add_argument("--nz", type=int, default=49)
    allp.add_argument("--json", default=None)
    args = p.parse_args(argv)

    if args.cmd == "rung":
        print("@@JSON@@" + json.dumps(
            rung(args.name, args.nx, args.ny, args.nz)))
        return 0

    rows = []
    order = ["import-only", "dry", "mp10", "+km_opt4", "+sfclay", "+YSU",
             "+Noah", "+RRTMGP", "full", "full-norad"]
    for name in order:
        proc = subprocess.run(
            [sys.executable, "-m", "tilestream.rrtmgp_nonpool", "rung", name,
             "--nx", str(args.nx), "--ny", str(args.ny),
             "--nz", str(args.nz)],
            capture_output=True, text=True)
        row = None
        for line in proc.stdout.splitlines():
            if line.startswith("@@JSON@@"):
                row = json.loads(line[len("@@JSON@@"):])
        if row is None:
            print(f"  {name}: FAILED\n{proc.stderr[-1500:]}")
            continue
        rows.append(row)

    print(f"NON-POOL DEVICE BYTES BY RUNG   {args.nx}x{args.ny}x{args.nz}"
          f"   (fresh process per rung)")
    print()
    print(f"  {'rung':<13}{'context':>10}{'modules':>10}{'nonpool':>10}"
          f"{'pool':>10}{'marginal nonpool':>18}{'rad':>5}   source")
    print("  " + "-" * 84)
    prev = None
    for row in rows:
        marginal = ("" if prev is None
                    else f"{(row['nonpool_bytes'] - prev) / MIB:+.1f} MiB")
        print(f"  {row['rung']:<13}"
              f"{row.get('context_bytes', row['nonpool_bytes']) / MIB:>10.1f}"
              f"{row.get('modules_bytes', 0) / MIB:>10.1f}"
              f"{row['nonpool_bytes'] / MIB:>10.1f}"
              f"{row['pool_bytes'] / MIB:>10.1f}"
              f"{marginal:>18}"
              f"{row['radiation_firings']:>5}   {row['source']}")
        prev = row["nonpool_bytes"]

    by = {r["rung"]: r for r in rows}
    if "full" in by and "full-norad" in by:
        d = by["full"]["nonpool_bytes"] - by["full-norad"]["nonpool_bytes"]
        print()
        print(f"  RRTMGP's NON-POOL share (full minus full-norad, identical "
              f"in every other selector):")
        print(f"    {d / MIB:,.1f} MiB = {d / (1 << 30):.3f} GiB of compiled "
              "module image, paid once per process")
        if not by["full"]["radiation_firings"]:
            print("    !! the 'full' rung never fired radiation; this "
                  "marginal is not a radiation measurement")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
