"""Assemble ``output-scaling.json`` -- the plot data for the OUTPUT lane.

Reads the raw blocks ``tilestream.bench_ooc_output`` dumped, folds in the
constants that lane MEASURED, works out the projections, and writes one file
that :mod:`tilestream.make_plots` can render without recomputing anything.

Every value carries a ``kind``: MEASURED (observed on this box), DERIVED
(arithmetic on measured values) or ESTIMATED (a measured value from another
configuration carried across).

    python -m tilestream.make_output_json
"""

from __future__ import annotations

import json
from pathlib import Path

JSON = Path(__file__).with_name("output-scaling.json")

NZ = 49

# --- MEASURED constants -----------------------------------------------------
#: netCDF4/HDF5 container overhead over the raw payload.  MEASURED at 1024^2:
#: file 1734.83 MB / payload 1673.93 MB.  Phase 1 got the same 1.0364.
CONTAINER = 1734.827118 / 1673.9291

#: History frame payload, bytes per cell, by physics rung.  MEASURED at 96^2
#: and 192^2 and Richardson-extrapolated to remove the staggered +1 columns
#: (B(n) = Binf + c/n).  The dry value reproduces the 1024^2 sweep's
#: 1673.93 MB / 51.38 Mcell = 32.57 exactly.
BYTES_PER_CELL = {
    "dry": 32.57,
    "mp10 Morrison": 78.20,
    "full(real74)+KF": 79.67,
    "full+MYNN+Noah-MP": 133.14,
}
FRAME_FIELDS = {"dry": 15, "mp10 Morrison": 46,
                "full(real74)+KF": 55, "full+MYNN+Noah-MP": 122}

#: Sustained sequential write through fsync, MEASURED with dd at 4 / 16 / 32
#: GiB: 1.48 / 1.40 / 1.41 GB/s.  Flat, so it is a device rate and not the
#: page cache -- 32 GiB is well past the dirty limit (20% of 93.9 GiB).
DISK_CEILING_GBS = 1.41

#: Solver step, nanoseconds per cell.  MEASURED here at 1448^2 (341.63 ms,
#: 4% spread, the only size the sharing lane left alone); Phase 1 measured
#: 3.218 at 1024^2 and RESULTS.md 3.712 at 1950^2 monolithic.  RESULTS.md
#: documents a 3% cross-process reproducibility floor on figures like this.
STEP_NS_PER_CELL = 3.325

#: 3276^2 out-of-core, from RESULTS.md row 4: 2043.37 ms/step, 3.886 ns/cell.
BIG_STEP_MS = 2043.37
BIG_NS_PER_CELL = 3.886


def fit(xs, ys):
    """Least squares y = a + b*x, returning (a, b, max relative residual)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    res = max(abs((a + b * x) - y) / y for x, y in zip(xs, ys))
    return a, b, res


def main() -> int:
    blob = json.loads(JSON.read_text())
    blocks = blob["blocks"]
    e2e = blocks["endtoend"]
    tr = {r["nx"]: r for r in blocks["transfer"]}

    # --- per-cell costs, fitted over the measured sweep ---------------------
    cells = [r["cells"] / 1e6 for r in e2e]                     # Mcell
    write = [r["common_write_ms"] for r in e2e]
    a_w, b_w, res_w = fit(cells, write)

    # The frame write is better described by throughput than by a fit: the
    # fit's intercept comes out NEGATIVE on this data (the 1448^2 point is
    # 22% above the line drawn by the other three), which is unphysical and
    # says the write is throughput-limited with size-dependent noise, not
    # fixed-cost-plus-linear.  So the projections use the MEASURED effective
    # throughput and quote its range instead of pretending the fit is good.
    thr = [(r["file_MB"] / 1e3) / (r["common_write_ms"] / 1e3) for r in e2e]
    thr_sorted = sorted(thr)
    thr_med = 0.5 * (thr_sorted[1] + thr_sorted[2])

    n_big = 1448
    per_cell = {
        "d2h": tr[n_big]["d2h_ms"] * 1e6 / tr[n_big]["cells"],
        "derive_device": tr[n_big]["derive_dev_ms"] * 1e6 / tr[n_big]["cells"],
        "host_derive": tr[n_big]["host_derive_ms"] * 1e6 / tr[n_big]["cells"],
        "snapshot": tr[n_big]["snapshot_ms"] * 1e6 / tr[n_big]["cells"],
        "step": STEP_NS_PER_CELL,
    }
    per_cell["mono_exposed"] = per_cell["d2h"] + per_cell["derive_device"]
    per_cell["ooc_exposed"] = per_cell["snapshot"]

    # --- the CM1 figure: timesteps per second vs domain size ---------------
    # Model, taken from the code's own structure and CHECKED against it:
    #   the netCDF write runs on the writer thread and overlaps (Phase 1
    #   measured 97% overlap efficiency), so it is not an addend unless it
    #   exceeds the interval; the staging is serialised into the GPU timeline
    #   by submit()'s two wait_events, so it IS an addend.
    #     wall(C steps) = max(C*step + exposed, write)
    cadences = [1, 10, 60, 240]
    curves: dict[str, list] = {"no output": [], "sizes": []}
    for c in cadences:
        curves[f"monolithic C={c}"] = []
        curves[f"streamed C={c}"] = []
    sizes = [512, 724, 1024, 1448, 1950, 2400, 3276]
    for n in sizes:
        cell = n * n * NZ
        step = per_cell["step"] * cell / 1e9                    # s
        frame_gb = cell * BYTES_PER_CELL["dry"] * CONTAINER / 1e9
        wr = frame_gb / thr_med                                  # s
        curves["sizes"].append(n)
        curves["no output"].append(1.0 / step)
        for c in cadences:
            for tag, exp in (("monolithic", per_cell["mono_exposed"]),
                             ("streamed", per_cell["ooc_exposed"])):
                exposed = exp * cell / 1e9
                wall = max(c * step + exposed, wr)
                curves[f"{tag} C={c}"].append(c / wall)

    # --- the frame breakdown, monolithic vs streamed (1024^2, MEASURED) ----
    r = next(x for x in e2e if x["nx"] == 1024)
    t = tr[1024]
    enc = 0.5 * (r["mono_encode_ms"] + r["ooc_encode_ms"])
    disk = r["common_write_ms"] - enc
    breakdown = {
        "nx": 1024, "cells": r["cells"], "payload_MB": r["payload_MB"],
        "file_MB": r["file_MB"],
        "monolithic": {"transfer D2H": t["d2h_ms"],
                       "derive (device)": t["derive_dev_ms"],
                       "netCDF encode": enc,
                       "disk (close+fsync)": disk},
        "streamed": {"snapshot (host copy)": t["snapshot_ms"],
                     "derive (host)": t["host_derive_ms"],
                     "netCDF encode": enc,
                     "disk (close+fsync)": disk},
    }
    for k in ("monolithic", "streamed"):
        breakdown[k + "_total"] = sum(breakdown[k].values())

    # --- projections -------------------------------------------------------
    def project(label, n, rung, step_s, note):
        cell = n * n * NZ
        payload = cell * BYTES_PER_CELL[rung] / 1e9
        onfile = payload * CONTAINER
        secs = onfile / thr_med
        return {
            "label": label, "nx": n, "rung": rung, "fields": FRAME_FIELDS[rung],
            "Mcell": cell / 1e6, "frame_GB_payload": payload,
            "frame_GB_on_disk": onfile,
            "frame_seconds": secs,
            "frame_seconds_range": [onfile / max(thr), onfile / min(thr)],
            "step_seconds": step_s,
            "frame_in_timesteps": secs / step_s,
            "frames_24h_hourly": 24,
            "disk_24h_hourly_GB": 24 * onfile,
            "steps_24h_dt3": 86400 / 3.0,
            "wall_24h_hours": 86400 / 3.0 * step_s / 3600.0,
            "output_pct_of_wall_hourly":
                100.0 * 24 * secs / (86400 / 3.0 * step_s),
            "note": note,
        }

    big_step = BIG_STEP_MS / 1e3
    projections = [
        project("3276^2 dry, out-of-core", 3276, "dry", big_step,
                "step time MEASURED in RESULTS.md row 4 (2043.37 ms, "
                "3.886 ns/cell); cannot be run monolithically at all"),
        project("1950^2 dry, largest resident", 1950, "dry",
                1950 * 1950 * NZ * 3.712 / 1e9,
                "step time from RESULTS.md row 1 (691.70 ms, 3.712 ns/cell)"),
        project("1873^2 full physics + rings", 1873, "full+MYNN+Noah-MP",
                1873 * 1873 * NZ * 19.0 / 1e9,
                "ESTIMATED step: RESULTS.md section 6 uses ~19 ns/cell of "
                "compute for a physics rung; no physics step time has been "
                "measured on this box, so this row's TIME is estimated while "
                "its BYTES are measured"),
        project("1873^2, moisture only (mp10)", 1873, "mp10 Morrison",
                1873 * 1873 * NZ * 19.0 / 1e9,
                "same estimated step; shows what dropping the MYNN/Noah-MP "
                "diagnostics saves on the disk bill"),
    ]

    blob["meta"] = {
        "machine": "RTX 5090, PCIe 5.0 x16, 93.9 GiB host RAM, WSL2",
        "filesystem": "/dev/sdd ext4 on the WSL2 VHD (NOT /mnt/c DrvFs)",
        "disk_ceiling_GBs": DISK_CEILING_GBS,
        "disk_ceiling_kind": "MEASURED, dd 4/16/32 GiB conv=fsync -> "
                             "1.48/1.40/1.41 GB/s, flat past the dirty limit",
        "frame_throughput_GBs": thr_med,
        "frame_throughput_range": [min(thr), max(thr)],
        "frame_throughput_kind": "MEASURED end to end incl. netCDF encode "
                                 "and fsync; 77% of the raw dd ceiling",
        "container_overhead": CONTAINER,
        "nz": NZ,
        "caveat_gpu": "another lane shared this GPU throughout; GPU-side "
                      "figures are MINIMA over many reps (contention only "
                      "adds), and solver step times at 512/724/1024 were "
                      "discarded as unusable",
    }
    blob["per_cell_ns"] = per_cell
    blob["write_fit"] = {"intercept_ms": a_w, "slope_ns_per_cell": b_w,
                         "max_residual": res_w,
                         "kind": "DERIVED, and REJECTED for projection: the "
                                 "intercept is negative, which is unphysical"}
    blob["throughput_GBs"] = thr
    blob["cadence_curves"] = curves
    blob["breakdown"] = breakdown
    blob["projections"] = projections
    blob["bytes_per_cell"] = BYTES_PER_CELL
    # write_bytes, not write_text: on Windows the text-mode writer turns
    # every "\n" into "\r\n", and tilestream/output-scaling.json is a
    # tracked file.
    JSON.write_bytes(json.dumps(blob, indent=1).encode("utf-8"))

    print(f"wrote {JSON}")
    print(f"\nframe throughput MEASURED {thr_med:.3f} GB/s "
          f"(range {min(thr):.3f}-{max(thr):.3f}), disk ceiling "
          f"{DISK_CEILING_GBS} GB/s")
    print("\nPER-CELL COSTS (ns/cell, MEASURED at 1448^2):")
    for k, v in per_cell.items():
        print(f"   {k:16s} {v:7.3f}")
    print(f"\n   monolithic exposes {per_cell['mono_exposed']:.3f} ns/cell, "
          f"streamed exposes {per_cell['ooc_exposed']:.3f} -> streaming is "
          f"{per_cell['ooc_exposed']/per_cell['mono_exposed']:.2f}x WORSE "
          f"on the non-overlappable part")
    print("\nPROJECTIONS")
    print(f"  {'domain':32s} {'flds':>4s} {'GB/frame':>9s} {'s/frame':>8s} "
          f"{'=steps':>7s} {'24h disk':>9s} {'out %':>6s}")
    for p in projections:
        print(f"  {p['label']:32s} {p['fields']:4d} "
              f"{p['frame_GB_on_disk']:9.2f} {p['frame_seconds']:8.1f} "
              f"{p['frame_in_timesteps']:7.1f} "
              f"{p['disk_24h_hourly_GB']:8.0f}G "
              f"{p['output_pct_of_wall_hourly']:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
