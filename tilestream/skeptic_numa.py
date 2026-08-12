"""Does the NUMA control have teeth?  Measured on a genuinely dual-socket box.

The mgstream probe ran its affinity control on a SINGLE-socket EPYC 7713 and
concluded: "Pinning changes nothing -- and it cannot, because the NUMA
artefact the brief warned about is impossible here."  That is correct for that
box and says nothing about the 8-GPU deployment the report reasons about,
which will almost certainly be dual-socket.

This box (EPYC dual socket, GPU on NUMA node 1) is the case the probe could
not test.  Three arms, same code path as ``bw_probe.measure``:

  near     worker pinned to node 1 -- the GPU's own socket
  far      worker pinned to node 0 -- pinned host pages land across the
           inter-socket link, which the DMA must then cross
  unbound  no affinity at all -- what an unaware harness gets

If ``far`` is materially below ``near``, then an unbound streaming run on a
dual-socket host is measuring the inter-socket interconnect, and per-GPU
transport on the deployment box is not what a single-socket measurement says.
"""

from __future__ import annotations

import argparse
import json
import statistics

from tilestream import bw_probe

NODE0 = list(range(0, 64)) + list(range(128, 192))
NODE1 = list(range(64, 128)) + list(range(192, 255))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--chunk-mib", type=int, default=256)
    ap.add_argument("--span-mib", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--direction", default="h2d")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    d = args.device
    info = bw_probe.device_info(d)
    info["pcie"] = bw_probe.pcie_link(info["pci_bus_id"])
    print(f"{info['name']}  {info['pci_bus_id']}  link "
          f"x{info['pcie'].get('current_link_width')} "
          f"(max x{info['pcie'].get('max_link_width')})")

    arms = {"near_node1": NODE1, "far_node0": NODE0, "unbound": None}
    out = {"gpu": info, "direction": args.direction, "arms": {}}
    for name, cores in arms.items():
        pin = None if cores is None else {d: cores}
        vals = []
        for _ in range(args.reps):
            r = bw_probe.measure([d], seconds=args.seconds,
                                 chunk_mib=args.chunk_mib,
                                 host_span_mib=args.span_mib,
                                 queue=4, direction=args.direction, pin=pin)
            if "error" in r:
                print(f"  {name}: ERROR {r['error']}")
                vals = []
                break
            vals.append(r["aggregate_gbs"])
        if not vals:
            out["arms"][name] = {"error": True}
            continue
        med = statistics.median(vals)
        spread = (max(vals) - min(vals)) / med * 100
        out["arms"][name] = {"gbs_median": med, "spread_pct": spread,
                             "runs": vals}
        print(f"  {name:11s} {med:7.2f} GB/s   spread {spread:5.1f}%")

    a = out["arms"]
    if "gbs_median" in a.get("near_node1", {}) and "gbs_median" in a.get("far_node0", {}):
        ratio = a["far_node0"]["gbs_median"] / a["near_node1"]["gbs_median"]
        out["far_over_near"] = ratio
        print(f"\nfar/near = {ratio:.3f}")
        if ratio < 0.9:
            print("VERDICT: the NUMA control HAS TEETH on a dual-socket host; "
                  "an unbound streaming run can measure the inter-socket link.")
        else:
            print("VERDICT: no material far-socket penalty on this box.")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
