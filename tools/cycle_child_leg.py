"""One cycle's worth of child work: plan in, dormant domains out, legs run.

This is the child half of one cycle boundary, and nothing else.  It reads
the placement plan the spine computed from the analysis, renders the slot
pool as ``[[domain]]`` blocks the unchanged 2.3.0 spawn machinery can
execute, advances every live child through the leg, and writes one
receipt per child.

**The model advance is injected.**  ``run_children`` takes an ``advance``
callable, so the whole file is exercised without a GPU and the shipped
caller hands it the forecast driver's own leg runner.  That is not a
testing convenience bolted on: it is what lets the child-may-die rule be
proved rather than asserted.

**A child may die; the cycle may not.**  A non-finite state or a runner
refusal comes back as ``DIVERGED`` with the field, the cell index and the
value, and the child's slot is released.  Nothing is re-raised.  A parent
refusal is not this file's business and is not caught anywhere in it.

HONESTY: the child plan is a placement decision, not a skill claim.  No
case names, no station names anywhere in this file (standing owner rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gpuwm.cycle.children import (ChildSlot, SlotPool, emit_dormant_domains,
                                  plan_children, run_child_leg)
from gpuwm.cycle.contracts import TRANSITION_SCHEMA, CycleRefusal
from gpuwm.cycle.placement import (PlacementRequest, ResolvedPlacement,
                                   parent_geometry_from_fields)

#: The receipt this leg writes.  Read it with a schema check, never by
#: listing the directory -- a half-written leg looks like a finished one
#: to a directory listing and like nothing at all to a schema check.
LEG_SCHEMA = TRANSITION_SCHEMA


def slots_from_payload(payload) -> list[ChildSlot]:
    return [ChildSlot(grid_id=int(item["grid_id"]), nx=int(item["nx"]),
                      ny=int(item["ny"]), dx_m=float(item["dx_m"]),
                      dt_seconds=float(item["dt_seconds"]),
                      parent_grid_ratio=int(item["parent_grid_ratio"]))
            for item in payload]


def requests_from_payload(payload) -> list[PlacementRequest]:
    return [PlacementRequest(
        lat=float(item["lat"]), lon=float(item["lon"]),
        dx_m=float(item["dx_m"]), nx=int(item["nx"]), ny=int(item["ny"]),
        source=str(item["source"]), strength=float(item["strength"]),
        evidence=dict(item.get("evidence", {}))) for item in payload]


def run_children(*, cycle_index: int, pool: SlotPool, requests,
                 previous_children, parent_geometry, advance,
                 retire_below_strength: float, min_separation_km: float,
                 allow_clamp: bool = False) -> dict:
    """Plan the boundary, advance every child, collect the receipts.

    Returns the transition payload: the dormant-domain block the loader
    needs, every placement decision with its geographic ask AND its
    resolved indices, and one leg record per child.
    """

    plan = plan_children(
        cycle_index=int(cycle_index), pool=pool, requests=requests,
        previous_children=previous_children,
        retire_below_strength=float(retire_below_strength),
        min_separation_km=float(min_separation_km),
        parent_geometry=parent_geometry, allow_clamp=bool(allow_clamp))

    legs = []
    for record in plan:
        if record["state"] not in ("LIVE", "PLANNED"):
            # RETIRED and REFUSED children have no leg to run; they are
            # still in the receipt, because a decision that leaves no
            # trace is a decision nobody can audit.
            continue
        placement = _resolved_from_payload(record["placement"])
        leg = run_child_leg(grid_id=record["grid_id"], placement=placement,
                            cycle_index=int(cycle_index), advance=advance)
        if leg["state"] == "DIVERGED":
            # The despawn on divergence: one slot, not the run.
            pool.release(record["grid_id"], reason="diverged")
        legs.append(leg)

    return {
        "schema": LEG_SCHEMA,
        "cycle_index": int(cycle_index),
        "dormant_domains": emit_dormant_domains(
            pool, template_domain={"parent_id": 1}),
        "placements": plan,
        "legs": legs,
        "slots_free_after": list(pool.free()),
        "slots_released": pool.released(),
    }


def _resolved_from_payload(payload) -> ResolvedPlacement:
    from gpuwm.cycle.children import _placement_from_payload

    return _placement_from_payload(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.cycle_child_leg",
        description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True,
                        help="the boundary's plan: slots, requests, "
                             "previous children and the parent's lat/lon "
                             "fields")
    parser.add_argument("--out", type=Path, required=True,
                        help="where the transition receipt is written")
    parser.add_argument("--retire-below-strength", type=float, required=True,
                        help="a live child whose signal falls below this "
                             "is RETIRED and its slot returns to the pool "
                             "(argument, never a default: the units are "
                             "the trigger field's)")
    parser.add_argument("--min-separation-km", type=float, default=20.0,
                        help="two children closer than this are one storm; "
                             "the weaker request is refused by name")
    parser.add_argument("--allow-placement-clamp", action="store_true",
                        help="clamp a footprint that leaves the parent "
                             "instead of refusing it. A WORKAROUND, and "
                             "receipted as one: a clamped nest keeps "
                             "integrating while it has stopped following "
                             "its storm, which looks exactly like a "
                             "working nest")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    pool = SlotPool(slots_from_payload(payload["slots"]))
    parent = parent_geometry_from_fields(
        lat_field=payload["parent"]["lat"], lon_field=payload["parent"]["lon"],
        dx_m=float(payload["parent"]["dx_m"]))

    def advance(**kwargs):
        # Without a driver this front door PLANS and refuses to pretend it
        # integrated.  A tool that returned a healthy-looking receipt for
        # a leg it never ran is the failure this program has paid for.
        raise CycleRefusal(
            "no model driver was supplied to this front door",
            grid_id=kwargs.get("grid_id"),
            cycle_index=kwargs.get("cycle_index"),
            remedy="call run_children() with advance=<driver leg runner>")

    report = run_children(
        cycle_index=int(payload.get("cycle_index", 0)), pool=pool,
        requests=requests_from_payload(payload.get("requests", [])),
        previous_children=payload.get("previous_children", []),
        parent_geometry=parent, advance=advance,
        retire_below_strength=args.retire_below_strength,
        min_separation_km=args.min_separation_km,
        allow_clamp=args.allow_placement_clamp)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True),
                              encoding="utf-8")
    print(f"cycle {report['cycle_index']}: "
          f"{sum(1 for r in report['placements'] if r['state'] == 'LIVE')} "
          f"live, "
          f"{sum(1 for r in report['placements'] if r['state'] == 'PLANNED')} "
          f"planned, "
          f"{sum(1 for r in report['placements'] if r['state'] == 'RETIRED')} "
          f"retired, "
          f"{sum(1 for r in report['placements'] if r['state'] == 'REFUSED')} "
          f"refused -> {args.out}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
