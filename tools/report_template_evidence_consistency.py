"""Enumerate every registered template under each candidate composition axis.

The receipt this writes is the measurement the composition rule is chosen
over, not a restatement of a rule already chosen.  It evaluates all three
candidate axes side by side against the shipped registry and reports, per
template, the label rank, the ceiling each axis computes, the resolution
status of any evidence pointer, and the verdict each axis reaches.  Nothing
here decides which axis is in force; ``maturity_ladder.composition_rule``
in the registry does, and this tool reads that decision only to record it.

The three candidate axes:

``A-strict-min``
    A template's maturity may not exceed the lowest maturity among the
    component options it selects.  A composed suite is only as conformant
    as its weakest member.
``B-trajectory-only``
    No ceiling.  A template at or above the matched-run candidate rung must
    carry an evidence pointer that resolves; below that rung nothing is
    required.
``C-supported-exempt``
    Strict-min, except that options resting at ``supported`` do not pull the
    ceiling down: a scheme whose conformance is settled is treated as no
    longer limiting the composition.

Usage
-----
    python tools/report_template_evidence_consistency.py --out <path>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

MODEL = pathlib.Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from gpuwm.physics_registry import canonical_json  # noqa: E402

RECEIPT_PATH = (
    MODEL / "docs" / "public" / "receipts" / "F2-composition-blast-radius.json"
)

#: The ordering used when the registry under test carries no
#: ``maturity_ladder`` yet.  It exists so this tool can be run at a commit
#: BEFORE the ladder lands -- which is the whole point of a blast-radius
#: measurement -- and the receipt records which source was used, so a reader
#: can tell a measured ladder from this fallback.
_CANDIDATE_ORDER = (
    "planned",
    "port-in-progress",
    "implemented-unverified",
    "experimental-runtime",
    "supported",
    "validation-candidate",
    "wrf-matched-run-candidate",
    "model-validated",
    "wrf-matched-run",
)

AXES = ("A-strict-min", "B-trajectory-only", "C-supported-exempt")

#: The rung at and above which clause C1 requires a resolvable pointer,
#: named by both its pre-rename and post-rename spellings so one tool
#: measures both sides of the Path A rename.
_TRAJECTORY_RUNGS = ("validation-candidate", "wrf-matched-run-candidate",
                     "model-validated", "wrf-matched-run")


def ladder_order(registry: dict) -> tuple[list[str], str]:
    """Return the rung order and where it came from."""

    ladder = registry.get("maturity_ladder")
    if isinstance(ladder, dict):
        order = ladder.get("order")
        if isinstance(order, list) and all(
                isinstance(name, str) for name in order):
            return list(order), "registry.maturity_ladder.order"
    return list(_CANDIDATE_ORDER), "tools fallback candidate order"


def resolve_evidence_pointer(pointer: object) -> dict[str, object]:
    """Resolve a template evidence pointer against the shipped tree.

    A pointer is a mapping naming a matched-run manifest under
    ``gpuwm/authorities/matched_runs/``.  Resolution means the manifest file
    exists, parses, and carries the id the pointer names.  Prose carried in
    an ``evidence`` key elsewhere is not a pointer and is reported as such
    rather than being coerced into one.
    """

    if pointer is None:
        return {"status": "absent"}
    if not isinstance(pointer, dict):
        return {"status": "not-a-pointer", "detail": type(pointer).__name__}
    manifest_id = pointer.get("matched_run_id")
    path = pointer.get("path")
    if not isinstance(manifest_id, str) or not isinstance(path, str):
        return {"status": "malformed",
                "detail": "matched_run_id and path are both required"}
    resolved = MODEL / path
    if not resolved.is_file():
        return {"status": "dangling", "path": path}
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "unparseable", "path": path, "detail": str(exc)}
    if not isinstance(manifest, dict) or manifest.get("id") != manifest_id:
        return {"status": "id-mismatch", "path": path,
                "detail": f"manifest id is {manifest.get('id')!r}"}
    return {"status": "resolved", "path": path, "matched_run_id": manifest_id}


def _rank(order: list[str], name: object) -> int | None:
    if isinstance(name, str) and name in order:
        return order.index(name)
    return None


def _ratified_verdicts(rows: list[dict], rule: object) -> dict[str, object]:
    """Apply the ratified axis with its discharge rule over the enumeration.

    A flagged template is discharged either by a resolvable evidence
    pointer -- the composed suite itself was matched against WRF, which
    outranks a component-wise minimum -- or by an exemption record naming
    the owner decision that granted it.  Anything left is a silent
    violation, and the point of this block is that the set be empty.
    """

    if not isinstance(rule, dict):
        return {"status": "no ratified rule in this registry"}
    axis = rule.get("axis")
    exemptions = rule.get("composition_exemptions", {})
    exemptions = exemptions if isinstance(exemptions, dict) else {}
    discharged, undischarged = [], []
    for row in rows:
        if axis not in row["axes"]:
            continue
        if row["axes"][axis]["verdict"] != "violation":
            continue
        template_id = row["template_id"]
        pointer_resolved = row["evidence_pointer"]["status"] == "resolved"
        exemption = exemptions.get(template_id)
        if pointer_resolved:
            discharged.append({
                "template_id": template_id,
                "discharged_by": "evidence_pointer",
                "detail": row["evidence_pointer"].get("matched_run_id"),
                "reasons": row["axes"][axis]["reasons"],
            })
        elif isinstance(exemption, dict) and exemption.get(
                "owner_decision_id"):
            discharged.append({
                "template_id": template_id,
                "discharged_by": "composition_exemptions",
                "detail": exemption["owner_decision_id"],
                "reasons": row["axes"][axis]["reasons"],
            })
        else:
            undischarged.append({
                "template_id": template_id,
                "reasons": row["axes"][axis]["reasons"],
            })
    unused = sorted(
        set(exemptions)
        - {entry["template_id"] for entry in discharged}
        - {entry["template_id"] for entry in undischarged}
    )
    return {
        "axis": axis,
        "enforcement_point": rule.get("enforcement_point"),
        "owner_decision_id": rule.get("owner_decision_id"),
        "discharged": discharged,
        "undischarged": undischarged,
        "unused_exemptions": unused,
    }


def evaluate(registry: dict) -> dict[str, object]:
    """Return the full three-axis enumeration over every template."""

    order, order_source = ladder_order(registry)
    components = registry.get("components", {})
    templates = registry.get("templates", {})
    supported_rank = _rank(order, "supported")

    rows = []
    for template_id in sorted(templates):
        template = templates[template_id]
        label_maturity = template.get("maturity")
        label_rank = _rank(order, label_maturity)
        selected = template.get("components", {})
        members = []
        for component_id in sorted(selected):
            option_id = selected[component_id]
            option = (
                components.get(component_id, {})
                .get("options", {})
                .get(option_id, {})
            )
            option_maturity = option.get("maturity")
            members.append({
                "component_id": component_id,
                "option_id": option_id,
                "maturity": option_maturity,
                "rank": _rank(order, option_maturity),
            })
        pointer = resolve_evidence_pointer(template.get("evidence_pointer"))
        member_ranks = [m["rank"] for m in members if m["rank"] is not None]
        strict_min = min(member_ranks) if member_ranks else None
        above_supported = [
            m["rank"] for m in members
            if m["rank"] is not None
            and (supported_rank is None or m["rank"] < supported_rank)
        ]
        supported_exempt = (
            min(above_supported) if above_supported else label_rank
        )
        needs_pointer = label_maturity in _TRAJECTORY_RUNGS

        verdicts = {}
        for axis in AXES:
            if axis == "B-trajectory-only":
                ceiling = None
                ceiling_ok = True
            else:
                ceiling = (
                    strict_min if axis == "A-strict-min" else supported_exempt
                )
                ceiling_ok = (
                    label_rank is None or ceiling is None
                    or label_rank <= ceiling
                )
            pointer_ok = (
                pointer["status"] == "resolved" if needs_pointer else True
            )
            reasons = []
            if not ceiling_ok:
                reasons.append("C2 composition ceiling exceeded")
            if not pointer_ok:
                reasons.append(
                    f"C1 trajectory pointer {pointer['status']}")
            verdicts[axis] = {
                "ceiling_rank": ceiling,
                "ceiling_maturity": (
                    order[ceiling] if isinstance(ceiling, int) else None),
                "verdict": "ok" if not reasons else "violation",
                "reasons": reasons,
            }

        rows.append({
            "template_id": template_id,
            "label_maturity": label_maturity,
            "label_rank": label_rank,
            "components": members,
            "evidence_pointer": pointer,
            "c1_applies": needs_pointer,
            "axes": verdicts,
        })

    ladder = registry.get("maturity_ladder")
    rule = (
        ladder.get("composition_rule") if isinstance(ladder, dict) else None
    )
    ratified = _ratified_verdicts(rows, rule)
    return {
        "ratified": ratified,
        "schema": "gpuwm-f2-composition-blast-radius-v1",
        "registry_version": registry.get("registry_version"),
        "ladder_order": order,
        "ladder_order_source": order_source,
        "ratified_axis": (
            rule.get("axis") if isinstance(rule, dict) else None),
        "ratified_enforcement_point": (
            rule.get("enforcement_point") if isinstance(rule, dict) else None),
        "templates": rows,
        "violation_counts": {
            axis: sum(
                1 for row in rows
                if row["axes"][axis]["verdict"] == "violation")
            for axis in AXES
        },
        "violating_template_ids": {
            axis: [
                row["template_id"] for row in rows
                if row["axes"][axis]["verdict"] == "violation"
            ]
            for axis in AXES
        },
        "template_count": len(rows),
    }


def render(report: dict) -> bytes:
    return (canonical_json(report) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registry", type=pathlib.Path, default=None,
        help="registry document to measure; defaults to the production "
             "loader gpuwm.physics_registry.physics_registry()")
    parser.add_argument("--out", type=pathlib.Path, default=RECEIPT_PATH)
    args = parser.parse_args(argv)

    if args.registry is None:
        from gpuwm.physics_registry import physics_registry
        registry = physics_registry()
    else:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))

    report = evaluate(registry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(render(report))
    for axis in AXES:
        print(axis, "violations:", report["violation_counts"][axis],
              report["violating_template_ids"][axis])
    print("templates:", report["template_count"], "|", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
