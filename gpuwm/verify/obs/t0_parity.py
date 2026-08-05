"""The t=0 gap between the two engines, measured and published.

Both arms of the battery start from one analysis, and the battery's whole
claim rests on their starting from the *same* state.  The tree already owns
the instrument that measures a staged pair's initial state --
:mod:`gpuwm.verify.t0_state_digest` -- and this module does not measure it
again.  It composes that digest into the receipt a campaign case ships, and
it changes exactly one thing about how the answer is read.

**The bit-parity gate does not apply across engines, and this module says so
instead of quietly failing.**  The digest's ceilings come from
:mod:`gpuwm.verify.nest_gates`; they were pinned to hold one engine to its
own arithmetic, and a pair of *different* engines reading one analysis will
miss them for reasons that are not defects.  Two responses would be wrong.
Widening a ceiling is forbidden and would break every same-engine receipt
that ceiling protects.  Reporting the cross-engine FAIL as a defect would
bury the real number under a verdict that was never about this comparison.

So the gap is carried as a measurement: per carrier group, the largest
absolute difference any scored variable showed and the variable that showed
it, in the variables' own units, beside the digest's bit-parity verdict
recorded verbatim and marked non-binding.  The specification's instruction
for this comparison is that the t=0 gap is measured, published, and never
assumed; a receipt that graded it would be assuming the answer it was built
to report.

What this module *does* gate is whether the receipt is a t=0 receipt at all:

* every carrier group the digest requires must have scored something, and
* every scored frame must be the initial frame -- the file's own valid time
  against its own ``SIMULATION_START_DATE``.

A pair staged at the wrong lead scores perfectly well and answers a
different question, so it is refused rather than published under this name.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from gpuwm.verify.obs.cross_reader import (
    canonical_json,
    frame_identity,
    registration_sha256,
    resolve_evaluator_commit,
)
from gpuwm.verify.t0_state_digest import (
    CARRIER_GROUPS,
    build_t0_receipt,
    discover_frame_pairs,
)

SCHEMA_ID = "gpuwm.obs-t0-parity/v1"

#: How the two engines came to share an initial state.  The default route
#: has one preparer emit what both engines consume; the fallback runs the
#: node-side WPS chain and carries its own provenance caveat.  Which route a
#: case used is a fact about that case, so it is recorded, never inferred.
IC_ROUTES: tuple[str, ...] = ("exporter-parity", "wps-real")

_MEASURED = "MEASURED"

_STAMP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[_ ](?P<time>\d{2}[:_]\d{2}[:_]\d{2})")


def _normalise_stamp(value: object) -> str | None:
    """One spelling for the several a history tape uses for a time.

    ``Times`` carries ``1974-04-03_14:00:00``, the global attribute carries
    ``1974-04-03_12:00:00``, and file names carry underscores where the
    others carry colons.  Comparing the spellings would report a format
    difference as a lead-time difference.
    """
    if value is None:
        return None
    match = _STAMP_RE.match(str(value).strip())
    if match is None:
        return None
    return f"{match.group('date')}_{match.group('time').replace(':', '_')}"


def make_registration() -> dict[str, object]:
    """What this receipt gates, and what it refuses to gate."""
    return {
        "schema": SCHEMA_ID,
        "measurement": {
            "module": "gpuwm.verify.t0_state_digest",
            "function": "build_t0_receipt",
            "note": ("the state comparison is the tree's existing full-state "
                     "digest, unmodified; this module composes it and does "
                     "not re-measure"),
        },
        "gap": {
            "reduction": ("per carrier group, the largest max_abs_diff any "
                          "scored variable reported, and the variable that "
                          "reported it"),
            "units": "each variable's own",
            "gated": False,
            "reason": ("a cross-engine t=0 comparison is measured and "
                       "published; grading it would assume the answer the "
                       "receipt exists to report"),
        },
        "bit_parity_gate": {
            "source": "gpuwm.verify.nest_gates",
            "binding": False,
            "reason": ("those ceilings hold one engine to its own "
                       "arithmetic; they are recorded here verbatim and "
                       "are not this receipt's verdict"),
        },
        "coverage_gate": {
            "required_carrier_groups": [group.name for group in CARRIER_GROUPS
                                        if group.required],
            "initial_frame_required": True,
            "reason": ("a pair staged at the wrong lead answers a different "
                       "question and is refused under this name"),
        },
        "ic_routes": list(IC_ROUTES),
    }


def summarise_gap(digest: Mapping[str, object]) -> dict[str, object]:
    """Per domain and group, the largest difference and what carried it."""
    summary: dict[str, object] = {}
    for domain in sorted(digest["domains"]):
        groups: dict[str, object] = {}
        for name in sorted(digest["domains"][domain]["groups"]):
            group = digest["domains"][domain]["groups"][name]
            worst_value: float | None = None
            worst_variable: str | None = None
            for variable, metrics in group.get("variables", {}).items():
                value = metrics.get("max_abs_diff")
                if value is None:
                    continue
                if worst_value is None or float(value) > worst_value:
                    worst_value = float(value)
                    worst_variable = variable
            groups[name] = {
                "status": group["status"],
                "scored_arrays": group["scored_arrays"],
                "max_abs_diff": worst_value,
                "max_abs_diff_variable": worst_variable,
                "bit_parity_verdict": group["verdict"],
            }
        summary[domain] = groups
    return summary


def check_initial_frames(candidate_dir: Path, reference_dir: Path, *,
                         valid_time: str | None = None) -> dict[str, object]:
    """Is every scored frame its own run's initial frame?"""
    domains: dict[str, object] = {}
    for pair in discover_frame_pairs(Path(candidate_dir), Path(reference_dir),
                                     valid_time=valid_time):
        sides: dict[str, object] = {}
        agree = True
        for side, path in (("candidate", pair.candidate),
                           ("reference", pair.reference)):
            identity = frame_identity(path)
            start = _normalise_stamp(
                identity["attributes"].get("SIMULATION_START_DATE"))
            frame = _normalise_stamp(identity["valid_time"])
            initial = start is not None and frame is not None and start == frame
            agree = agree and initial
            sides[side] = {
                "title": identity["title"],
                "simulation_start": start,
                "frame_valid_time": frame,
                "is_initial_frame": initial,
            }
        domains[pair.domain] = {
            "scored_valid_time": pair.valid_time,
            "sides": sides,
            "is_initial_frame": agree,
        }
    not_initial = sorted(domain for domain, entry in domains.items()
                         if not entry["is_initial_frame"])
    return {
        "domains": domains,
        "all_initial": bool(domains) and not not_initial,
        "not_initial_domains": not_initial,
    }


def build_t0_parity_receipt(candidate_dir: Path, reference_dir: Path, *,
                            evaluator_commit: str, ic_route: str,
                            valid_time: str | None = None
                            ) -> dict[str, object]:
    """Compose the full-state digest into the campaign's t=0 receipt."""
    if ic_route not in IC_ROUTES:
        raise ValueError(
            f"ic_route must be one of {IC_ROUTES}; a receipt that cannot "
            f"name how the two engines came to share an initial state is "
            f"not evidence (got {ic_route!r})")

    registration = make_registration()
    digest = build_t0_receipt(Path(candidate_dir), Path(reference_dir),
                              evaluator_commit=evaluator_commit,
                              valid_time=valid_time)
    frames = check_initial_frames(Path(candidate_dir), Path(reference_dir),
                                  valid_time=valid_time)

    reasons: list[str] = []
    if not digest["domains"]:
        reasons.append("no frame pair is present on both sides")
    if digest["uncovered_required_groups"]:
        reasons.append(
            "required carrier group(s) scored nothing: "
            + ", ".join(digest["uncovered_required_groups"]))
    if digest["domains"] and not frames["all_initial"]:
        reasons.append(
            "scored frame is not the initial frame on domain(s): "
            + ", ".join(frames["not_initial_domains"]))

    return {
        "schema": SCHEMA_ID,
        "evaluator_commit": evaluator_commit,
        "registration": registration,
        "registration_sha256": registration_sha256(registration),
        "comparison_kind": "cross-engine",
        "ic_route": ic_route,
        "initial_frames": frames,
        "gap": summarise_gap(digest),
        "bit_parity_gate": {
            "verdict": digest["verdict"],
            "binding": False,
            "ceilings": digest["registration"]["ceilings"],
        },
        "coverage_verdict": "PASS" if not reasons else "FAIL",
        "coverage_reasons": reasons,
        "verdict": _MEASURED if not reasons else "REFUSED",
        "digest": digest,
    }


def render_markdown(receipt: Mapping[str, object]) -> str:
    """A reviewer-facing table of the same numbers the receipt carries."""
    gate = receipt["bit_parity_gate"]
    lines = [
        "# t=0 parity: the gap between the engines, measured",
        "",
        f"- schema: `{receipt['schema']}`",
        f"- evaluator commit: `{receipt['evaluator_commit']}`",
        f"- registration sha256: `{receipt['registration_sha256']}`",
        f"- initial-condition route: `{receipt['ic_route']}`",
        f"- coverage: **{receipt['coverage_verdict']}**"
        + ("" if not receipt["coverage_reasons"]
           else " (" + "; ".join(receipt["coverage_reasons"]) + ")"),
        f"- bit-parity gate (recorded, **not binding** across engines): "
        f"{gate['verdict']}",
        f"- verdict: **{receipt['verdict']}**",
        "",
        "| domain | group | scored arrays | max abs diff | carried by | "
        "bit-parity |",
        "|---|---|---:|---:|---|---|",
    ]
    for domain in sorted(receipt["gap"]):
        for name in sorted(receipt["gap"][domain]):
            entry = receipt["gap"][domain][name]
            value = entry["max_abs_diff"]
            lines.append(
                f"| {domain} | {name} | {entry['scored_arrays']} | "
                f"{'-' if value is None else f'{value:.6g}'} | "
                f"{entry['max_abs_diff_variable'] or '-'} | "
                f"{entry['bit_parity_verdict'] or entry['status']} |")
    frames = receipt["initial_frames"]
    lines.extend(["", "## frames scored", ""])
    for domain in sorted(frames["domains"]):
        entry = frames["domains"][domain]
        marker = "initial" if entry["is_initial_frame"] else "NOT INITIAL"
        lines.append(f"- {domain} @ {entry['scored_valid_time']} ({marker})")
        for side in sorted(entry["sides"]):
            side_entry = entry["sides"][side]
            lines.append(
                f"  - {side}: {side_entry['title']!r}, start "
                f"{side_entry['simulation_start']}, frame "
                f"{side_entry['frame_valid_time']}")
    return "\n".join(lines) + "\n"


__all__ = [
    "IC_ROUTES",
    "SCHEMA_ID",
    "build_t0_parity_receipt",
    "canonical_json",
    "check_initial_frames",
    "make_registration",
    "registration_sha256",
    "render_markdown",
    "resolve_evaluator_commit",
    "summarise_gap",
]
