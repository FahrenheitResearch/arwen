"""Survey the live tree for the Level-2 spectral hook seam.

The delivered Level-2 package prescribed: "Before editing, run the
integration survey against your current checkout and compare it with the
delivered survey.  If the seam moved, update the handoff record and wire
the current seam; do not force an old line-number patch."  The package's
combined patch arrived EMPTY, so the delivered survey tool and its survey
were lost with it; this tool is the reconstruction.  It derives the seam
from the LIVE sources -- structural anchors, not stored line numbers -- so
a drifted tree is reported as drift instead of being patched blind.

Usage:
    python -m tools.spectral_seam_survey            # print the survey JSON
    python -m tools.spectral_seam_survey --check    # compare with the
        committed handoff record and exit nonzero on structural drift
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SURVEY_SCHEMA = "gpuwm.core-spectral-seam-survey/v2"
HANDOFF_PATH = (Path(__file__).resolve().parents[1] / "docs" / "handoffs"
                / "CURRENT-CORE-SPECTRAL-SEAM-SURVEY.json")

#: The ordered anchors that ARE the seam.  Every marker must exist inside
#: ``execute_experiment``'s ``on_step`` closure, in this order; each names
#: the breakage its absence would mean.
ON_STEP_ANCHORS = (
    ("stepper_call", "steppers.get(grid_id, step)(",
     "the STEP op no longer steps through the shared stepper binding"),
    ("slow_state_commit", "after_step=True",
     "the post-step clock refresh (the commit marker) moved or vanished"),
    ("spectral_hook", "spectral_seam.after_step(",
     "the Level-2 hook no longer fires at the slow-step commit"),
    ("post_step_observers", "poison()",
     "the hook drifted below the post-step observers"),
)


def _index_all(text: str, markers) -> list[dict[str, object]]:
    rows = []
    for name, marker, breakage in markers:
        position = text.find(marker)
        rows.append({
            "anchor": name,
            "marker": marker,
            "found": position >= 0,
            "line_offset": (None if position < 0
                            else text[:position].count("\n")),
            "breakage_if_missing": breakage,
        })
    return rows


def survey() -> dict[str, object]:
    from gpuwm.core import dycore
    from gpuwm.core.model import execute_experiment

    source = inspect.getsource(execute_experiment)
    module = inspect.getmodule(execute_experiment)
    on_step_start = source.find("def on_step")
    on_force_start = source.find("def on_force")
    on_step = source[on_step_start:on_force_start] \
        if 0 <= on_step_start < on_force_start else ""
    anchors = _index_all(on_step, ON_STEP_ANCHORS)
    positions = [row["line_offset"] for row in anchors]
    ordered = (all(p is not None for p in positions)
               and positions == sorted(positions))
    attach_offset = source.find("attach_seam(")

    step_doc = inspect.getdoc(dycore.step) or ""
    return {
        "schema": SURVEY_SCHEMA,
        "surveyed_utc": datetime.now(timezone.utc).isoformat(),
        "seam": {
            "module": module.__name__,
            "function": "execute_experiment",
            "closure": "on_step",
            "found": bool(on_step),
            "anchors": anchors,
            "anchors_in_order": ordered,
            "attach_before_on_step": (0 <= attach_offset < on_step_start),
            "statement": (
                "the hook fires once per domain per model time step, "
                "immediately after refresh_model_time(after_step=True) -- "
                "the slow RK state commit -- inside the STEP op's domain "
                "turn, before poison/health/step_observer, before output, "
                "nest feedback and the next large step"),
        },
        "acoustic_containment": {
            "module": "gpuwm.core.dycore",
            "function": "step",
            "substeps_inside_step": "acoustic substep" in step_doc.lower()
            or "acoustic substeps" in step_doc.lower(),
            "statement": (
                "acoustic substeps live entirely inside dycore.step (the "
                "stepper the STEP op calls), so a hook that fires after "
                "the stepper returns cannot fire inside one"),
        },
        "routes": {
            "honored_via_execute_experiment": [
                "runtime.run_experiment:domain-tree",
                "prepared_single_domain_forecast",
                "prepared_domain_tree_forecast",
            ],
            "refuse_active_config": [
                "runtime.run_experiment:single-domain (frozen loop)",
                "ensemble member (frozen single-domain loop)",
            ],
        },
        "refusals": {
            "streamed_domain": (
                "gpuwm.spectral_seam.SpectralSeam.validate_domain: "
                "node.state is the t=0 attach snapshot; target planes are "
                "not resident at the seam"),
            "false_periodic_declaration": (
                "gpuwm.spectral_seam.SpectralSeam.validate_domain: "
                "periodic_domain on an open/specified/nested domain"),
            "unrouted_active_config": (
                "gpuwm.experiment.refuse_unrouted_spectral_numerics"),
        },
    }


def structural(view: dict[str, object]) -> dict[str, object]:
    """The drift-comparable core: anchors and routing, no timestamps."""
    seam = dict(view.get("seam", {}))
    return {
        "module": seam.get("module"),
        "function": seam.get("function"),
        "closure": seam.get("closure"),
        "found": seam.get("found"),
        "anchor_names_found": [
            row["anchor"] for row in seam.get("anchors", ())
            if row.get("found")],
        "anchors_in_order": seam.get("anchors_in_order"),
        "attach_before_on_step": seam.get("attach_before_on_step"),
        "routes": view.get("routes"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare against the committed handoff survey "
                             "and exit 1 on structural drift")
    parser.add_argument("--handoff", type=Path, default=HANDOFF_PATH)
    args = parser.parse_args(argv)

    live = survey()
    if not args.check:
        print(json.dumps(live, indent=2, sort_keys=True))
        return 0
    recorded = json.loads(args.handoff.read_text(encoding="utf-8"))
    live_core = structural(live)
    recorded_core = structural(recorded)
    if live_core == recorded_core:
        print("seam survey: live tree matches the committed handoff record")
        return 0
    print("seam survey: STRUCTURAL DRIFT between the live tree and "
          f"{args.handoff}", file=sys.stderr)
    print(json.dumps({"live": live_core, "recorded": recorded_core},
                     indent=2, sort_keys=True), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
