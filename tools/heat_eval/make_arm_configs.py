"""Write each case's arm configurations as overlays on its arm A configuration.

Arm A for a case is the door template (``door-template.toml``) with its
start, length, name and boundary cycle set for that case.  Every other arm
is arm A with at most three things changed: the start time, the run length
and an added ``[radar_heating]`` table.  Two checks hold that line:

* the arm row in ``arms.toml`` may carry no key beyond those
  (:func:`tools.heat_eval.plan.arm_from_row`);
* the written config is parsed back and compared with arm A's, key by key,
  and any other difference is refused (:func:`check_overlay`).

Why: an arm that differs from A in a second knob confounds the comparison,
and the decision rule would credit (or blame) the heating for it.

    python -m tools.heat_eval.make_arm_configs --case ID [--root /work/nh] [--plan core|full]

writes ``<case>/arms/<ARM>/experiment.toml``, ``<case>/arms/<ARM>/arm.json``
and one ``<case>/preps/<HHMM>z-<N>h/experiment.toml`` per start and length:
the arm config with its ``[radar_heating]`` table cut out, exactly as the
door cuts it (:func:`table_free`), checked identical for every arm that
shares it.  The door binds the SHA-256 of the config its preparation read
and refuses any other, so a preparation's bytes must be the table-free bytes
of every arm that runs on it.  It also writes the one-hour preparation of
gate G0's door case (:func:`tools.heat_eval.plan.g0_door_key`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

from tools.heat_eval import plan as planmod
from tools.heat_eval.plan import Arm, Case, PlanError

TEMPLATE = planmod.HERE / "door-template.toml"

#: The only keys an arm config may differ from arm A's in.
OVERLAY_KEYS = frozenset({("experiment", "start_time"), ("experiment", "run_seconds")})
OVERLAY_TABLE = "radar_heating"


class OverlayRefused(PlanError):
    """An arm config that differs from arm A outside the overlay."""


_HEADER = re.compile(r"^\s*\[\[?\s*([A-Za-z0-9_.\-]+)\s*\]\]?\s*(#.*)?$")
_TABLE_HEADER = re.compile(r"^\s*\[\s*radar_heating\s*\]\s*(#.*)?$")


def table_free(text: str) -> str:
    """The config with its ``[radar_heating]`` table cut out, as the door cuts it.

    The rule of work package 2's ``split_experiment_text``
    (``gpuwm/da/forecast_heating.py``): every other byte stays, and one
    blank line written just before the table goes with it, so an arm config
    made by appending the table to its preparation's config splits back into
    exactly that config's bytes.  The door checks those bytes against its
    preparation's binding; any difference is a refusal at the door, after
    the cards were taken.
    """
    lines = text.splitlines(keepends=True)
    head = next((n for n, line in enumerate(lines) if _TABLE_HEADER.match(line)), None)
    if head is None:
        return text
    end = next((n for n in range(head + 1, len(lines)) if _HEADER.match(lines[n])), len(lines))
    start = head - 1 if head > 0 and not lines[head - 1].strip() else head
    return "".join(lines[:start] + lines[end:])


def _set_keys(text: str, table: str, values: Mapping[str, str]) -> str:
    """Replace ``key = ...`` lines inside ``[table]``, each exactly once."""
    lines = text.splitlines(keepends=True)
    current = None
    done: dict[str, int] = {key: 0 for key in values}
    for index, line in enumerate(lines):
        header = _HEADER.match(line)
        if header:
            current = header.group(1)
            continue
        if current != table:
            continue
        for key, value in values.items():
            if re.match(rf"^\s*{re.escape(key)}\s*=", line):
                ending = "\n" if line.endswith("\n") else ""
                lines[index] = f"{key} = {value}{ending}"
                done[key] += 1
    bad = {key: count for key, count in done.items() if count != 1}
    if bad:
        raise PlanError(f"[{table}] keys {bad} were not found exactly once in the template")
    return "".join(lines)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def _local(time: datetime) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def case_config(template: str, case: Case, start: datetime, run_seconds: int,
                boundary_end_hour: int) -> str:
    """Arm A's config for ``case`` at ``start`` (the template, re-timed)."""
    text = _set_keys(template, "experiment", {
        "name": json.dumps(f"heat-eval-{case.id}-{start.strftime('%H')}z"),
        "start_time": _local(start),
        "run_seconds": f"{float(run_seconds)!r}",
    })
    return _set_keys(text, "fetch", {
        "cycle": json.dumps(case.boundary_cycle.strftime("%Y-%m-%dT%H")),
        "hours": str(int(boundary_end_hour)),
        "cadence": str(case.boundary_cadence_hours),
    })


def arm_config(a_text: str, arm: Arm, case: Case, plan: str, windows: Path | None) -> str:
    """Arm A's config with this arm's overlay: start, length, heating table."""
    text = _set_keys(a_text, "experiment", {
        "start_time": _local(arm.start(case)),
        "run_seconds": f"{float(arm.run_seconds(plan))!r}",
    })
    if arm.heated:
        if windows is None:
            raise PlanError(f"arm {arm.id} heats and needs its windows root")
        rows = [f"\n[{OVERLAY_TABLE}]\n"]
        for key in planmod.RADAR_HEATING_KEYS:
            if key == "windows":
                rows.append(f"windows = {json.dumps(windows.as_posix())}\n")
            elif key in arm.heating:
                rows.append(f"{key} = {_toml_value(arm.heating[key])}\n")
        text = text.rstrip("\n") + "\n" + "".join(rows)
    return text


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, Mapping):
        out: dict[tuple[str, ...], object] = {}
        for key, item in value.items():
            out.update(_flatten(item, prefix + (str(key),)))
        return out
    if isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, prefix + (f"[{index}]",)))
        return out
    return {prefix: value}


def check_overlay(a_text: str, arm_text: str) -> list[tuple[str, ...]]:
    """The keys the arm config changes, refused unless all are overlay keys."""
    a_doc = tomllib.loads(a_text)
    arm_doc = tomllib.loads(arm_text)
    if OVERLAY_TABLE in a_doc:
        raise OverlayRefused("arm A carries a [radar_heating] table; A is the unheated control")
    a_flat = _flatten(a_doc)
    arm_flat = _flatten({key: value for key, value in arm_doc.items() if key != OVERLAY_TABLE})
    changed = sorted(key for key in set(a_flat) | set(arm_flat)
                     if a_flat.get(key, KeyError) != arm_flat.get(key, KeyError))
    outside = [key for key in changed if key not in OVERLAY_KEYS]
    if outside:
        raise OverlayRefused(
            f"the arm config differs from arm A in {['.'.join(key) for key in outside]}; an "
            f"arm may change only experiment.start_time, experiment.run_seconds and the "
            f"[radar_heating] table, because a second changed knob makes the comparison with "
            f"A meaningless")
    heating = arm_doc.get(OVERLAY_TABLE, {})
    extra = set(heating) - set(planmod.RADAR_HEATING_KEYS)
    if extra:
        raise OverlayRefused(f"[radar_heating] carries unknown keys {sorted(extra)}")
    return changed + [(OVERLAY_TABLE, key) for key in sorted(heating)]


def write_case(root: Path, case: Case, table: planmod.ArmTable, plan: str,
               template: str) -> dict[str, object]:
    """Every config the case needs, checked, and a summary of what was written."""
    arms = table.for_case(case, plan)
    control = table.arm("A")
    boundary_end = int((max(arm.end(case, plan) for arm in arms)
                        - case.boundary_cycle).total_seconds() // 3600)
    a_text = case_config(template, case, control.start(case), control.run_seconds(plan), boundary_end)
    written: dict[str, object] = {"case": case.id, "plan": plan, "preps": {}, "arms": {}}
    prep_texts: dict[str, str] = {}

    def add_prep(key: str, text: str, start: datetime, run_seconds: int, user: str) -> Path:
        base = table_free(text)
        if key in prep_texts and prep_texts[key] != base:
            raise PlanError(
                f"preparation {key}: {user}'s config without its [radar_heating] table differs "
                f"from another arm's on the same preparation; the door would refuse one of them")
        prep = planmod.prep_dir(root, case, key)
        if key not in prep_texts:
            prep_texts[key] = base
            prep.mkdir(parents=True, exist_ok=True)
            (prep / "experiment.toml").write_text(base, encoding="utf-8", newline="\n")
            written["preps"][key] = {
                "folder": str(prep), "start": planmod.iso(start), "run_seconds": run_seconds,
                "boundary_leads": planmod.boundary_leads(
                    case, start, start + timedelta(seconds=run_seconds)),
                "arms": []}
        written["preps"][key]["arms"].append(user)
        return prep

    for arm in arms:
        windows = planmod.windows_root(root, case, arm) if arm.heated else None
        text = arm_config(a_text, arm, case, plan, windows)
        changed = check_overlay(a_text, text)
        out = planmod.arm_dir(root, case, arm)
        out.mkdir(parents=True, exist_ok=True)
        (out / "experiment.toml").write_text(text, encoding="utf-8", newline="\n")
        key = arm.prep_key(case, plan)
        prep = add_prep(key, text, arm.start(case), arm.run_seconds(plan), arm.id)
        record = {
            "schema": "gpuwm.heat-eval-arm.v1", "case": case.id, "arm": arm.id,
            "label": arm.label, "column": arm.column, "plan": plan,
            "start": planmod.iso(arm.start(case)), "issue_time": planmod.iso(case.t0),
            "run_seconds": arm.run_seconds(plan),
            "prep": str(prep), "prep_key": key,
            "needs": arm.needs(case, plan),
            "lead_class": arm.lead_class, "cards": arm.cards, "min_cards": arm.min_cards,
            "heating": dict(arm.heating, windows=str(windows)) if windows else None,
            "window_ends": [planmod.iso(end) for end in arm.window_ends(case)],
            "overlay_keys": [".".join(key) for key in changed],
        }
        (out / "arm.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        written["arms"][arm.id] = record
    # Gate G0's door case: arm A's config, one hour long (the door runs the
    # config's own length, so the hour has to be prepared, not asked for).
    g0_text = arm_config(a_text, Arm(
        id="G0", label="G0 door case", column="A", plan="core", roles=(),
        start_offset_minutes=0, run_hours=planmod.G0_DOOR_RUN_SECONDS // 3600,
        core_run_hours=planmod.G0_DOOR_RUN_SECONDS // 3600, lead_class="none",
        cards=0, min_cards=0), case, plan, None)
    check_overlay(a_text, g0_text)
    add_prep(planmod.g0_door_key(case), g0_text, case.t0, planmod.G0_DOOR_RUN_SECONDS, "G0")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.heat_eval.make_arm_configs")
    parser.add_argument("--case", required=True)
    parser.add_argument("--root", type=Path, default=planmod.DEFAULT_ROOT)
    parser.add_argument("--plan", default="core", choices=planmod.PLANS)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    args = parser.parse_args(argv)
    try:
        case = planmod.case_by_id(args.case)
        written = write_case(args.root, case, planmod.load_arms(), args.plan,
                             args.template.read_text(encoding="utf-8"))
    except PlanError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
