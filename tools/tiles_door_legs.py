"""The two [tiles] nesting shapes, driven through the PRODUCT FRONT DOOR.

WHAT THIS GATES, AND WHY IT IS NOT A PYTEST
-------------------------------------------
``tilestream/`` gates the streamed transport against a monolithic reference
inside the engine.  This gate asks a different and strictly weaker-looking
question that turned out to be the one that failed: can a USER REACH those
shapes by typing the command the documentation gives them?

    gpuwm run <config> --outdir <dir>

Zero workaround flags, zero environment nudges beyond the card budget.  The
2.2 headline defect (task #196) was precisely that streaming worked and was
UNREACHABLE from every product front door, and the 2.3.0 re-cut exists
because two nesting shapes were still unreachable after that.  A gate that
imports ``gpuwm.core.streaming`` and calls ``steppers_for_tree`` cannot see
that class of defect at all -- it walks around the door.

So this is a scripted gate in ``tools/battery/tiles_gates.txt``, beside the
other whole-forecast comparisons, rather than a pytest: what it compares is
a whole forecast against another whole forecast, it ends in one verdict
line, and collecting it as pytest would throw the matrix away.

THE TWO LEGS
------------
Both carry ``feedback = 1`` -- two-way nesting, the shape a user actually
runs -- and both are digest-compared against an all-resident CONTROL of the
same tree.

  LEG A   streamed parent, resident child.  Tree-wide ``[tiles]`` names
          mode ``on`` and the budget; ``d02`` carries ``tiles = { mode =
          "off" }``.
  LEG B   resident parent, streamed child.  Tree-wide ``[tiles]`` is mode
          ``off`` and carries ONLY the budget -- a per-domain table may not
          name the card, so the budget has nowhere else to live -- and
          ``d02`` carries ``tiles = { mode = "on" }``.

Leg B is the shape the round-2 receipt rider was written for.  Before it,
that leg streamed a grid and ``streaming_receipt`` returned ``{}``, so the
run tiled a domain and printed no line naming it.

WHY A CONTROL AND A DUAL RUN, BOTH
----------------------------------
The DUAL RUN (each leg twice) catches nondeterminism and, on a card with no
ECC, silent corruption -- it is the project's standing corruption detector.
The CONTROL catches the failure the dual run structurally cannot: a leg that
quietly declined to stream and ran resident matches its own second run
perfectly and passes.  ``[tiles].streamed_any`` in the run receipt is
asserted for the same reason and is the cheaper half of it: a leg whose
receipt says nothing streamed is a leg that proved nothing, and is FAILED
here rather than reported green.  That is the A/B-arms law -- an exact-0.0
delta means the treatment never ran.

THE BUDGET IS NOT COMMITTED
---------------------------
``configs/battery/shape_tiles_*.toml.tmpl`` carry
``VRAM_BUDGET_BYTES_PLACEHOLDER`` where ``vram_budget_bytes`` goes, because
that number is a fact about a CARD and a committed value would be a claim
about whichever machine happened to run the gate once.  This module
substitutes it, from ``GPUWM_TILES_VRAM_BUDGET_BYTES`` if set, otherwise
from the free VRAM the planner's own probe reports.

INVOCATION
----------
From the repository root, as ``tools/battery/tiles_gates.txt`` says::

    LEG=A python -m tools.tiles_door_legs
    LEG=B python -m tools.tiles_door_legs

``GPUWM_CASE_DATA_ROOT`` must point at the directory holding ``WPS_GEOG``,
and the ERA5 forcing named by ``[case_data]`` must be staged under the case
directory.  Exit status is the verdict: 0 passes, non-zero fails.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = REPOSITORY_ROOT / "configs" / "battery"

#: ``LEG`` value -> (template, human name).
LEGS = {
    "A": ("shape_tiles_streamed_parent_resident_child.toml.tmpl",
          "streamed parent + resident child"),
    "B": ("shape_tiles_resident_parent_streamed_child.toml.tmpl",
          "resident parent + streamed child"),
}

CONTROL = "shape_tiles_control_resident_tree.toml"

PLACEHOLDER = "VRAM_BUDGET_BYTES_PLACEHOLDER"


def _budget_bytes() -> int:
    """The card budget the templates leave blank.

    The environment wins, so an operator can pin the leg to a number and
    compare two boxes.  Otherwise the planner's own probe answers, which is
    the same source ``[tiles] mode = "auto"`` would consult -- asking
    anything else here would gate a budget the product never uses.
    """
    named = os.environ.get("GPUWM_TILES_VRAM_BUDGET_BYTES")
    if named:
        return int(named)
    from tilestream import autoplan

    return int(autoplan.Machine.detect().vram_budget_bytes)


def _materialize(template: pathlib.Path, into: pathlib.Path,
                 budget: int) -> pathlib.Path:
    """Write the template out with its budget filled in.

    Into the WORK directory, never back over the committed template: the
    placeholder is the committed state and a gate that edited its own case
    files in place would leave the tree dirty and the next run pre-filled
    with the last box's number.
    """
    text = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in text:
        raise SystemExit(
            f"{template.name} carries no {PLACEHOLDER}; the committed "
            "templates must not name a card, so this is either the wrong "
            "file or a budget got baked in")
    out = into / template.name[: -len(".tmpl")]
    out.write_text(text.replace(PLACEHOLDER, str(budget)), encoding="utf-8")
    return out


def _stage_case(work: pathlib.Path) -> None:
    """Put the case's side files where the config's relative paths expect."""
    work.mkdir(parents=True, exist_ok=True)
    namelist = CASES / "shape_tiles_doorleg.namelist.wps"
    shutil.copy2(namelist, work / namelist.name)
    for extra in ("data",):
        source = CASES / extra
        if source.is_dir() and not (work / extra).exists():
            shutil.copytree(source, work / extra)


def _run_front_door(config: pathlib.Path, outdir: pathlib.Path) -> None:
    """``gpuwm run <config> --outdir <dir>`` and nothing else.

    Through ``python -m gpuwm.cli`` rather than the ``gpuwm`` console script
    so the leg is pinned to the interpreter running this gate -- the
    provenance program's whole point is that a run states which tree is
    executing, and a console script resolved off PATH can be a different
    tree entirely.  The ARGUMENTS are exactly what a user types; that is
    what is under test.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", "run", str(config),
         "--outdir", str(outdir)],
        cwd=config.parent, text=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"FAILED  the front door refused: gpuwm run {config.name} "
            f"--outdir {outdir.name} exited {completed.returncode}.  That "
            "is the defect this gate exists to catch -- the shape is "
            "unreachable from the command a user types.")


def _wrfout_digests(outdir: pathlib.Path) -> dict[str, dict[str, str]]:
    """``{wrfout name: {variable: sha256}}`` for every history file written."""
    from tilestream.bench_ooc_output import file_var_digests

    files = sorted(p for p in outdir.rglob("wrfout*") if p.is_file())
    if not files:
        raise SystemExit(
            f"FAILED  {outdir} holds no wrfout; the run reported success "
            "and produced nothing to compare, which is a pass this gate "
            "must never award")
    return {p.name: file_var_digests(p) for p in files}


def _streamed_any(outdir: pathlib.Path) -> tuple[bool, str]:
    """The run receipt's ``[tiles]`` verdict, and the line it printed.

    Read rather than inferred.  A leg that ran resident writes the same
    kind of wrfout as a leg that streamed, so without this the digest
    comparison would happily certify a run in which the mode never engaged.
    """
    import json

    receipts = sorted(outdir.rglob("*run*.json"))
    for path in receipts:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        tiles = payload.get("tiles")
        if isinstance(tiles, dict) and "streamed_any" in tiles:
            return bool(tiles["streamed_any"]), str(tiles.get("summary", ""))
    return False, "(no [tiles] section found in any run receipt)"


def _compare(label: str, left: dict, right: dict) -> int:
    """Print one matrix block; return the number of disagreeing variables."""
    differing = 0
    names = sorted(set(left) | set(right))
    for name in names:
        if name not in left or name not in right:
            print(f"  {label:<28} {name:<24} MISSING on one side")
            differing += 1
            continue
        for var in sorted(set(left[name]) | set(right[name])):
            a, b = left[name].get(var), right[name].get(var)
            if a != b:
                print(f"  {label:<28} {name}:{var:<18} DIFFERS")
                differing += 1
    print(f"  {label:<28} {'':<24} "
          f"{len(names)} file(s), {differing} differing")
    return differing


def main() -> int:
    leg = os.environ.get("LEG", "").strip().upper()
    if leg not in LEGS:
        print(f"LEG must be one of {sorted(LEGS)}; got {leg!r}",
              file=sys.stderr)
        return 2
    template_name, human = LEGS[leg]

    work = REPOSITORY_ROOT / "runs" / f"tiles-door-leg-{leg.lower()}"
    budget = _budget_bytes()
    print(f"[tiles] DOOR-LEG {leg}: {human}")
    print(f"  budget      {budget} bytes "
          f"({budget / (1024 ** 3):.2f} GiB)")
    print(f"  work        {work}")

    _stage_case(work)
    config = _materialize(CASES / template_name, work, budget)
    control = shutil.copy2(CASES / CONTROL, work / CONTROL)

    runs = {}
    for arm in ("run1", "run2"):
        outdir = work / f"{leg.lower()}-{arm}"
        _run_front_door(config, outdir)
        runs[arm] = _wrfout_digests(outdir)
    streamed, summary = _streamed_any(work / f"{leg.lower()}-run1")
    print(f"  receipt     {summary}")

    control_out = work / "control"
    _run_front_door(pathlib.Path(control), control_out)
    control_digests = _wrfout_digests(control_out)

    print("  MATRIX")
    dual = _compare("dual run (must MATCH)", runs["run1"], runs["run2"])
    versus = _compare("vs control (must MATCH)", runs["run1"], control_digests)

    failures = []
    if not streamed:
        failures.append(
            "the run receipt says NO domain streamed, so this leg measured "
            "a resident run against a resident run")
    if dual:
        failures.append(
            f"{dual} variable(s) differ between two identical runs -- "
            "nondeterminism or corruption, not a streaming result")
    if versus:
        failures.append(
            f"{versus} variable(s) differ from the all-resident control; "
            "streaming must not change the answer")

    if failures:
        for line in failures:
            print(f"  ! {line}")
        print(f"GATE FAILED  door-leg {leg} ({human})")
        return 1
    print(f"GATE PASSED  door-leg {leg} ({human}): streamed, "
          "bit-identical to the resident control, reproducible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
