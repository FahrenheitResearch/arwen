"""Refuse the A/B now, so a freed GPU cannot be spent on a typo.

Three classes of failure are knowable before anything runs, and this
checks all three.

**Grammar.**  Every argument list in the plan goes through the REAL
parser of the REAL entry point it names -- not a schema of it, not a
copy of it.  The two shim tools (``prepare_case``, ``run_arm``) are
invoked with their own ``--validate-only``, which pushes the assembled
list through ``gpuwm.source_cli``'s requirement check and
``tools.da_cycle_prepared``'s parser respectively, so the grammar that
is proved is the grammar that will run.

**Inputs.**  Every path the plan reads that must exist BEFORE the queue
starts is checked.  Paths under the run directory are produced by an
earlier arm and are deliberately not checked; a plan that demanded them
would be unverifiable until after it had run.

**Physics and arithmetic.**  The things a parser cannot know: that the
requested HRRR cycles are published and reach the requested leads, that
HRRR's grid covers this domain with its halo, that the leg schedule fits
inside the boundary data each case carries, that every arm assimilates
the same observations at the same times with the same member count and
the same perturbation, and that the ensemble the driver would build is
not a fabricated one.

Exit 0 means every arm would at least start on this host, with these
files.  It does not mean every arm will finish: a card that fills up, an
S3 outage, a decoder refusal are all still ahead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VERDICT_SCHEMA = "gpuwm-da.background-ab-preflight.v1"

#: Flags naming inputs that must exist before the queue starts.  Outputs
#: and anything an earlier arm writes must NOT.
INPUT_FLAGS = frozenset({
    "--gfs-authority", "--gfs-prepared-root", "--obs", "--grid-wrfout",
    "--geog-root", "--wps-namelist", "--reference-prepared-root",
    "--prepared-root", "--authority-dir",
})

#: Modules validated by importing them and stopping at ``parse_args``.
IN_PROCESS = (
    "tools.download_hrrr_native_subset",
    "tools.da_background_ab.build_case_inputs",
    "tools.da_background_ab.check_case_parity",
    "tools.da_background_ab.score_background_ab",
)

#: Modules that validate themselves, given their own flag.
SELF_VALIDATING = {
    "tools.da_background_ab.prepare_case": "--validate-only",
    "tools.da_background_ab.run_arm": "--validate-only",
}


class Parsed(Exception):
    pass


def parse_in_process(module: str, argv: list[str]) -> argparse.Namespace:
    import importlib

    target = importlib.import_module(module)
    real = argparse.ArgumentParser.parse_args
    captured = {}

    def intercept(self, args=None, namespace=None):
        captured["ns"] = real(self, args if args is not None else argv,
                              namespace)
        raise Parsed()

    argparse.ArgumentParser.parse_args = intercept
    try:
        target.main()
    except Parsed:
        return captured["ns"]
    finally:
        argparse.ArgumentParser.parse_args = real
    raise SystemExit(f"{module}.main() returned without parsing")


def substitute(argv: list[str], *, run_dir: str, repo: str) -> list[str]:
    return [item.replace("${RUN_DIR}", run_dir).replace("${REPO}", repo)
            for item in argv]


def check_paths(argv: list[str], run_dir: str, problems: list[str],
                where: str) -> None:
    for flag, value in zip(argv, argv[1:]):
        if flag not in INPUT_FLAGS or value.startswith("-"):
            continue
        if run_dir in value:
            continue                     # produced by an earlier arm
        if not Path(value).exists():
            problems.append(f"{where}: {flag} points at a missing path: "
                            f"{value}")


def value_after(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def values_after(argv: list[str], flag: str) -> list[str]:
    return [argv[index + 1] for index, item in enumerate(argv)
            if item == flag and index + 1 < len(argv)]


def check_background_registry(plan: dict, problems: list[str],
                              notes: list[str]) -> None:
    """The things only the background registry knows."""

    from gpuwm.da import background as bg
    from gpuwm.experiment import load_experiment

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    init = datetime.strptime(plan["case"]["init"], "%Y-%m-%dT%H:%M:%SZ")
    case_root = Path(plan["case"]["prepared_case_root"])
    authority = case_root / "go" / "authority" / "experiment.toml"

    for source in ("gfs", "hrrr"):
        try:
            bg.resolve_background_source(source)
        except bg.BackgroundError as error:
            problems.append(f"background registry: {error}")

    if authority.is_file():
        experiment = load_experiment(authority)
        try:
            bg.refuse_uncovered_domain("hrrr", experiment)
            notes.append(
                "HRRR domain coverage: PASS (the strict gate -- "
                "interpolation stencil and surface-donor halo have real "
                "source cells on every side)")
        except bg.BackgroundError as error:
            problems.append(f"HRRR cannot force this domain: {error}")
    else:
        problems.append(f"case authority missing: {authority}")

    # Both HRRR cycles must be far enough in the past to be published,
    # and must reach the leads the preparation asks for.  Asked with the
    # registry's own fail-closed arithmetic rather than a calendar.
    for arm in plan["arms"]:
        if not arm["name"].startswith("prepare-hrrr-"):
            continue
        argv = arm["steps"][0]["argv"]
        cycle = datetime.strptime(value_after(argv, "--valid-time"),
                                  "%Y-%m-%d_%H:%M:%S")
        start_hour = int(value_after(argv, "--forecast-start-hour"))
        run_seconds = float(value_after(argv, "--run-seconds"))
        horizon = bg.BACKGROUND_SOURCES["hrrr"].horizon(cycle)
        end_hour = start_hour + int(run_seconds // 3600)
        lag = bg.BACKGROUND_SOURCES["hrrr"].lag_seconds(end_hour)
        age = (now - cycle).total_seconds()
        if end_hour > horizon:
            problems.append(
                f"{arm['name']}: needs f{start_hour:03d}..f{end_hour:03d} "
                f"of the {cycle:%Y-%m-%dT%H}Z cycle, which publishes only "
                f"to f{horizon:03d}")
        elif age < lag:
            problems.append(
                f"{arm['name']}: the {cycle:%Y-%m-%dT%H}Z cycle is "
                f"{age / 3600:.1f} h old and f{end_hour:03d} is not "
                f"plausibly published until {lag / 3600:.1f} h")
        else:
            notes.append(
                f"{arm['name']}: {cycle:%Y-%m-%dT%H}Z "
                f"f{start_hour:03d}..f{end_hour:03d} within the "
                f"f{horizon:03d} horizon, cycle age {age / 3600:.1f} h "
                f"against a {lag / 3600:.1f} h publication lag")


def check_ensemble(plan: dict, problems: list[str], notes: list[str]) -> None:
    """The refusal the driver would raise, raised here instead."""

    from gpuwm.da import background as bg

    for arm in plan["arms"]:
        if "members" not in arm:
            continue
        argv = arm["steps"][0]["argv"]
        members = int(value_after(argv, "--members"))
        seed = int(value_after(argv, "--seed"))
        wind = float(value_after(argv, "--wind-sigma-ms"))
        try:
            plan_members = bg.plan_member_backgrounds(
                control_name="control", members=members, seed=seed,
                perturbed_fields=[{"name": "u", "amplitude": wind},
                                  {"name": "v", "amplitude": wind}],
                perturbed_species=[])
        except bg.BackgroundError as error:
            problems.append(f"{arm['name']}: {error}")
            continue
        notes.append(
            f"{arm['name']}: {len(plan_members) - 1} distinct members plus "
            "one never-analysed control, none of them a duplicate")


def check_shared_configuration(plan: dict, problems: list[str],
                               notes: list[str]) -> None:
    """Everything that must be the SAME across arms, compared as data."""

    cycling = [arm for arm in plan["arms"] if "members" in arm]
    if len(cycling) < 2:
        problems.append("fewer than two cycling arms: this is not an A/B")
        return

    def fingerprint(arm):
        argv = arm["steps"][0]["argv"]
        shared = {}
        for flag in ("--members", "--seed", "--leg-seconds", "--free-legs",
                     "--wind-sigma-ms", "--length-scale-km",
                     "--horizontal-loc-m", "--vertical-loc-m",
                     "--rtps-alpha", "--relaxation", "--thin-cells",
                     "--err-inflation", "--physics-profile",
                     "--history-interval-seconds"):
            shared[flag] = value_after(argv, flag)
        shared["--obs"] = values_after(argv, "--obs")
        shared["--grid-wrfout"] = values_after(argv, "--grid-wrfout")
        return shared

    reference = fingerprint(cycling[0])
    for arm in cycling[1:]:
        other = fingerprint(arm)
        for key, value in reference.items():
            if other[key] != value:
                problems.append(
                    f"{arm['name']} differs from {cycling[0]['name']} in "
                    f"{key}: {value!r} vs {other[key]!r}.  This A/B is only "
                    "attributable to the background if nothing else moves")
    if not problems:
        notes.append(
            f"all {len(cycling)} cycling arms share their observations, "
            "georeference frames, member count, seed, perturbation, "
            "localization, relaxation, thinning, physics profile and leg "
            "schedule -- compared flag by flag, not asserted")

    # Leg arithmetic against each case's own boundary horizon.
    for arm in cycling:
        argv = arm["steps"][0]["argv"]
        legs = (len(values_after(argv, "--obs"))
                + int(value_after(argv, "--free-legs")))
        span = legs * float(value_after(argv, "--leg-seconds"))
        bound = float(value_after(argv, "--run-seconds"))
        if span > bound + 1e-6:
            problems.append(
                f"{arm['name']}: {legs} legs integrate to {span:.0f} s but "
                f"the case is bound to {bound:.0f} s of boundary data")
        else:
            notes.append(
                f"{arm['name']}: {legs} legs integrate to {span:.0f} s "
                f"inside {bound:.0f} s of boundary data "
                f"({bound - span:.0f} s of margin)")

    # The verification set must have one volume per free leg.
    for arm in plan["arms"]:
        if arm["name"] != "analyse":
            continue
        argv = arm["steps"][0]["argv"]
        free_legs = int(value_after(cycling[0]["steps"][0]["argv"],
                                    "--free-legs"))
        verify = values_after(argv, "--obs")
        if len(verify) != free_legs:
            problems.append(
                f"analyse: {len(verify)} verification volume(s) against "
                f"{free_legs} free leg(s)")
        missing = [path for path in verify if not Path(path).exists()]
        if missing:
            problems.append(f"analyse: missing verification volume(s): "
                            f"{missing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_background_ab.validate_plan",
        description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    problems: list[str] = []
    notes: list[str] = []
    checked = 0

    for arm in plan["arms"]:
        for step in arm["steps"]:
            where = f"{arm['name']}/{step['name']}"
            raw = substitute(step["argv"], run_dir=args.run_dir,
                             repo=args.repo)
            module = raw[2]
            tail = raw[3:]
            checked += 1
            if module in SELF_VALIDATING:
                command = [sys.executable, "-m", module,
                           SELF_VALIDATING[module]] + tail
                proc = subprocess.run(command, capture_output=True, text=True,
                                      cwd=args.repo)
                if proc.returncode != 0:
                    problems.append(
                        f"{where}: {module} refused its own argument list: "
                        + (proc.stderr.strip() or proc.stdout.strip()
                           ).splitlines()[-1])
                else:
                    print(f"  OK  {where:<28} {module}")
            elif module in IN_PROCESS:
                try:
                    parse_in_process(module, tail)
                except SystemExit as refusal:
                    problems.append(f"{where}: parser refused: {refusal}")
                    continue
                except Exception as error:               # pragma: no cover
                    problems.append(
                        f"{where}: {error.__class__.__name__}: {error}")
                    continue
                print(f"  OK  {where:<28} {module}")
            else:
                problems.append(f"{where}: no validator for {module}")
                continue
            check_paths(raw, args.run_dir, problems, where)

    check_background_registry(plan, problems, notes)
    check_ensemble(plan, problems, notes)
    check_shared_configuration(plan, problems, notes)

    verdict = {
        "schema": VERDICT_SCHEMA,
        "plan": str(args.plan),
        "run_dir": args.run_dir,
        "host": None,
        "steps_checked": checked,
        "arms": [arm["name"] for arm in plan["arms"]],
        "status": "PASS" if not problems else "REFUSED",
        "problems": problems,
        "notes": notes,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    import platform
    verdict["host"] = platform.node()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=1) + "\n",
                            encoding="utf-8", newline="\n")

    print()
    for note in notes:
        print(f"  ..  {note}")
    if problems:
        print("\nREFUSED:")
        for problem in problems:
            print(f"  -  {problem}")
        return 1
    print(f"\nall {checked} steps across {len(plan['arms'])} arms parse, "
          "every input path exists, and every cross-arm invariant holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
