"""Stage a scored A/B over cadence and radar count, on a prepared case.

Emits a ``gpuwm-da.sweep-plan.v1`` document that ``tools/da_sweep_run.py``
executes unattended.  Nothing here runs a model; this is the plan, and the
whole point is that every configuration is written down, validated, and
readable before a single GPU second is spent.

**Two axes, and the controls that separate them.**

Per-volume cycling and multi-radar are independent changes, and an A/B
that moved both at once against one baseline could not attribute whatever
it found.  Two further confounds have to be separated as well:

* a cadence that follows the radar has *irregular* leg lengths, which the
  cycling driver reaches by chaining one process per cycle through
  ``--save-ensemble``/``--resume-ensemble`` rather than by one process
  with one ``--leg-seconds``.  So there is a fixed-cadence arm run through
  the SAME chaining, and the difference between it and the single-process
  baseline is the cost of chaining alone;
* the inflation and observation-error settings were tuned at 15 minutes.
  There is a per-volume arm with them scaled (see
  :mod:`gpuwm.da.cadence` for the argument) and one with them held at the
  old constants, so "we changed the cadence" and "we changed the tuning"
  are two measurements rather than one.

**Why chaining rather than a new driver flag.**  ``da_cycle_prepared.py``
already carries an ensemble across process boundaries, precisely because
"the next radar volume does not exist when this one is assimilated" is
the nowcast's own problem.  A cadence that follows the antenna is that
same mechanism with the leg length read off the feed, so per-volume
cycling needs no change to the cycling driver at all -- the cadence lives
here, in the plan.

No site names and no case names: sites, times and paths are arguments.
EXPERIMENTAL, like everything it drives.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gpuwm.da.cadence import (CadenceError, check_overrun, plan_fixed,
                              plan_per_volume, scaled_settings)

SCHEMA = "gpuwm-da.sweep-plan.v1"

#: The published anchor this A/B has to land beside.  Carried into the
#: plan so a reader never has to go looking for what "beside tonight's
#: baseline" meant, and so a baseline arm that fails to reproduce it is
#: visible as a discrepancy rather than as a new number.
PUBLISHED_ANCHOR = {
    "run": "evidence/da-demo/live-fire-3",
    "fss30_fcst": [0.7274, 0.7557, 0.7655, 0.7376, 0.7316, 0.7239],
    "fss30_control": [0.236, 0.2719, 0.3173, 0.3156, 0.3205, 0.341],
    "note": ("tools/da_sweep_score.py reproduces all six exactly from the "
             "committed composites; the baseline arm below re-runs the "
             "configuration that produced them"),
}

#: Neighborhood sides to trace, km.  One box is a point on a curve chosen
#: in advance; FSS rises monotonically with the box and reaches 1 when it
#: covers the domain, so the shape is the result and 27 km is one row.
NEIGHBORHOODS_KM = (15.0, 27.0, 51.0, 75.0, 135.0)


class _Parsed(Exception):
    """Raised the instant a real parser has accepted an argument list."""

    def __init__(self, namespace):
        self.namespace = namespace


def _through_real_parser(module: str, argv: list[str]):
    """Push one step's argv through the REAL parser of the tool it names.

    Not a copy of the parser and not a schema of it: the entry point's own
    ``main`` is called with ``parse_args`` intercepted, so the grammar
    that validates here is the grammar that will run.  A plan that cannot
    be parsed must never reach a queue that would discover it at 3 a.m.
    """

    import importlib

    entry = importlib.import_module(module)
    real = argparse.ArgumentParser.parse_args

    def intercept(self, args=None, namespace=None):
        raise _Parsed(real(self, args if args is not None else argv[3:],
                           namespace))

    argparse.ArgumentParser.parse_args = intercept
    try:
        # Called with no arguments on purpose: these entry points differ
        # (``da_cycle_prepared.main()`` takes none and reads ``sys.argv``,
        # the others take an optional list), and with none passed they all
        # reach ``parse_args(None)``, which the intercept above answers
        # from this step's own argv.  One call shape, every tool.
        entry.main()
    except _Parsed as done:
        return done.namespace
    finally:
        argparse.ArgumentParser.parse_args = real
    raise RuntimeError(f"{module}.main() returned without parsing")


#: Flags naming an input that must already exist.  Outputs must not, and a
#: ``${RUN_DIR}`` path is produced by an earlier step of the same arm.
_INPUT_FLAGS = {"--prepared-root", "--authority-dir", "--obs",
                "--grid-wrfout", "--composites", "--obs-dir"}


def _grid_binding_problems(namespace, where: str) -> list[str]:
    """Does each observation file actually carry the grid it is paired with?

    This is the check that would have caught the plan whose six legs all
    named one history file: a grid identity digests ``z_w``, so two
    history files from the same domain at different times are DIFFERENT
    grids, and the driver refuses the mismatch -- one second into the arm,
    after the queue has already waited for the card.

    Only files that ALREADY exist can be checked here; the ones this plan
    is going to build are covered by the obs-build/cycle pairing check
    instead.  Reading a grid identity means opening a wrfout, so results
    are cached across the legs and arms that share one.
    """

    problems: list[str] = []
    for obs, wrfout in zip(namespace.obs, namespace.grid_wrfout):
        obs, wrfout = Path(obs), Path(wrfout)
        if "${RUN_DIR}" in str(obs) or not obs.is_file():
            continue
        if not wrfout.is_file():
            continue
        try:
            import netCDF4
            with netCDF4.Dataset(str(obs)) as ds:
                bound = ds.getncattr("grid_identity_sha256")
            required = _grid_identity(wrfout)
        except Exception as error:              # pragma: no cover
            problems.append(f"{where}: could not compare {obs.name} with "
                            f"{wrfout.name}: {error}")
            continue
        if bound != required:
            problems.append(
                f"{where}: {obs.name} is bound to grid {bound[:16]}... but "
                f"is paired with {wrfout.name}, which hashes "
                f"{required[:16]}.... A grid identity digests z_w, so two "
                "history files at different times are different grids and "
                "the driver will refuse this pairing")
    return problems


def _filter_config_problems(namespace, where: str) -> list[str]:
    """Would the FILTER accept the knobs this step is about to pass it?

    Built by constructing the real ``RadarAssimilationConfig`` rather than
    re-stating its rules, so the plan is checked against the constraint
    that will actually run and not against a copy of it that can drift.

    This is the check that would have caught a scaled observation-error
    inflation of 0.9129 on a per-volume leg longer than the tuned
    baseline: the filter refuses anything below 1 because deflating a
    stated observation error is a claim of skill nobody measured, and it
    refused it forty-nine seconds into the arm rather than here.
    """

    try:
        from gpuwm.da.letkf import Localization
        from gpuwm.da.radar_assimilation import (RadarAssimilationConfig,
                                                 RadarAssimilationError)
    except Exception:                               # pragma: no cover
        return []
    try:
        RadarAssimilationConfig(
            localization=Localization(
                horizontal_m=namespace.horizontal_loc_m,
                vertical_m=namespace.vertical_loc_m),
            rtps_alpha=namespace.rtps_alpha,
            relaxation=namespace.relaxation,
            analysis_fields=("u", "v"),
            velocity=True, reflectivity=False, fall_speed="none",
            velocity_thinning_cells=namespace.thin_cells,
            velocity_error_inflation=namespace.err_inflation,
            reflectivity_thinning_cells=namespace.z_thin_cells,
            reflectivity_error_inflation=namespace.z_err_inflation,
            positivity_policy=namespace.positivity_policy,
            solve_device=namespace.solve_device,
            memory_budget_mib=namespace.memory_budget_mib)
    except (RadarAssimilationError, ValueError) as refusal:
        return [f"{where}: the filter would refuse this configuration: "
                f"{refusal}"]
    return []


_GRID_IDENTITY_CACHE: dict[str, str] = {}


def _grid_identity(wrfout: Path) -> str:
    key = str(wrfout)
    if key not in _GRID_IDENTITY_CACHE:
        from gpuwm.obs.target_grid import TargetGrid
        _GRID_IDENTITY_CACHE[key] = TargetGrid.from_wrfout(
            wrfout).identity_sha256()
    return _GRID_IDENTITY_CACHE[key]


def validate_plan(plan: dict) -> list[str]:
    """Every step's argv through its own parser; every input path checked.

    Returns the problems.  Empty means every step would at least start --
    not that it will finish: a full card, an S3 outage or a physics
    refusal are all still ahead.  This is the class of failure that is
    knowable now and cheap to fix now.
    """

    problems: list[str] = []
    #: (arm, obs path) -> the georeference the build step will grid onto.
    built: dict[tuple[str, object], object] = {}
    for arm in plan["arms"]:
        for step in arm["steps"]:
            argv = list(step["argv"])
            module = argv[2]
            try:
                namespace = _through_real_parser(module, argv)
            except SystemExit as refusal:
                problems.append(
                    f"{arm['name']}/{step['name']}: {module} refused: "
                    f"{refusal}")
                continue
            except Exception as error:              # pragma: no cover
                problems.append(
                    f"{arm['name']}/{step['name']}: {module}: "
                    f"{error.__class__.__name__}: {error}")
                continue
            if module == "tools.da_cycle_prepared":
                if len(namespace.obs) != len(namespace.grid_wrfout):
                    problems.append(
                        f"{arm['name']}/{step['name']}: {len(namespace.obs)}"
                        f" --obs against {len(namespace.grid_wrfout)} "
                        "--grid-wrfout; the driver pairs them by position")
                problems += _grid_binding_problems(
                    namespace, f"{arm['name']}/{step['name']}")
                problems += _filter_config_problems(
                    namespace, f"{arm['name']}/{step['name']}")
            if module == "tools.obs_radar_grid_build":
                built[(arm["name"], namespace.out)] = namespace.grid_wrfout
            for flag, value in zip(argv, argv[1:]):
                if flag not in _INPUT_FLAGS or "${RUN_DIR}" in value:
                    continue
                if not Path(value.replace("${REPO}", ".")).exists():
                    problems.append(
                        f"{arm['name']}/{step['name']}: {flag} points at a "
                        f"missing path: {value}")
    # An observation file this plan BUILDS cannot be opened yet, so the
    # identity check above cannot see it.  What can be checked is that the
    # build step and the cycle step that consumes it name the SAME
    # georeference -- which is the same bug one step earlier.
    for arm in plan["arms"]:
        for step in arm["steps"]:
            if step["argv"][2] != "tools.da_cycle_prepared":
                continue
            argv = step["argv"]
            obs = [v for f, v in zip(argv, argv[1:]) if f == "--obs"]
            refs = [v for f, v in zip(argv, argv[1:])
                    if f == "--grid-wrfout"]
            for path, wrfout in zip(obs, refs):
                promised = built.get((arm["name"], Path(path)))
                if promised is None:
                    continue
                if Path(promised) != Path(wrfout):
                    problems.append(
                        f"{arm['name']}/{step['name']}: {Path(path).name} "
                        f"is BUILT onto {Path(promised).name} but the "
                        f"cycle pairs it with {Path(wrfout).name}; the "
                        "driver binds an observation file to the "
                        "georeference it was gridded onto and will refuse "
                        "this")
    return problems


#: History filenames carry their own valid time; this reads it rather
#: than counting files, so a directory holding a longer run than the
#: case still resolves correctly.
_WRFOUT_STEM = "wrfout_d01_"


def wrfout_index(directory: Path) -> list[tuple[datetime, Path]]:
    """Every history file in a directory, by the time it is valid at."""

    found: list[tuple[datetime, Path]] = []
    for path in sorted(Path(directory).glob(f"{_WRFOUT_STEM}*")):
        stamp = path.name[len(_WRFOUT_STEM):]
        try:
            when = datetime.strptime(stamp, "%Y-%m-%d_%H_%M_%S").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        found.append((when, path))
    if not found:
        raise SystemExit(
            f"no {_WRFOUT_STEM}* history files under {directory}; the "
            "georeference every observation file is gridded onto has to "
            "come from somewhere and this tool will not invent one")
    return found


def nearest_wrfout(index, when: datetime, *,
                   max_offset_seconds: float) -> tuple[Path, float]:
    """The history file whose georeference a cycle at ``when`` uses.

    **Why this is per leg and not per run.**  A grid identity digests
    ``z_w`` -- the model's own terrain-following layer-interface heights
    -- so two history files from the same domain at different times are
    DIFFERENT grids.  The cycling driver binds each observation file to
    the georeference it was gridded onto and refuses a mismatch, which is
    exactly right and is why one wrfout cannot serve six legs.

    **What the offset costs.**  History lands on its own interval and a
    cadence that follows the radar does not, so a per-volume cycle is
    georeferenced by the nearest history file rather than by one at its
    own instant.  The error this introduces is in the layer heights the
    observations are placed against, and it is the same KIND of error the
    driver already documents for members: every member's own column
    heights differ from the shared georeference by its own perturbation,
    which is representativeness error rather than a binding error.  The
    offset is recorded per cycle so its size is never a guess.
    """

    when = when.astimezone(timezone.utc)
    best, offset = min(
        ((path, (stamp - when).total_seconds()) for stamp, path in index),
        key=lambda pair: abs(pair[1]))
    if abs(offset) > float(max_offset_seconds):
        raise SystemExit(
            f"the nearest history file to {_iso(when)} is {best.name}, "
            f"{offset:+.0f} s away, beyond the "
            f"{float(max_offset_seconds):.0f} s ceiling. A cycle "
            "georeferenced by a history file that far from its own "
            "analysis time places observations against layer heights the "
            "model no longer has")
    return best, offset


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _iso(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def list_volume_times(binary, *, site: str, start: datetime, end: datetime,
                      bucket: str | None) -> list[datetime]:
    """Every volume the archive holds for a site in a window.

    A listing moves no payload, so asking what the radar actually did is
    cheap.  MDM objects are metadata, not volumes, and are excluded here
    for the same reason the feed selector excludes them.
    """

    from gpuwm.obs.nexrad import run_list

    listing = run_list(binary, site=site, start=_iso(start), end=_iso(end),
                       bucket=bucket)
    return sorted(
        _parse_iso(volume["valid_time"])
        for volume in listing.get("volumes", ())
        if not volume["filename"].endswith("MDM"))


def obs_step(*, name, sites, discover, valid_time, grid_wrfout, out, work,
             source, extra=()) -> dict:
    """One observation file: fetch, decode, verify, superob, grid, receipt."""

    argv = ["${PYTHON}", "-m", "tools.obs_radar_grid_build",
            "--valid-time", _iso(valid_time),
            "--grid-wrfout", str(grid_wrfout),
            "--out", str(out), "--work-dir", str(work),
            "--source", source, "--overwrite"]
    if discover:
        argv += ["--discover-sites"]
    for site in sites:
        argv += ["--site", site]
    argv += list(extra)
    return {"name": name, "argv": argv}


def cycle_step(*, name, common, leg_seconds, obs, grid_wrfout, out,
               stage_dir, rtps_alpha, err_inflation, leg_number,
               resume=None, save=None) -> dict:
    """One assimilation cycle, as its own process.

    The ensemble enters through ``--resume-ensemble`` and leaves through
    ``--save-ensemble``, which is how a cadence with irregular legs is
    expressed without the driver needing a list of leg lengths.
    """

    argv = ["${PYTHON}", "-m", "tools.da_cycle_prepared", *common,
            "--leg-seconds", f"{float(leg_seconds):.1f}",
            "--obs", str(obs), "--grid-wrfout", str(grid_wrfout),
            "--rtps-alpha", f"{float(rtps_alpha):.6f}",
            "--err-inflation", f"{float(err_inflation):.6f}",
            "--leg-number-offset", str(int(leg_number)),
            "--save-composites",
            "--out", str(out), "--stage-dir", str(stage_dir)]
    if resume is not None:
        argv += ["--resume-ensemble", str(resume)]
    if save is not None:
        argv += ["--save-ensemble", str(save)]
    return {"name": name, "argv": argv}


def free_step(*, name, common, free_legs, free_leg_seconds, out, stage_dir,
              resume, leg_number) -> dict:
    """The free forecast, branching off the last analysis."""

    return {"name": name, "argv": [
        "${PYTHON}", "-m", "tools.da_cycle_prepared", *common,
        "--leg-seconds", f"{float(free_leg_seconds):.1f}",
        "--free-legs", str(int(free_legs)),
        "--free-leg-seconds", f"{float(free_leg_seconds):.1f}",
        "--leg-number-offset", str(int(leg_number)),
        "--resume-ensemble", str(resume), "--save-composites",
        "--out", str(out), "--stage-dir", str(stage_dir)]}


def score_step(*, name, composites, obs_dir, first_free_leg, dx_km, label,
               out) -> dict:
    argv = ["${PYTHON}", "-m", "tools.da_sweep_score"]
    for directory in composites:
        argv += ["--composites", str(directory)]
    argv += ["--obs-dir", str(obs_dir),
             "--first-free-leg", str(int(first_free_leg)),
             "--dx-km", f"{float(dx_km):g}", "--label", label,
             "--out", str(out)]
    for box in NEIGHBORHOODS_KM:
        argv += ["--neighborhood-km", f"{box:g}"]
    return {"name": name, "argv": argv}


def build_arm(*, name, family, what, plan, common, case, scaling,
              sites, discover, obs_tag, needs_obs=True) -> dict:
    """One arm: build its observations, cycle it, run it free, score it."""

    run = "${RUN_DIR}/" + name
    steps: list[dict] = []
    obs_dir = (f"${{RUN_DIR}}/obs/{obs_tag}" if needs_obs
               else str(case["existing_obs_dir"]))

    # -- observations, one per analysis time -------------------------------
    obs_paths = []
    georef: list[tuple[str, float]] = []
    for index, cycle in enumerate(plan.cycles):
        # ONE georeference per leg, used by the observation build AND by
        # the cycle that consumes it.  A grid identity digests z_w, so
        # two history files are two different grids and the driver
        # refuses a mismatch -- which is what makes pairing them here,
        # once, the only safe way to write this.
        wrfout, offset = nearest_wrfout(
            case["wrfout_index"], cycle.valid_time,
            max_offset_seconds=case["wrfout_max_offset_s"])
        georef.append((str(wrfout), offset))
        if needs_obs:
            path = f"{obs_dir}/cycle{index:02d}.nc"
            steps.append(obs_step(
                name=f"obs{index:02d}", sites=sites, discover=discover,
                valid_time=cycle.valid_time,
                grid_wrfout=wrfout, out=path,
                work=f"${{RUN_DIR}}/volumes", source=case["source_feed"],
                extra=case["obs_extra"]))
        else:
            path = case["existing_obs"][index]
        obs_paths.append(path)

    # -- the cycles, chained through the ensemble generation ---------------
    composites = []
    previous_generation = None
    for index, cycle in enumerate(plan.cycles):
        block = scaled_settings(
            cycle_interval_s=cycle.leg_seconds,
            baseline_interval_s=case["baseline_interval_s"],
            rtps_alpha=case["rtps_alpha"],
            error_inflation=case["err_inflation"],
            horizontal_loc_m=case["horizontal_loc_m"],
            vertical_loc_m=case["vertical_loc_m"],
            scaling=scaling)
        out = f"{run}/cycle{index:02d}"
        generation = f"{run}/gen{index:02d}"
        steps.append(cycle_step(
            name=f"cycle{index:02d}", common=common,
            leg_seconds=cycle.leg_seconds, obs=obs_paths[index],
            grid_wrfout=georef[index][0], out=out,
            stage_dir=f"{run}/stage",
            rtps_alpha=block["applied"]["rtps_alpha"],
            err_inflation=block["applied"]["error_inflation"],
            leg_number=index, resume=previous_generation, save=generation))
        composites.append(f"{out}/composites")
        previous_generation = generation

    # -- the free forecast, then the score ---------------------------------
    free_out = f"{run}/free"
    steps.append(free_step(
        name="free", common=common, free_legs=case["free_legs"],
        free_leg_seconds=case["free_leg_seconds"], out=free_out,
        stage_dir=f"{run}/stage", resume=previous_generation,
        leg_number=len(plan.cycles)))
    composites.append(f"{free_out}/composites")
    steps.append(score_step(
        name="score", composites=composites, obs_dir=case["verify_obs_dir"],
        first_free_leg=len(plan.cycles), dx_km=case["dx_km"], label=name,
        out=f"${{RUN_DIR}}/results/{name}.json"))

    # The scaling block for the MEAN interval, recorded once per arm as the
    # human-readable summary; the per-cycle values above are what runs.
    summary = scaled_settings(
        cycle_interval_s=plan.mean_interval_seconds,
        baseline_interval_s=case["baseline_interval_s"],
        rtps_alpha=case["rtps_alpha"], error_inflation=case["err_inflation"],
        horizontal_loc_m=case["horizontal_loc_m"],
        vertical_loc_m=case["vertical_loc_m"], scaling=scaling)
    return {
        "name": name, "family": family, "what": what,
        "needs_gpu": True,
        "radars": {"selection": "discovered" if discover else "named",
                   "named": list(sites),
                   "count_expected": (None if discover else len(sites))},
        "cadence": plan.to_payload(),
        "georeference": [
            {"cycle": index, "wrfout": path,
             "offset_seconds_from_analysis": round(offset, 1)}
            for index, (path, offset) in enumerate(georef)],
        "tuning": summary,
        "per_cycle_tuning": [
            {"cycle": index,
             "leg_seconds": cycle.leg_seconds,
             "rtps_alpha": scaled_settings(
                 cycle_interval_s=cycle.leg_seconds,
                 baseline_interval_s=case["baseline_interval_s"],
                 rtps_alpha=case["rtps_alpha"],
                 error_inflation=case["err_inflation"],
                 horizontal_loc_m=case["horizontal_loc_m"],
                 vertical_loc_m=case["vertical_loc_m"],
                 scaling=scaling)["applied"]["rtps_alpha"],
             "error_inflation": scaled_settings(
                 cycle_interval_s=cycle.leg_seconds,
                 baseline_interval_s=case["baseline_interval_s"],
                 rtps_alpha=case["rtps_alpha"],
                 error_inflation=case["err_inflation"],
                 horizontal_loc_m=case["horizontal_loc_m"],
                 vertical_loc_m=case["vertical_loc_m"],
                 scaling=scaling)["applied"]["error_inflation"]}
            for index, cycle in enumerate(plan.cycles)],
        "steps": steps,
    }


def release_gate_arm(*, status_file, restore, name="Z0-release-gate",
                     quiet_seconds=180.0, max_hours=8.0) -> dict:
    """A wait, stated in the plan instead of hidden in a launcher.

    Another lane's GPU suite has priority on the card, and both queues
    were waiting on the SAME handover file -- so whichever noticed first
    would start while the other was still arming, which is the contention
    trap that turns a release suite red for reasons that have nothing to
    do with the release.

    Declaring it as an arm rather than as a sleep inside a launcher buys
    three things: it is visible to anyone reading the plan, it is
    validated with every other step before the plan is emitted, and it
    writes the same .done marker as any arm so a rerun skips it. It
    declares needs_gpu false, so it never holds the card while waiting
    for the card.
    """

    argv = ["${PYTHON}", "-m", "tools.da_release_gate",
            "--status-file", str(status_file),
            "--quiet-seconds", f"{float(quiet_seconds):g}",
            "--max-hours", f"{float(max_hours):g}",
            "--poll-seconds", "30",
            "--log", "${RUN_DIR}/release-gate.log"]
    for spec in restore:
        argv += ["--restore-if-missing", spec]
    return {
        "name": name,
        "family": "sequencing",
        "needs_gpu": False,
        "what": ("hold every GPU arm below until the release cut's own "
                 "status trail reaches a terminal marker and stops "
                 "changing; the card order is the owner's, not this "
                 "queue's"),
        "steps": [{"name": "wait", "argv": argv}],
    }


def baseline_arm(*, name, common, case, obs_paths, georef) -> dict:
    """Tonight's configuration, in tonight's shape: ONE process.

    Not chained, not re-tuned, not rebuilt -- it reuses the observation
    files the published run assimilated, so if this arm does not land on
    ``PUBLISHED_ANCHOR`` something moved underneath the whole comparison
    and every other arm is suspect.
    """

    run = "${RUN_DIR}/" + name
    argv = ["${PYTHON}", "-m", "tools.da_cycle_prepared", *common,
            "--leg-seconds", f"{float(case['baseline_interval_s']):.1f}",
            "--free-legs", str(int(case["free_legs"])),
            "--rtps-alpha", f"{float(case['rtps_alpha']):.6f}",
            "--err-inflation", f"{float(case['err_inflation']):.6f}",
            "--save-composites",
            "--out", f"{run}/cycle-out", "--stage-dir", f"{run}/stage"]
    # Pairs by POSITION, so the two loops must stay one loop: an obs
    # file and the georeference it was gridded onto travel together.
    for path, wrfout in zip(obs_paths, georef):
        argv += ["--obs", str(path), "--grid-wrfout", str(wrfout)]
    return {
        "name": name, "family": "baseline (exact control)",
        "needs_gpu": True,
        "what": ("the shipped configuration: one radar, one volume every "
                 "900 s, one process, RTPS 0.9 -- the arm every other "
                 "number is read against"),
        "radars": {"selection": "named", "named": [case["baseline_site"]],
                   "count_expected": 1},
        "cadence": {"mode": "fixed", "cycle_count": len(obs_paths),
                    "interval_seconds": {"min": case["baseline_interval_s"],
                                         "max": case["baseline_interval_s"],
                                         "mean": case["baseline_interval_s"]}},
        "tuning": {"scaling": "none (this IS the tuning)",
                   "applied": {"rtps_alpha": case["rtps_alpha"],
                               "error_inflation": case["err_inflation"],
                               "horizontal_loc_m": case["horizontal_loc_m"],
                               "vertical_loc_m": case["vertical_loc_m"]}},
        "steps": [
            {"name": "cycle", "argv": argv},
            score_step(name="score", composites=[f"{run}/cycle-out/composites"],
                       obs_dir=case["verify_obs_dir"],
                       first_free_leg=len(obs_paths), dx_km=case["dx_km"],
                       label=name, out=f"${{RUN_DIR}}/results/{name}.json"),
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_cycle_plan",
        description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True,
                        help="the radar the baseline cycles on, and the "
                             "one whose volume times set the per-volume "
                             "cadence. An argument; there is no default")
    parser.add_argument("--multi-site", action="append", default=[],
                        help="a radar the multi-radar arms add. Repeatable. "
                             "Omit to have the arms discover their own from "
                             "the georeference")
    parser.add_argument("--anchor", required=True,
                        help="case time zero (ISO-8601 UTC): the instant "
                             "elapsed_seconds is measured from")
    parser.add_argument("--window-start", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--cycles", type=int, default=6,
                        help="fixed-cadence cycle count (default 6)")
    parser.add_argument("--cycle-seconds", type=float, default=900.0)
    parser.add_argument("--dt-seconds", type=float, required=True,
                        help="the model timestep; analysis times are "
                             "snapped to its lattice")
    parser.add_argument("--dx-km", type=float, required=True)
    parser.add_argument("--free-legs", type=int, default=6)
    parser.add_argument("--free-leg-seconds", type=float, default=900.0)
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--cycle-cost-seconds", type=float, required=True,
                        help="MEASURED wall time of one assimilation cycle "
                             "on the card that will run this. The overrun "
                             "check compares the cadence against it")
    parser.add_argument("--overrun-policy", default="refuse",
                        choices=("refuse", "skip", "queue"))
    # -- the prepared authority, passed through verbatim -------------------
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--authority-dir", required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--prepared-content-sha256", required=True)
    parser.add_argument("--physics-profile", required=True)
    parser.add_argument("--run-seconds", type=float, required=True)
    parser.add_argument("--history-interval-seconds", type=float,
                        default=900.0)
    parser.add_argument("--background-source", default="gfs")
    parser.add_argument("--wrfout-dir", type=Path, required=True,
                        help="directory of history files. Each cycle is "
                             "georeferenced by the one NEAREST its own "
                             "analysis time, because a grid identity "
                             "digests z_w and two history files at "
                             "different times are different grids")
    parser.add_argument("--wrfout-max-offset-seconds", type=float,
                        default=480.0,
                        help="refuse a cycle whose nearest history file is "
                             "further than this from its analysis time")
    parser.add_argument("--existing-obs", action="append", default=[],
                        help="the baseline arm's observation files, in leg "
                             "order -- the ones the published run used, so "
                             "the control reproduces rather than rebuilds")
    parser.add_argument("--verify-obs-dir", required=True)
    parser.add_argument("--source-feed", default="archive",
                        choices=("live", "archive", "auto"))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--horizontal-loc-m", type=float, default=12000.0)
    parser.add_argument("--vertical-loc-m", type=float, default=3000.0)
    parser.add_argument("--rtps-alpha", type=float, default=0.9)
    parser.add_argument("--err-inflation", type=float, default=1.0)
    parser.add_argument("--wind-sigma-ms", type=float, default=1.5)
    parser.add_argument("--length-scale-km", type=float, default=50.0)
    parser.add_argument("--thin-cells", type=int, default=2)
    parser.add_argument("--solve-device", default="cuda")
    parser.add_argument("--memory-budget-mib", type=float, default=6144.0)
    parser.add_argument("--min-coverage-fraction", type=float, default=0.30)
    parser.add_argument("--max-radars", type=int, default=3)
    parser.add_argument("--max-radar-time-spread-seconds", type=float,
                        default=420.0)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--release-gate-status", type=Path, default=None,
                        help="another lane's status trail that must reach "
                             "a terminal marker before any GPU arm here "
                             "starts. Emits a needs_gpu=false first arm "
                             "that waits for it")
    parser.add_argument("--release-gate-restore", action="append",
                        default=[], metavar="SRC=DST",
                        help="passed to the gate arm: restore a handover "
                             "file a sibling queue renamed while another "
                             "lane was still waiting on it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    anchor = _parse_iso(args.anchor)
    window_start = _parse_iso(args.window_start)
    window_end = _parse_iso(args.window_end)

    from gpuwm.obs.nexrad import find_nexrad_bin, nexrad_remedy
    binary = find_nexrad_bin()
    if binary is None:
        raise SystemExit(f"no rw_nexrad front door: {nexrad_remedy()}")

    # -- what the radar actually did, asked rather than assumed ------------
    volume_times = list_volume_times(
        binary, site=args.site, start=anchor - timedelta(seconds=1800),
        end=window_end + timedelta(seconds=600), bucket=args.bucket)
    if not volume_times:
        raise SystemExit(
            f"the archive lists no volume for {args.site} between "
            f"{_iso(anchor)} and {_iso(window_end)}; there is no cadence to "
            "plan and inventing a nominal VCP period would be inventing data")

    try:
        fixed = plan_fixed(volume_times, anchor=anchor,
                           interval_seconds=args.cycle_seconds,
                           cycles=args.cycles, dt_seconds=args.dt_seconds)
        per_volume = plan_per_volume(
            volume_times, anchor=anchor, dt_seconds=args.dt_seconds,
            window_start=window_start, window_end=window_end)
        per_volume, overrun = check_overrun(
            per_volume, cycle_cost_seconds=args.cycle_cost_seconds,
            policy=args.overrun_policy)
    except CadenceError as error:
        raise SystemExit(str(error)) from error

    existing = [str(p) for p in args.existing_obs]
    if existing and len(existing) != len(fixed.cycles):
        raise SystemExit(
            f"--existing-obs was given {len(existing)} time(s) but the "
            f"fixed cadence has {len(fixed.cycles)} cycles; the baseline "
            "arm pairs them by position and a mismatch would silently "
            "score a different run")

    common = [
        "--prepared-root", args.prepared_root,
        "--authority-dir", args.authority_dir,
        "--source", args.background_source,
        "--proof-sha256", args.proof_sha256,
        "--source-manifest-sha256", args.source_manifest_sha256,
        "--prepared-content-sha256", args.prepared_content_sha256,
        "--physics-profile", args.physics_profile,
        "--run-seconds", f"{float(args.run_seconds):.1f}",
        "--history-interval-seconds",
        f"{float(args.history_interval_seconds):.1f}",
        "--members", str(int(args.members)),
        "--seed", str(int(args.seed)),
        "--wind-sigma-ms", f"{float(args.wind_sigma_ms):g}",
        "--length-scale-km", f"{float(args.length_scale_km):g}",
        "--horizontal-loc-m", f"{float(args.horizontal_loc_m):g}",
        "--vertical-loc-m", f"{float(args.vertical_loc_m):g}",
        "--thin-cells", str(int(args.thin_cells)),
        "--memory-budget-mib", f"{float(args.memory_budget_mib):g}",
        "--solve-device", args.solve_device,
    ]

    index = wrfout_index(args.wrfout_dir)
    case = {
        "wrfout_index": index,
        "wrfout_max_offset_s": args.wrfout_max_offset_seconds,
        "verify_obs_dir": args.verify_obs_dir,
        "source_feed": args.source_feed,
        "baseline_site": args.site,
        "baseline_interval_s": args.cycle_seconds,
        "rtps_alpha": args.rtps_alpha,
        "err_inflation": args.err_inflation,
        "horizontal_loc_m": args.horizontal_loc_m,
        "vertical_loc_m": args.vertical_loc_m,
        "free_legs": args.free_legs,
        "free_leg_seconds": args.free_leg_seconds,
        "dx_km": args.dx_km,
        "existing_obs": existing,
        "existing_obs_dir": (str(Path(existing[0]).parent) if existing
                             else None),
        "obs_extra": (["--max-radar-time-spread-seconds",
                       f"{float(args.max_radar_time_spread_seconds):g}"]),
    }
    multi_sites = [s.strip().upper() for s in args.multi_site]
    discover = not multi_sites
    multi_extra = list(case["obs_extra"])
    if discover:
        multi_extra += ["--min-coverage-fraction",
                        f"{float(args.min_coverage_fraction):g}",
                        "--max-radars", str(int(args.max_radars))]
    multi_case = dict(case, obs_extra=multi_extra)

    arms = []
    if args.release_gate_status is not None:
        # FIRST in the list, because da_sweep_run walks arms in plan
        # order: whatever --only names, this one runs before any of them.
        arms.append(release_gate_arm(
            status_file=args.release_gate_status,
            restore=list(args.release_gate_restore)))
    if existing:
        # The baseline reuses the PUBLISHED observation files, so its
        # georeferences are not a choice: each file is already bound to
        # the history file it was gridded onto, and the fixed marks land
        # exactly on the history interval.
        baseline_refs = [
            str(nearest_wrfout(index, cycle.valid_time,
                               max_offset_seconds=1.0)[0])
            for cycle in fixed.cycles]
        arms.append(baseline_arm(name="A0-fixed900-1radar", common=common,
                                 case=case, obs_paths=existing,
                                 georef=baseline_refs))
    arms.append(build_arm(
        name="A1-fixed900-1radar-chained", family="cadence control",
        what=("the same 900 s cadence and the same single radar, driven "
              "through the per-cycle chaining the irregular arms need; the "
              "gap to A0 is the cost of chaining and nothing else"),
        plan=fixed, common=common, case=case, scaling="none",
        sites=[args.site], discover=False, obs_tag="fixed-1radar",
        needs_obs=not existing))
    arms.append(build_arm(
        name="B1-pervolume-1radar-scaled", family="axis 1: cadence",
        what=("one analysis per volume the radar produced, with inflation "
              "and observation error carried to the new cadence by the "
              "documented scaling"),
        plan=per_volume, common=common, case=case, scaling="documented",
        sites=[args.site], discover=False, obs_tag="pervolume-1radar"))
    arms.append(build_arm(
        name="B2-pervolume-1radar-unscaled", family="axis 1: cadence",
        what=("the same per-volume cadence with the 15-minute constants "
              "held FIXED -- the arm that measures what not retuning "
              "costs, rather than assuming it"),
        plan=per_volume, common=common, case=case, scaling="none",
        sites=[args.site], discover=False, obs_tag="pervolume-1radar"))
    arms.append(build_arm(
        name="C1-fixed900-multiradar", family="axis 2: radar count",
        what=("the baseline cadence with every radar covering the domain, "
              "so the multi-radar increment is separated from the cadence "
              "change"),
        plan=fixed, common=common, case=multi_case, scaling="none",
        sites=multi_sites, discover=discover, obs_tag="fixed-multiradar"))
    arms.append(build_arm(
        name="D1-pervolume-multiradar-scaled", family="both axes",
        what="both changes together, with the cadence scaling applied",
        plan=per_volume, common=common, case=multi_case,
        scaling="documented", sites=multi_sites, discover=discover,
        obs_tag="pervolume-multiradar"))

    plan = {
        "schema": SCHEMA,
        "generated": _iso(datetime.now(timezone.utc)),
        "generator": "tools/da_cycle_plan.py",
        "case": {
            "site": args.site,
            "anchor": _iso(anchor),
            "window_start": _iso(window_start),
            "window_end": _iso(window_end),
            "dt_seconds": args.dt_seconds,
            "dx_km": args.dx_km,
            "members": args.members,
            "free_legs": args.free_legs,
            "physics_profile": args.physics_profile,
            "prepared_content_sha256": args.prepared_content_sha256,
            "wrfout_dir": str(args.wrfout_dir),
            "georeference_rule": (
                "each cycle uses the history file nearest its own analysis "
                "time, for the observation build AND the cycle that "
                "consumes it; a grid identity digests z_w so two history "
                "files at different times are different grids"),
        },
        "feed": {
            "listed_volumes": [_iso(stamp) for stamp in volume_times],
            "volume_count": len(volume_times),
            "note": ("the plan is bound to the volumes the archive listed "
                     "when it was generated; a rerun against a changed "
                     "listing is a different plan"),
        },
        "exploitation": {
            # Named precisely: the listing spans a little either side of
            # the cycled window so the nearest-volume search has room, so
            # "listed" and "inside the cycled window" are different counts
            # and conflating them would overstate the case being made.
            "volumes_listed": len(volume_times),
            "listing_window": {
                "start": _iso(anchor - timedelta(seconds=1800)),
                "end": _iso(window_end + timedelta(seconds=600))},
            "assimilated_by_fixed_cadence": len(fixed.cycles),
            "discarded_inside_the_fixed_cycled_window": len(
                fixed.unused_volumes),
            "assimilated_by_per_volume_cadence": len(per_volume.cycles),
            "fraction_of_cycled_window_used_by_fixed_cadence": round(
                len(fixed.cycles)
                / (len(fixed.cycles) + len(fixed.unused_volumes)), 4),
            "note": ("the measurement the first axis exists for: how much "
                     "of what the radar sent the shipped cadence never "
                     "looks at"),
        },
        "overrun": overrun,
        "metric": {
            "function": "gpuwm.verify.field_metrics.fss_distance",
            "threshold_dbz": 30.0,
            "box_km_across": 27.0,
            "convention": ("square SIDE LENGTH, not a radius; "
                           "half_width = round(27/2/dx_km) cells and the "
                           "box is 2*half_width+1 cells across"),
            "truth_smoothing": ("the same boxcar is applied to the "
                                "observation as to the forecast, which "
                                "scores higher than a binary-truth FSS"),
            "scored_field": ("the arithmetic mean over members of each "
                             "member's column-max reflectivity -- one "
                             "deterministic map, NOT an ensemble FSS"),
            "also_reported": ("per-member FSS at the published box, and an "
                              "FSS-versus-neighborhood curve at "
                              f"{list(NEIGHBORHOODS_KM)} km sides"),
            "published_anchor": PUBLISHED_ANCHOR,
        },
        "honesty": [
            "Every arm is a single draw. No arm is repeated, so none of "
            "these numbers carries an error bar, and the standing no-ECC "
            "dual-run screen is NOT applied to any of them.",
            "The per-volume arms make their last analysis at the last "
            "volume before the window end, which is not the same instant "
            "as the fixed arms' last mark; the free forecast therefore "
            "starts up to one volume period earlier and is scored against "
            "the same verification files. The offset is in each arm's "
            "cadence block.",
            "A1 exists because chaining one process per cycle is not "
            "bit-identical to one process running every leg: physics "
            "driver state is re-initialised per leg in both, but the "
            "process boundary is an extra prepared-cache restore. Read B "
            "and D against A1, not against A0.",
            "The multi-radar arms change observation COUNT and observation "
            "DENSITY at once. The filter thins each radar's batch "
            "independently, so the overlap region carries roughly N times "
            "the density of a single-radar analysis; the per-radar cell "
            "counts and the overlap histogram are in every obs receipt.",
            "Radial velocity is masked for aliasing risk, never dealiased. "
            "A spatially coherent fold passes every mask, so the velocity "
            "reaching the filter in these arms is assumed unfolded and is "
            "not proven to be.",
        ],
        "arms": arms,
    }
    # A plan is not emitted until every step in it has been through the
    # real parser of the real tool it names.  A queue must never be the
    # thing that discovers a typo.
    problems = validate_plan(plan)
    if problems:
        for problem in problems:
            print(f"REFUSED: {problem}")
        raise SystemExit(
            f"{len(problems)} problem(s); no plan was written. Every step "
            "is validated against its own entry point's parser before this "
            "tool will emit anything a queue could run")
    plan["validation"] = {
        "checked": "every step's argv through its own tool's real parser, "
                   "and every input path for existence",
        "steps_checked": sum(len(arm["steps"]) for arm in plan["arms"]),
        "result": "all parse; all input paths exist",
        "not_checked": ("GPU admission, S3 availability, physics refusals "
                        "and anything that can only fail at run time"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"wrote {args.out} "
          f"({plan['validation']['steps_checked']} steps validated)")
    print(f"volumes listed {len(volume_times)}; fixed cadence assimilates "
          f"{len(fixed.cycles)}, per-volume {len(per_volume.cycles)}")
    for arm in arms:
        print(f"  {arm['name']:34s} {len(arm['steps']):3d} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
