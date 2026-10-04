"""The front door for time-lagged and multi-model ensemble members.

``gpuwm ensemble CONFIG --recipe time-lagged`` (and ``go`` and ``run``, or
``[ensemble] recipe`` in the config) lands here.  The door owns no model
code and no preparation code:

* the member plan is :func:`gpuwm.ensemble.recipes.build_recipe`, the same
  plan ``python -m gpuwm.ensemble.recipes`` prints;
* each member's source trajectory is fetched and prepared by the ordinary
  preparation chain its source already has
  (:func:`gpuwm.regional_preparation.preparation_chains`), from a copy of
  the config whose ``[fetch]`` table names that trajectory;
* the members run through the existing ensemble session
  (:class:`gpuwm.ensemble.production.PreparedEnsembleSession`), which is
  handed each member's own prepared inputs.

Everything a request can be refused for is decided before the run folder
is claimed and before the first member is fetched: the member plan, every
member's own config and chain plan, and the gates the ordinary door asks
(the card, the memory envelope, the geography tree, the renderer, the
products and the disk).  ``--dry-run`` stops after the plan and the member
configs, and spends nothing.

A recipe member is a real source trajectory.  Nothing here perturbs a
field, so the observation-calibration refusal does not apply to it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

#: The recipes a front door can select.  ``input-ensemble`` is the
#: automatic choice for a source with operational members and
#: ``recentered`` has no observation-calibrated amplitude, so neither is
#: a door option here.
RECIPES = ("time-lagged", "multi-model")

RECEIPT_NAME = "ensemble-recipe.json"
RECEIPT_SCHEMA = "gpuwm-ensemble-recipe-door.v1"
#: Where the traceback of a failure that is not a stage's own exit goes,
#: beside the receipt that names it.
FAILURE_LOG_NAME = "ensemble-recipe-failure.log"


class RecipeRefusal(ValueError):
    """A recipe request this door will not run, in one sentence."""


def requested(args) -> bool:
    """Did the command line ask for a recipe?  The config is asked separately."""
    return getattr(args, "recipe", None) is not None or getattr(args, "trajectories", None) is not None


def configured(config) -> bool:
    """Does the config's own ``[ensemble]`` table select a recipe?"""
    import tomllib

    path = Path(config)
    if not path.is_file():
        return False
    try:
        table = tomllib.loads(path.read_text(encoding="utf-8-sig")).get("ensemble")
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    return isinstance(table, dict) and (table.get("recipe") is not None or bool(table.get("trajectories")))


def refuse_continuation(command, config) -> None:
    """``gpuwm resume`` and ``gpuwm branch`` on a config that selects a recipe.

    Breakage it prevents: a checkpoint holds ONE prepared trajectory, so
    the continued run would be N copies of it under a config that asks
    for N different source trajectories.  The session refused that only
    at the forecast stage, after the card was locked and a worker started.
    """
    if config is None or not configured(config):
        return
    raise RecipeRefusal(
        f"gpuwm {command} continues one prepared trajectory from its checkpoint, and "
        f"{Path(config).name} selects an ensemble recipe, whose members each come from "
        "their own source trajectory: every continued member would be a copy of the "
        "checkpointed one. Next: gpuwm ensemble CONFIG --recipe time-lagged starts a "
        f"recipe ensemble; remove [ensemble] recipe and trajectories to {command} this run")


def refuse_unsupervised(args) -> None:
    """Refuse the ``gpuwm run`` supervision flags a recipe does not consume.

    A recipe ensemble under ``gpuwm run`` runs in the calling process,
    ahead of the supervisor: no worker, no card lock, no fresh-process
    recovery and no checkpoint to continue.  Breakage it prevents: each of
    these flags was parsed and read by nothing, so the run exited 0
    without doing what its command line said, most visibly a ``--gpu-uuid``
    pin on a multi-card host whose members then ran on every visible card.
    """
    given = [(flag, why) for flag, stated, why in (
        ("--restart", getattr(args, "restart", None) is not None,
         "it continues one forecast from its checkpoint, and a recipe ensemble "
         "prepares and starts every member from its own source, so the run "
         "would be a fresh fetch and forecast in place of the continuation"),
        ("--gpu-uuid", getattr(args, "gpu_uuid", None) is not None,
         "it pins and locks one card for a supervised worker, and the members "
         "run in this process on every visible card with no lock, so they "
         "would land on cards the pin excludes (set CUDA_VISIBLE_DEVICES to "
         "the cards this ensemble may use)"),
        ("--supervisor-max-restarts", getattr(args, "supervisor_max_restarts", 3) != 3,
         "there is no supervised worker to restart, so a member that fails "
         "ends the run whatever the count"),
        ("--prep-timeout", getattr(args, "prep_timeout", None) is not None,
         "no supervisor watches a preparation heartbeat on this route, so "
         "the timeout would never fire"),
        ("--allow-shared-gpu", bool(getattr(args, "allow_shared_gpu", False)),
         "no exclusive-card check runs on this route for it to relax"),
        ("--health-debug", bool(getattr(args, "health_debug", False)),
         "the phase health hooks are not handed to the member forecasts, so "
         "the run would report no attribution as if none were needed"),
        ("--preprocess-backend", getattr(args, "preprocess_backend", None) is not None,
         "it pins a [case_data] config's preparation, and each member is "
         "prepared by its source's own chain, which does not read it"),
        ("--directory-input-hash", getattr(args, "directory_input_hash", None) is not None,
         "it sets how a supervised run binds its declared directory inputs, "
         "and this route starts no supervised run to bind them"),
    ) if stated]
    if given:
        raise RecipeRefusal(
            "An ensemble recipe under gpuwm run runs in this process, unsupervised "
            "and with no checkpoint to resume, and does not use "
            + ", ".join(flag for flag, _ in given) + ": "
            + "; ".join(f"{flag}: {why}" for flag, why in given)
            + ". Next: omit " + ("it" if len(given) == 1 else "them") + ".")


def load_trajectories(path):
    """``--trajectories FILE``: a JSON or TOML list of ``{source, cycle[, member]}``.

    The reader is :func:`gpuwm.ensemble.recipes.load_trajectories`, which
    the planning door shares; this door's refusals stay its own class.
    """
    from gpuwm.ensemble.recipes import load_trajectories as load

    return load(path, refusal=RecipeRefusal)


def flag_overrides(args) -> dict:
    """The ``--recipe`` and ``--trajectories`` flags as request overrides."""
    listed = getattr(args, "trajectories", None)
    return {"recipe": getattr(args, "recipe", None),
            "trajectories": None if listed is None else load_trajectories(listed)}


def _utc(value):
    from gpuwm.ensemble.recipes import trajectory_time

    return trajectory_time(value, refusal=RecipeRefusal)


def plan_recipe(request, payload, experiment, *, cycle=None):
    """The member plan for this config, without downloading or touching a card.

    ``cycle`` asks for the same plan at another base cycle, the window
    moved with it and nothing written: ``--cycle`` with ``--readiness``
    answers for that cycle's member windows without re-timing the config.

    A plain member count (no recipe named) whose plan is refused leads with
    what the request would otherwise have been, N copies of one forecast
    (:func:`gpuwm.ensemble.member_inputs.planned_refusal`).  Only the
    plan's own refusals take that lead.  The gates the door asks after the
    plan (the card, the memory envelope, the geography tree, the renderer,
    the disk) are not about where the members come from, and a card
    refusal that opened with "N copies of one forecast" right under a
    printed member plan said the opposite of what had happened.
    """
    try:
        return _member_plan(request, payload, experiment, cycle)
    except RecipeRefusal as refusal:
        from gpuwm.ensemble.member_inputs import needs_member_sources, planned_refusal
        if not needs_member_sources(request):
            raise
        raise RecipeRefusal(planned_refusal(request, refusal)) from refusal


def _member_plan(request, payload, experiment, cycle):
    from gpuwm.ensemble.recipes import SourceTrajectory, build_recipe

    fetch = payload.get("fetch")
    if not isinstance(fetch, dict) or not {"source", "cycle"} <= fetch.keys():
        # Breakage it prevents: a [case_data] config names the local files
        # of ONE trajectory, so every member would run the same inputs and
        # the ensemble would report spread it does not have.
        raise RecipeRefusal(
            "an ensemble recipe fetches and prepares each member's own source "
            "trajectory, and this config has no [fetch] source and cycle. "
            "Next: gpuwm domain --help")
    if len(experiment.domains) != 1:
        # Breakage it prevents: this door prepares and reads one prepared
        # domain per member; a tree's children would all be initialized
        # from the first member's bundle and run as if they were distinct.
        raise RecipeRefusal(
            "an ensemble recipe runs one-domain configs: each member's nests "
            "would need their own inputs from that member's trajectory, and "
            "this door prepares one domain per member. Next: run the recipe "
            "on the outer domain's config")
    start = _utc(experiment.start_time)
    end = start + timedelta(seconds=float(experiment.run_seconds))
    lead = fetch.get("forecast_start_hour", 0)
    base_cycle = _utc(str(fetch["cycle"]))
    if base_cycle + timedelta(hours=int(lead)) != start:
        raise RecipeRefusal(
            "[fetch].cycle plus forecast_start_hour is not [experiment].start_time, "
            "so the base member's own window is ambiguous. Next: regenerate the "
            "config with gpuwm domain")
    if cycle is not None:
        moved = _utc(cycle)
        start, end, base_cycle = start + (moved - base_cycle), end + (moved - base_cycle), moved
    # The config's own source member is the base trajectory's, and a
    # time-lagged roster keeps it at every lagged cycle.  A multi-model
    # roster runs the listed trajectories, each naming its own member.
    member = fetch.get("member") if request.recipe == "time-lagged" else None
    try:
        trajectories = tuple(SourceTrajectory(str(item["source"]), _utc(item["cycle"]), item.get("member"))
                             for item in request.trajectories)
        # A plain member count (no recipe named) is the planner's automatic
        # choice: the operational ensemble the source's adapter row declares.
        return build_recipe(source=str(fetch["source"]), cycle=base_cycle, start=start, end=end,
                            count=request.members, base_seed=request.base_seed,
                            kind=request.recipe or "auto", trajectories=trajectories,
                            member=None if member is None else str(member))
    except RecipeRefusal:
        raise
    except ValueError as error:
        if request.recipe is None:
            from gpuwm.ensemble.member_inputs import no_automatic_members
            raise RecipeRefusal(no_automatic_members(
                str(fetch["source"]), request.members, error)) from error
        raise RecipeRefusal(str(error)) from error


def _crop_box(raw, experiment, source):
    """The domain's own crop box for ``source``, or None when its fetch takes none.

    A member from another model does not inherit the config's ``area``:
    that box was bounded by the config's own source.  It is computed from
    the domain the way ``gpuwm domain`` computes it, bounded by this
    source's coverage, so it exists whether or not the config's own
    ``[fetch]`` table carries one.
    """
    from gpuwm import domain_wizard as wizard

    if not wizard.source_fetch_takes_a_crop_box(source):
        return None
    root = experiment.root.run
    return wizard.fetch_area_hint(raw["projection"], root.nx, root.ny,
                                  source=source, root_dx_m=root.dx)


def member_fetch(raw, experiment, recipe, member) -> dict:
    """The ``[fetch]`` table that acquires one member's trajectory.

    A member of the config's own source keeps the config's table and
    changes its cycle, its start lead and its source member.  A member
    from another model gets that model's own table: its forcing cadence
    and, where its fetch crops at the publisher, the domain's crop box.
    The config's product and provider hints do not carry over to another
    model: that model's own fetch route answers those.
    """
    from gpuwm.source_adapters import get_source_adapter

    trajectory = member.trajectory
    first, last = recipe.acquisition_window(trajectory)
    stamp = trajectory.cycle.strftime("%Y-%m-%dT%H")
    if trajectory.source == recipe.base.source:
        fetch = dict(raw["fetch"])
        fetch["cycle"] = stamp
        fetch.pop("forecast_start_hour", None)
        if first:
            fetch["forecast_start_hour"] = first
    else:
        cadence = get_source_adapter(trajectory.source).forcing_interval_seconds
        fetch = {"source": trajectory.source, "cycle": stamp, "hours": last - first,
                 **({} if cadence is None else {"cadence": max(1, int(cadence) // 3600)}),
                 **({"forecast_start_hour": first} if first else {})}
        area = _crop_box(raw, experiment, trajectory.source)
        if area is not None:
            fetch["area"] = area
    fetch.pop("member", None)
    if trajectory.member is not None:
        fetch["member"] = trajectory.member
    return fetch


def member_label(member) -> str:
    trajectory = member.trajectory
    return (f"member {member.index} ({trajectory.source} {trajectory.cycle:%Y-%m-%dT%H}Z"
            + ("" if trajectory.member is None else f" member {trajectory.member}") + ")")


def member_configuration(raw, config, wps, recipe, member, directory):
    """Write the config and WPS namelist that prepare one member's trajectory.

    The domain, physics and clock are the original document.  Only the
    ``[fetch]`` table changes (:func:`member_fetch`): it names this
    member's source and cycle and the lead at which that cycle reaches the
    shared start time.  A member from another model takes that model's own
    forcing cadence, and its WPS namelist is rewritten for it by the
    ordinary writer.
    """
    from gpuwm.experiment import load_experiment
    from gpuwm.toml_document import emit_experiment_toml

    trajectory = member.trajectory
    original = load_experiment(config)
    changed = deepcopy(raw)
    changed.pop("ensemble", None)
    changed["fetch"] = member_fetch(raw, original, recipe, member)
    if "static" in changed:
        from gpuwm.static.highres_production import parse_static_table
        carrier = parse_static_table(changed["static"], source=str(config),
                                     base_dir=Path(config).resolve().parent)
        if carrier is not None and getattr(carrier, "cache_root", None) is not None:
            changed.setdefault("static", {}).setdefault("highres", {})["cache_root"] = str(
                carrier.cache_root.resolve())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "experiment.toml"
    target.write_text(emit_experiment_toml(changed), encoding="utf-8")
    experiment = load_experiment(target)
    namelist = directory / "experiment.namelist.wps"
    if trajectory.source == recipe.base.source:
        namelist.write_bytes(Path(wps).read_bytes())
    else:
        from gpuwm.companion_domains import candidate_wps_text
        namelist.write_text(candidate_wps_text(changed, original, experiment, target,
                                               original_wps=Path(wps)), encoding="utf-8")
    return target, namelist, experiment


@dataclass(frozen=True)
class MemberPlan:
    """One member, reviewed: its trajectory, its fetch table and its chain."""

    member: object
    fetch: dict
    chain: str
    start_lead: int
    experiment: object
    #: The reviewed copy of the member's config, in the review's scratch
    #: folder: there while the gates are asked, gone before the run.
    config: Path

    @property
    def line(self) -> str:
        trajectory = self.member.trajectory
        return (f"member {self.member.index}: {trajectory.source} "
                f"{trajectory.cycle:%Y-%m-%dT%H}Z"
                + ("" if trajectory.member is None else f" member {trajectory.member}"))


def _posting(options) -> dict:
    """The host pin and posting rule a door carries into every member's fetch."""
    return {key: options[key] for key in ("transport", "as_posted", "late_after_minutes")
            if options.get(key) is not None}


def review_members(raw, config, wps, recipe, *, scratch, options=None):
    """Build every member's config and ask its chain's own plan, before any fetch.

    Breakage it prevents: a member its chain cannot prepare (a table its
    fetch refuses, a host its source cannot pin, a domain outside its
    grid, a chain plan with a missing key) was found only when the loop
    reached it, after every earlier member had been downloaded and
    prepared.  The configs are written to ``scratch``, a folder outside
    the output folder that the caller discards; the run writes them again
    into its own folder.
    """
    import tomllib

    from gpuwm import regional_preparation, runplan
    from gpuwm.go_cli import pinned_transport
    from gpuwm.source_coverage import config_source_coverage_refusal

    options = options or {}
    posting = _posting(options)
    chains = regional_preparation.preparation_chains()
    scratch = Path(scratch)
    plans = []
    for member in recipe.members:
        source = member.trajectory.source
        try:
            target, _namelist, experiment = member_configuration(
                raw, config, wps, recipe, member, scratch / f"member-{member.index:03d}")
            document = tomllib.loads(target.read_text(encoding="utf-8"))
            table = document["fetch"]
            chain = runplan.prepared_chain_for_source(source, source_root=None)
            if chain not in chains:
                raise RecipeRefusal(f"its source's route ({chain}) has no prepare-only chain")
            pinned_transport(table, posting.get("transport"))
            uncovered = config_source_coverage_refusal(experiment, source)
            if uncovered is not None:
                raise RecipeRefusal(uncovered)
            # The chain's own review of the config, read from its row: for
            # the native chain its planner (the keys its stages need, the
            # preparation preconditions and the profile), for the hourly
            # chain the namelists it runs from.
            review = regional_preparation.preparation_chain_reviews().get(chain)
            if review is not None:
                review(target, experiment, raw=document, scratch=scratch, posting=posting)
        except ValueError as error:
            raise RecipeRefusal(
                f"{member_label(member)} cannot be prepared: {error}") from error
        plans.append(MemberPlan(member, table, chain,
                                recipe.acquisition_window(member.trajectory)[0],
                                experiment, target))
    return tuple(plans)


def prepare_member(config, experiment, *, source, directory, downloads, geog_root,
                   recipe_sha256, options=None):
    """Fetch and prepare one trajectory through its source's own chain, and stop."""
    import tomllib

    from gpuwm import regional_preparation, runplan
    from gpuwm.go_cli import fetch_request, managed_download_dir, pin_request, pinned_transport

    posting = _posting(options or {})
    chain = runplan.prepared_chain_for_source(source, source_root=None)
    owner = regional_preparation.preparation_chains()[chain]
    hints = tomllib.loads(Path(config).read_text(encoding="utf-8"))["fetch"]
    # Keyed on the request the chain's fetch stage makes: the model top
    # this config's ladder needs and the host the door pins.
    pinned, _basis = pinned_transport(hints, posting.get("transport"))
    request = pin_request(fetch_request(hints, p_top=experiment.vertical.p_top), pinned)
    run_options = {"geog_root": str(geog_root), "supplement": [],
                   "data_dir": str(managed_download_dir(downloads, request)), **posting}
    plan = SimpleNamespace(run_options=run_options, config_intent={}, sha256=recipe_sha256)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    events = runplan.EventStream(directory / "prepare-events.jsonl", mirror=None)
    try:
        result = owner(plan, config_path=Path(config), exp=experiment,
                       observer=runplan.RunObserver(events), run_dir=directory / "prepare",
                       prepare_only=True)
    finally:
        events.close()
    return {key: result[key] for key in ("prepared_root", "experiment_config", "wps_namelist")}


def member_inputs(shared_inputs, prepared):
    """The ordinary preflight, applied to another member's prepared bundle."""
    from gpuwm import stage_cli
    from gpuwm.prepared_single_domain_forecast import preflight_prepared_forecast

    root = Path(prepared["prepared_root"]).resolve()
    bundle = stage_cli.resolve_bundle(root)
    digests = stage_cli.single_domain_digests(bundle)
    arguments = dict(shared_inputs.preflight_arguments)
    arguments.update(source=bundle["source"], prepared_root=root,
                     proof_sha256=digests["proof"],
                     source_manifest_sha256=digests["source_manifest"],
                     prepared_content_sha256=digests["prepared_content"],
                     experiment_config=Path(prepared["experiment_config"]),
                     wps_namelist=Path(prepared["wps_namelist"]))
    return preflight_prepared_forecast(**arguments)


def _say(text):
    print(f"ensemble: {text}", flush=True)


def disk_refusal(plans, *, case_root, request, options=None) -> str | None:
    """Why this ensemble's disk cannot hold it, or None: N downloads and N bundles.

    Breakage it prevents: every member downloads its own window and
    writes its own prepared bundle, so a disk with room for one run fills
    partway through the members and the ensemble ends with nothing
    usable.  Priced from the tables the single-run admission uses
    (:func:`gpuwm.disk_budget.projected_run_bytes`), one row per member,
    with what a matching download already left in its request cache taken
    off.  Member history is charged only when the request keeps member
    files; the aggregate products have no measured size and are left out.
    """
    from gpuwm import disk_budget, runplan
    from gpuwm.go_cli import fetch_request, managed_download_dir, pin_request, pinned_transport

    posting = _posting(options or {})
    rows, basis = [], []
    for plan in plans:
        pinned, _basis = pinned_transport(plan.fetch, posting.get("transport"))
        fetch = pin_request(fetch_request(plan.fetch, p_top=plan.experiment.vertical.p_top), pinned)
        try:
            folder = managed_download_dir(case_root, fetch)
        except (OSError, ValueError):
            folder = None
        row = disk_budget.projected_run_bytes(
            plan.experiment, keep_checkpoints=None, fetch=fetch, chain=plan.chain,
            render=False, render_products="none",
            download_present_bytes=runplan._present_download_bytes(fetch, folder, True))
        rows.append(row)
        basis.append(f"member {plan.member.index}: {row['download']['basis']}")
    stream = max(rows, key=lambda row: int(row.get("compose_scratch_min_bytes") or 0))
    members = len(rows)
    projection = {
        "download_bytes": sum(int(row["download_bytes"]) for row in rows),
        "preparation_bytes": sum(int(row["preparation_bytes"]) for row in rows),
        "history_bytes": (int(rows[0]["history_bytes"]) * members
                          if request.keep_member_files else 0),
        "checkpoint_bytes": 0, "picture_bytes": 0,
        # One preparation runs at a time and removes its frame stream
        # before the next starts, so the largest stream is the peak.
        "compose_scratch_bytes": int(stream.get("compose_scratch_bytes") or 0),
        "compose_scratch_min_bytes": int(stream.get("compose_scratch_min_bytes") or 0),
        "compose_scratch": stream.get("compose_scratch") or {},
    }
    if (projection["compose_scratch"] or {}).get("composes"):
        from gpuwm.ingest.source_coverage import compose_scratch_override_refusal
        absent = compose_scratch_override_refusal()
        if absent is not None:
            return absent + ". Refused before any download or preparation, so nothing was spent."
    scratch_folder = runplan._compose_scratch_folder(
        runplan._staged_prep_root(Path(case_root)), projection)
    scratch_free = (None if scratch_folder is None
                    or disk_budget.same_disk(scratch_folder, Path(case_root))
                    else disk_budget.free_bytes(scratch_folder))
    refusal = disk_budget.disk_refusal(
        projection, disk_budget.free_bytes(Path(case_root)), scratch_free=scratch_free,
        scratch_folder=scratch_folder, subject=f"this ensemble's {members} members",
        remedy="Free some disk, run fewer members, or name an --outdir on a disk with room")
    if refusal is None:
        return None
    return (refusal[0].upper() + refusal[1:] + ". Refused before the first download, "
            "so nothing was spent. " + "; ".join(basis))


def admit(plans, *, config, geog_root, case_root, request, options=None,
          command="ensemble") -> None:
    """The gates the ordinary door asks before its fetch, asked before N fetches.

    Breakage each prevents is the ordinary door's own (see
    :func:`gpuwm.go_cli._require_forecast_device`,
    :func:`gpuwm.go_cli.memory_gate`,
    :func:`gpuwm.go_cli.geography_refusal`): a card that cannot be opened,
    a config too large for it, a missing geography tree, a missing
    renderer or an unknown product was found on this door only after
    every member had been downloaded and prepared.
    """
    import sys

    from gpuwm import capabilities, go_cli, rustwx

    options = options or {}
    capabilities.require(
        f"gpuwm {command}", capabilities.GPU_RUNTIME,
        before=("Refusing here, before the first member's fetch downloads its "
                "forcing data, rather than at the forecast stage after every "
                "member is prepared."))
    try:
        go_cli._require_forecast_device()
        if not options.get("no_memory_gate"):
            # Once per source: the forecast is the same grid for every
            # member, and the preparation is priced per source.
            seen = set()
            for plan in plans:
                source = plan.member.trajectory.source
                if source in seen:
                    continue
                seen.add(source)
                gate = go_cli.memory_gate(
                    {"config": plan.config, "source": source,
                     "cadence": plan.fetch.get("cadence")}, experiment=plan.experiment)
                _say(f"memory ({source}) -- {gate['verdict']}")
                if gate["refuse"]:
                    raise RecipeRefusal(go_cli.memory_refusal_text(gate))
                if gate["warn"]:
                    print("warning: memory admission has limited headroom or an "
                          f"unmeasured phase; see gpuwm check {config} --explain.",
                          file=sys.stderr, flush=True)
                if gate.get("preparation_warning"):
                    print(f"warning: {gate['preparation_warning']}.",
                          file=sys.stderr, flush=True)
        geography = go_cli.geography_refusal(geog_root)
        if geography is not None:
            raise RecipeRefusal(geography)
        try:
            renderer = rustwx.find_renderer()
        except FileNotFoundError as error:
            raise RecipeRefusal(str(error)) from None
        if renderer is None:
            # The ensemble session draws its aggregate maps through the
            # Rust renderer whatever --products says, and refused its
            # absence only when the forecast started.
            raise RecipeRefusal(
                "the ensemble's aggregate maps are drawn by the Rust renderer, and "
                "this install has none.\n" + rustwx.renderer_remedy())
        go_cli.admit_render_products(options.get("render_products"), section=None)
    except go_cli.GoRefusal as refusal:
        raise RecipeRefusal(str(refusal)) from None
    refusal = disk_refusal(plans, case_root=case_root, request=request, options=options)
    if refusal is not None:
        raise RecipeRefusal(refusal)


def _exit_code(error) -> int | None:
    """The exit code a stage's own failure is worth, or None for any other error.

    75 for a source lead later than its budget, 130 for a stop, and the
    stage's own code otherwise: the codes the ordinary door returns, so a
    scheduler reads a recipe launch the way it reads any other.
    """
    if isinstance(error, KeyboardInterrupt):
        return 130
    if isinstance(error, SystemExit):
        # A stop, as run control treats it (a SIGTERM handler raising
        # SystemExit(128 + N), offline_child_run.ChildStopped): the stop's
        # own code, and 130 when it carries none.
        code = error.code
        return code if isinstance(code, int) and not isinstance(code, bool) and code else 130
    if not isinstance(error, Exception):
        return None
    from gpuwm.go_cli import GoStageFailed

    # ``exit_code`` is how a stage failure states its own code
    # (StageExitError, GoInterrupted, ChainInterrupted, SourceBehind);
    # GoStageFailed spells it ``code``.  Any other ``code`` attribute (an
    # HTTP status) is not a stage's exit.
    code = error.code if isinstance(error, GoStageFailed) else getattr(error, "exit_code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    if code < 0:
        import signal
        return 130 if -code == int(signal.SIGINT) else 1
    return code or 1


def _member_ids(rows):
    """Every member the failing batches of a forecast name, in their order."""
    ids = []
    for row in rows:
        for member in row.get("member_ids") or ():
            if member not in ids:
                ids.append(member)
    return ids


def _forecast_record(forecast_dir):
    """The session's own run record, when the runner returned a code instead of raising.

    ``run/ensemble-run.json`` holds the real error class, its text and the
    failing members; an unreadable or absent record (a runner that refused
    before the session opened) is an empty one.
    """
    try:
        record = json.loads((Path(forecast_dir) / "ensemble-run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(record, dict) or record.get("status") not in ("failed", "interrupted"):
        return {}
    return record


def _notify(observer, event, **fields):
    from gpuwm.go_cli import _notify as notify
    notify(observer, event, **fields)


def run_recipe_ensemble(args, request, *, observer=None, options=None) -> int:
    """Plan, review, prepare and run a time-lagged or multi-model ensemble.

    ``options`` is what the calling door carries into every member's
    fetch: ``transport``, ``as_posted`` and ``late_after_minutes``.  The
    doors refuse, by name, the flags of theirs this door does not consume.
    """
    import sys
    import tempfile
    import time
    import tomllib

    from gpuwm import run_stamp, stage_cli
    from gpuwm.ensemble.door import production_run_scope
    from gpuwm.experiment import load_experiment
    from gpuwm.explain import explain_enabled
    from gpuwm.geog_assets import default_geog_root

    options = dict(options or {})
    options.setdefault("render_products", getattr(args, "render_products", None))
    options.setdefault("no_memory_gate", bool(getattr(args, "no_memory_gate", False)))
    command = getattr(args, "command", None) or "ensemble"
    config = Path(args.config).resolve()
    if not config.is_file():
        raise RecipeRefusal(f"{config} does not exist. Next: gpuwm domain --help")
    raw = tomllib.loads(config.read_text(encoding="utf-8-sig"))
    experiment = load_experiment(config)
    recipe = plan_recipe(request, raw, experiment)
    wps = getattr(args, "wps_namelist", None) or config.with_name(config.stem + ".namelist.wps")
    if not Path(wps).is_file():
        raise RecipeRefusal(
            f"{Path(wps).name} is not beside {config.name}; each member is prepared "
            "from the WPS namelist the domain door wrote there. Next: gpuwm domain --help")
    geog_root = getattr(args, "geog_root", None)
    geog_root = Path(geog_root).resolve() if geog_root is not None else default_geog_root()
    case_root = Path(args.outdir) if getattr(args, "outdir", None) is not None else (
        config.parent / f"{config.stem}-go")
    with tempfile.TemporaryDirectory(prefix="gpuwm-recipe-") as scratch:
        plans = review_members(raw, config, wps, recipe, scratch=scratch, options=options)
        _say(f"{recipe.kind} recipe, {len(recipe.members)} members, valid "
             f"{recipe.start:%Y-%m-%dT%H:%M}Z to {recipe.end:%Y-%m-%dT%H:%M}Z")
        for plan in plans:
            _say(plan.line)
        if getattr(args, "dry_run", False):
            _say("dry run: nothing was fetched, prepared or run")
            return 0
        admit(plans, config=config, geog_root=geog_root, case_root=case_root,
              request=request, options=options, command=command)
    root = run_stamp.resolve(case_root, init=str(raw["fetch"]["cycle"]),
                             enabled=run_stamp.run_stamp_enabled(args), create=True)
    root = Path(root)
    receipt = {"schema": RECEIPT_SCHEMA, "status": "preparing", "config": str(config),
               "recipe": recipe.describe(), "recipe_sha256": recipe.sha256,
               "request": request.receipt(), "members": []}

    def publish():
        temporary = root / (RECEIPT_NAME + ".pending")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                             encoding="utf-8")
        temporary.replace(root / RECEIPT_NAME)

    #: Where the run is, for the failure record: the stage and its member.
    where = {"stage": "prepare", "member_id": None, "started": time.monotonic()}

    def forecast_members(rows):
        """The members a forecast failure belongs to: the error's own, else the provider's."""
        ids = _member_ids(rows)
        if not ids and where["member_id"] is not None:
            ids = [where["member_id"]]
        return ids

    def subject(ids=()):
        if where["stage"] == "prepare":
            return f"member {where['member_id']}'s preparation"
        if ids:
            from gpuwm.ensemble.execution import member_label as members_label
            return f"the forecast of {members_label(ids)}"
        return "the forecast"

    def failed(error, code, *, stopped=False, error_class=None, message=None, rows=None):
        """Record the failure where a reader of the run folder looks, and say it once.

        A forecast failure names its members from what run control wrote
        into the error (or, for a returned code, into ``ensemble-run.json``):
        ``member_id`` for one member, ``member_ids`` and ``failed_members``
        for a pack or a wave.
        """
        receipt["status"] = "interrupted" if stopped or code == 130 else "failed"
        failure = {"stage": where["stage"], "member_id": where["member_id"],
                   "error_class": error_class if error is None else type(error).__name__,
                   "message": message if error is None else str(error),
                   "exit_code": code}
        if where["stage"] == "forecast":
            if rows is None:
                from gpuwm.ensemble.execution import failed_member_rows
                rows = [] if error is None else failed_member_rows(error)
            ids = forecast_members(rows)
            failure["member_id"] = ids[0] if len(ids) == 1 else None
            failure["member_ids"] = ids
            if rows:
                failure["failed_members"] = rows
        receipt["failure"] = failure
        try:
            publish()
        except OSError:
            pass
        _notify(observer, "stage_end", label=where["stage"], exit_code=code, ok=False,
                elapsed_seconds=time.monotonic() - where["started"], progress=None)

    publish()
    prepared = {}
    try:
        for plan in plans:
            member = plan.member
            where.update(stage="prepare", member_id=member.index, started=time.monotonic())
            directory = root / "members" / f"member-{member.index:03d}"
            _say(f"preparing member {member.index} ({member.trajectory.source} "
                 f"{member.trajectory.cycle:%Y-%m-%dT%H}Z)")
            _notify(observer, "stage_begin", label="prepare", command=[
                "gpuwm", command, str(config), "#", member_label(member)])
            member_config, member_wps, member_experiment = member_configuration(
                raw, config, wps, recipe, member, directory / "source")
            prepared[member.index] = prepare_member(
                member_config, member_experiment, source=member.trajectory.source,
                directory=directory, downloads=case_root, geog_root=geog_root,
                recipe_sha256=recipe.sha256, options=options)
            receipt["members"].append({
                "member_id": member.index, "seed": member.seed,
                "source": member.trajectory.source, "cycle": member.trajectory.cycle.isoformat(),
                "source_member": member.trajectory.member,
                "trajectory_sha256": member.trajectory.identity,
                "start_lead_hours": plan.start_lead,
                **prepared[member.index]})
            publish()
            _notify(observer, "stage_end", label="prepare", exit_code=0, ok=True,
                    elapsed_seconds=time.monotonic() - where["started"], progress=None)

        base = recipe.members[0].index
        where.update(stage="forecast", member_id=None, started=time.monotonic())
        bundle = stage_cli.resolve_bundle(Path(prepared[base]["prepared_root"]))
        if bundle["layout"] != "single":
            raise RecipeRefusal("the prepared member bundle is a domain tree; this door runs one domain per member")
        forecast_dir = root / "run"
        cache = {}

        def provider(*, shared_inputs, member_id, request):
            # Member ``base`` IS the bundle the runner preflighted; every other
            # member is the same preflight applied to its own prepared bundle.
            if member_id == base:
                return shared_inputs
            if member_id not in cache:
                # A member whose bundle the preflight refuses is that
                # member's forecast failure, and the receipt says which.
                where["member_id"] = member_id
                cache[member_id] = member_inputs(shared_inputs, prepared[member_id])
                where["member_id"] = None
            return cache[member_id]

        sim = stage_cli.sim_command(
            bundle, experiment_config=Path(prepared[base]["experiment_config"]),
            wps_namelist=Path(prepared[base]["wps_namelist"]), outdir=forecast_dir,
            physics_profile=None, progress_format="jsonl",
            render_products=options.get("render_products"),
            **({"devices": args.devices} if getattr(args, "devices", None) is not None else {}),
            memory_gate=not options.get("no_memory_gate"))
        receipt["status"] = "forecasting"
        receipt["forecast_directory"] = str(forecast_dir)
        publish()
        _say(f"running {len(recipe.members)} members in {forecast_dir}")
        _notify(observer, "stage_begin", label="forecast", command=list(sim))
        from gpuwm import prepared_single_domain_forecast as runner
        with production_run_scope(request, output_directory=forecast_dir) as session:
            if session.input_provider is not None or session.member_roster is not None:
                raise RecipeRefusal("this ensemble session already has a member source owner")
            session.input_provider = provider
            code = runner.main(sim[3:], observer=observer)
            if code:
                # The runner said why on its own line (a refusal, or a
                # source lead later than its budget) and returned its code.
                # The session's run record keeps the real error and its
                # members; the receipt takes them from there.
                code = int(code)
                record = _forecast_record(forecast_dir)
                rows = list(record.get("failed_members") or ())
                failed(None, code, stopped=record.get("status") == "interrupted",
                       error_class=record.get("error_type"),
                       message=record.get("error") or f"the forecast stage exited {code}",
                       rows=rows)
                print(f"ensemble: stopped during {subject(forecast_members(rows))} "
                      f"(exit {code}). No later member or stage ran; "
                      f"{root / RECEIPT_NAME} records it.", file=sys.stderr, flush=True)
                return code
            products_receipt = session.completed_products()
        _notify(observer, "stage_end", label="forecast", exit_code=0, ok=True,
                elapsed_seconds=time.monotonic() - where["started"], progress=None)
    except BaseException as error:  # noqa: BLE001 - every exit is recorded in the receipt
        code = _exit_code(error)
        from gpuwm.ensemble.execution import failed_member_rows
        what = subject(forecast_members(failed_member_rows(error))
                       if where["stage"] == "forecast" else ())
        said = (str(error) or type(error).__name__).rstrip(".")
        if isinstance(error, SystemExit):
            # A stop signal, as run control reads it: the session says
            # ``interrupted``, so the receipt does too, with the stop's own
            # code.  One sentence, and the stop keeps the process status.
            failed(error, code, stopped=True)
            print(f"ensemble: interrupted during {what} (exit {code}). No later member "
                  f"or stage ran; {root / RECEIPT_NAME} records it.",
                  file=sys.stderr, flush=True)
            raise
        if code is not None:
            # A stage's own exit: it printed its reason above.  One line
            # and its code, as the ordinary door ends.
            failed(error, code)
            verb = "interrupted" if code == 130 else "stopped"
            print(f"ensemble: {verb} during {what}: {said}. No later member "
                  f"or stage ran; {root / RECEIPT_NAME} records it.",
                  file=sys.stderr, flush=True)
            return code
        failed(error, 2 if isinstance(error, ValueError) else 1)
        if isinstance(error, ValueError):
            # A refusal reached from inside a chain: the command's own
            # boundary prints its sentence at exit 2.
            raise
        if not isinstance(error, Exception):
            raise                              # GeneratorExit
        import traceback
        log = root / FAILURE_LOG_NAME
        try:
            log.write_text(traceback.format_exc(), encoding="utf-8")
            kept = f" Its traceback is in {log}."
        except OSError:
            kept = ""
        print(f"ensemble: {what} failed: {type(error).__name__}: {said}. "
              f"{root / RECEIPT_NAME} records it.{kept}", file=sys.stderr, flush=True)
        if explain_enabled(args):
            traceback.print_exc()
        return 1
    receipt["status"] = "complete"
    receipt["products"] = products_receipt
    publish()
    _say(f"complete: {len(recipe.members)} members, products under {forecast_dir}; "
         f"member plan in {root / RECEIPT_NAME}")
    return 0
