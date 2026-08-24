"""``gpuwm go``: run the documented GFS chain without hand-relaying hashes.

Getting from a wizard-authored config to a forecast means five commands
in a fixed order, and three of them take values the previous one
printed: the front-door manifest's digest, then the preparation proof's
digest, its bound input-manifest digest, and its prepared-cache content
digest.  ``docs/public/FIRST-LIGHT.md`` section 3a spells the relay out
and says "copy it rather than retyping".

That relay is integrity plumbing.  Every one of those digests exists so
a later stage can refuse inputs an earlier stage did not produce, and
none of them is a decision a person makes -- so asking a person to
carry them is asking them to be a checksum courier, which is how a
first run ends in a transcription error three stages deep.

**What this does not do is weaken a single check.**  ``go`` runs the
same commands, as subprocesses, in the documented order, and every
artifact lands on disk exactly where the manual chain leaves it.  The
relayed values are read back from those artifacts -- ``sha256`` of the
manifest file, and the three digests carried inside ``proof.json`` --
never scraped from a stage's printed prose.  Prose is a presentation
layer that ``--explain`` is free to move; the artifacts are the
contract.  Each stage still verifies what it always verified, against
values ``go`` transported but did not compute.

**Ordering is not negotiable and there is no continue-on-failure.**
The authority materialization has to precede preprocessing, because the
runner binds the experiment config into the prepared cache and doing it
afterwards means preprocessing twice.  Every later stage consumes the
previous one's output, so a failed stage replays its whole output and
stops -- unlike ``gpuwm setup``, whose steps are independent.

**Scope is the route that is actually documented.**  GFS
single-domain.  A config bound to a shipped ``--physics-profile``
carries that assertion through every stage; a config bound to none --
the wizard's default suite, or any hand-authored suite the engine
implements -- runs as written, with its WRF-verification status stated
in the receipts (owner ruling 2026-07-31: the suite choice is the
user's, and status is reported, never gating).  What is still refused
up front -- an ERA5 ``[case_data]`` config, a multi-domain tree, a
source with no printed-value relay -- is refused naming the manual
chain, rather than half-orchestrated into a state nobody can finish by
hand either.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from gpuwm import capabilities
from gpuwm import run_stamp as run_stamp_module
from gpuwm.explain import explain_enabled, layered, render
# The two-stage route is spelled in ONE place (the seam that IS those
# two stages), and every door that points a reader at it renders from
# there -- this refusal included.
from gpuwm.stage_cli import staged_route_commands

#: The runner that owns both the authority materialization and the
#: forecast, as an importable module rather than a script path.
#:
#: It used to be ``tools/prepared_single_domain_forecast.py`` and it
#: cannot be any more: ``go``'s whole job is that a person who has
#: ``pip install gpuwm`` can get from a config to a forecast, and a
#: script path under ``tools/`` is a promise only a git checkout keeps.
#: The runner's substance now lives in the package (``tools/`` keeps a
#: thin delegating entry point for the spelling the docs use), so every
#: stage below is ``python -m <module>`` -- resolvable from a wheel, a
#: checkout, or a venv with neither on the working directory.
RUNNER_MODULE = "gpuwm.prepared_single_domain_forecast"

#: The runner a DOMAIN TREE goes to.  Preparation is identical -- the
#: refusal below says so in as many words, "prepared the same way" --
#: and only the forecast stage differs: the tree runner binds ONE
#: preparation receipt where the single-domain runner binds three
#: proof digests.
TREE_RUNNER_MODULE = "gpuwm.prepared_domain_tree_forecast"

#: The checkout spelling of the same runner, kept for the docs and for
#: receipts that name it.  Not used to locate anything.
RUNNER_RELATIVE = "tools/prepared_single_domain_forecast.py"

#: Seconds between "still working" lines while a stage is quiet.
#:
#: rw-wps' static build and the runner's preparation each run for
#: minutes with nothing on the terminal, and a silent terminal is read
#: as a hang -- an owner watching one concluded the tool was broken and
#: stopped it.  Whatever else a long stage does, it must keep saying
#: that it is a long stage.
HEARTBEAT_SECONDS = 20.0

#: The sources whose printed-value relay is documented end to end.
#:
#: GFS only, and for a reason that is a property of the other routes
#: rather than a preference.  HRRR reaches the single-domain runner --
#: it is in that module's ``SUPPORTED_SOURCES`` and its preparation
#: publishes a ``proof.json`` on every run -- but it does not reach it
#: through THIS chain: the native HRRR preparation takes a different
#: stage vocabulary (a strict target domain, a native static cache and
#: receipt, a sealed decoder bridge) that ``go``'s five commands do not
#: compose, and ``gpuwm fetch --source hrrr`` says as much itself,
#: ending with "fetching cannot bind the run's own flags".  Its two
#: stages already stand alone with no digests to carry by hand:
#: ``gpuwm prep --source hrrr`` then ``gpuwm sim``.
#:
#: ERA5 reaches that runner but arrives by the ``[case_data]``
#: config-driven route, which ``gpuwm run`` executes in one command and
#: which needs no relay.  Both are refused above with their own pointer.
ORCHESTRATED_SOURCES = ("gfs",)

#: Where the manual chain points a reader whose config this refuses.
#:
#: A repo-relative path was the whole pointer until 1.4.1, and a pip
#: install contains no docs tree -- so the one instruction offered to
#: the reader named a file they provably did not have, with no URL to
#: reach it by (A-10, and the pointer half of B-03).  The URL is built
#: from the project's own declared Repository metadata so the two cannot
#: drift apart, and the local path stays for readers who do have a
#: checkout.
DOCS_BASE_URL = (
    "https://github.com/FahrenheitResearch/arwen/blob/main/")
MANUAL_CHAIN = (
    f"docs/public/FIRST-LIGHT.md section 3a "
    f"({DOCS_BASE_URL}docs/public/FIRST-LIGHT.md)")


class GoRefusal(ValueError):
    """A config this command will not half-orchestrate."""


def _repo_root() -> Path:
    """The directory containing the ``gpuwm`` package.

    A checkout root in a clone; ``site-packages`` in an install.  Only
    used to notice which of the two this is -- nothing is located
    through it any more.
    """

    return Path(__file__).resolve().parent.parent


def _is_checkout() -> bool:
    """True when this install is a repository clone."""

    return (_repo_root() / RUNNER_RELATIVE).is_file()


def _stage_cwd() -> Path:
    """Where the stage subprocesses run.

    The caller's directory, so a relative ``--out`` or ``--data-dir``
    means what the person who typed it meant.  Every path ``go`` itself
    composes is absolute, so this choice cannot move an artifact; it can
    only stop a relative one from landing somewhere surprising.  (It
    used to be the checkout root, which no wheel install has.)
    """

    return Path.cwd()


def _stage_env() -> dict:
    """The environment every stage subprocess gets.

    ``PYTHONSAFEPATH`` is the whole of it, and it is not a nicety.
    Every stage is spawned as ``python -m MODULE``, and ``-m`` prepends
    the CURRENT DIRECTORY to ``sys.path`` -- ahead of the installed
    package.  :func:`_stage_cwd` is deliberately the caller's directory
    so a relative ``--out`` means what the person typing it meant, so a
    chain started from inside a source checkout imported that checkout
    instead of the install.  A live run was hijacked exactly that way.

    Moving the cwd would fix the imports and break the relative paths.
    ``PYTHONSAFEPATH`` separates the two: the child still RESOLVES paths
    against the caller's directory and no longer IMPORTS from it.

    Set here rather than as a ``-P`` on each command line because the
    commands are composed in six places and an env var cannot be
    forgotten by the seventh.
    """

    return _with_git_handle({**os.environ, "PYTHONSAFEPATH": "1"})


def _with_git_handle(environment: dict) -> dict:
    """Add ``GPUWM_GIT_EXE`` so a stage can bind its own identity.

    Every stage resolves ``gpuwm.runtime_manifest.provenance`` before it
    reads a byte of the user's data, and on a source checkout that
    resolution is a ``git`` subprocess.  A child that cannot start git
    binds no identity and refuses the run -- which is what happened,
    with the parent's own identity resolving perfectly one process up.
    Handing over the path the parent ALREADY resolved removes the
    child's search entirely, and does it without widening ``PATH``:
    this adds one key naming one executable, not a directory of them.
    """

    from gpuwm.provenance import GIT_EXE_ENV, git_executable

    resolved = git_executable()
    if resolved is not None:
        environment[GIT_EXE_ENV] = resolved
    return environment


def _quote(value) -> str:
    return shlex.quote(str(value).replace("\\", "/"))


def printable(command: list[str]) -> str:
    """One command as a pasteable line."""

    return " ".join(_quote(token) for token in command)


# ---------------------------------------------------------------------------
# Reading the config: what the chain needs, and what it refuses
# ---------------------------------------------------------------------------

def plan_from_config(config: Path, *, outdir: Path | None = None,
                     data_dir: Path | None = None,
                     render_products: str | None = None,
                     allow_tree: bool = False,
                     run_stamp: bool = run_stamp_module.DEFAULT_RUN_STAMP,
                     claim: bool = False) -> dict:
    """Everything the five stages need, or a refusal saying why not.

    ``run_stamp`` puts this run's whole tree -- authority, prepared,
    run, png -- in its own timestamped folder under the output
    directory, which is the default and the reason two runs of one
    config no longer collide.  The DOWNLOAD stays at
    ``<outdir>/data``, shared across runs: it holds inputs, it is meant
    to be reused, and stamping it would re-fetch identical GRIB every
    time.

    A plan NAMES that folder; it does not create it.  The gates in
    :func:`go_main` refuse before anything is spent and a refused run
    must leave the disk as it found it, so the directory is claimed by
    :func:`claim_run_root` at the moment the first stage is about to
    write -- create-exclusively, so two chains started inside one second
    take two folders rather than sharing one and publishing receipts
    describing neither.  ``claim=True`` folds that step in for a caller
    that wants the plan and the folder in one call.

    Read from the wizard's own emitted tables rather than asked for
    again: ``[fetch]`` carries the resolved cycle, hours and area the
    wizard sized the domain against, and the physics tables carry the
    profile.  A flag that re-asked for any of them would let the two
    disagree, and the run would then be sized for one box and fetched
    for another.
    """

    import tomllib

    from gpuwm.experiment import load_experiment
    from gpuwm.physics_compat import identify_single_domain_profile
    from gpuwm.static.corridor import config_declares_follow_source

    config = Path(config)
    if not config.is_file():
        raise GoRefusal(f"{config} does not exist")
    payload = tomllib.loads(config.read_text(encoding="utf-8"))

    if "case_data" in payload:
        raise GoRefusal(
            f"{config} declares a [case_data] table, which is the ERA5 "
            "config-driven route -- `gpuwm run` executes that one "
            "directly and needs no chain.\n"
            "  remedy: gpuwm check " + str(config) + " && gpuwm run "
            + str(config))

    domains = payload.get("domain")
    domain_count = len(domains) if isinstance(domains, list) else 1
    if domain_count > 1 and not allow_tree:
        # Named by its INSTALLED spelling: a wheel has no tools/
        # directory, so a tools/-path pointer names a file a pip reader
        # provably does not have.  And the re-emit remedy is complete
        # rather than trailing off in "...": the wizard's default IS the
        # single-domain shape now, so "without --ladder" is the whole
        # instruction -- a tree config is one this reader explicitly
        # opted into.
        raise GoRefusal(
            f"{config} declares {len(domains)} domains, and the "
            "single-domain runner this chain drives takes one.  A domain "
            "tree is prepared the same way but run by "
            "gpuwm-prepared-tree-forecast (module form: python -m "
            "gpuwm.prepared_domain_tree_forecast), which binds a "
            "preparation receipt rather than three digests; the full "
            f"tree sequence is {MANUAL_CHAIN}.\n"
            "  remedy: re-emit the config without --ladder (the wizard's "
            "default is one 12 km domain, the shape this chain runs end "
            "to end), or prepare the tree with rw-wps and run "
            "gpuwm-prepared-tree-forecast on what it wrote")

    fetch_table = payload.get("fetch")
    if not isinstance(fetch_table, dict) or "source" not in fetch_table:
        raise GoRefusal(
            f"{config} carries no [fetch] table, so the cycle and area "
            "this domain was sized against are not recorded in it.\n"
            "  remedy: re-emit the config with `gpuwm domain`, which "
            "writes that table")

    source = str(fetch_table["source"])
    if source not in ORCHESTRATED_SOURCES:
        raise GoRefusal(
            f"{config} is a --source {source} config, and this chain is "
            f"the {'/'.join(ORCHESTRATED_SOURCES)} one.  "
            + ("hrrr's preparation takes a different stage vocabulary "
               "-- a strict target domain, a native static cache and "
               "receipt, a sealed decoder bridge -- which these five "
               "commands do not compose.  It needs no relay either: "
               "its preparation publishes the authorities its forecast "
               "binds, so the whole route is two commands.\n"
               "  remedy: " + " && ".join(staged_route_commands("hrrr"))
               if source == "hrrr" else
               f"{source} has no printed-value relay to automate.\n"
               f"  remedy: follow {MANUAL_CHAIN} for this source"))

    try:
        experiment = load_experiment(config)
        profile = identify_single_domain_profile(experiment.root.run)
    except Exception as error:  # a config that will not load is the finding
        raise GoRefusal(f"{config} does not load as an experiment: "
                        f"{error}") from error
    # ``profile`` may be None: the runner executes the config's own
    # suite as written (owner ruling 2026-07-31), so a config matching
    # no shipped profile is not a refusal any more -- the chain just
    # omits --physics-profile and the receipts report the suite's
    # verification status.  A matched profile is still forwarded, so a
    # bound config keeps its exactness assertion end to end.
    #
    # Matched at the ROOT is not matched by the CONFIG, and the gap is
    # not theoretical: the wizard's own --ladder trees write their nests
    # with deliberate departures from the root suite (cumulus OFF below
    # the gray zone, a tighter radt, the certified diff_6th_factor
    # ladder), and stage 1 REFUSES a named profile that any declared
    # value on any domain contradicts -- it used to silently flatten
    # those nests onto the profile instead, which is the ledger #90
    # defect.  Forwarding the root-derived name would therefore compose
    # a stage-1 command guaranteed to refuse this chain's own config.
    # The derivation asks the materializer's own conflict predicate, so
    # this door and that refusal read the same sentence: agreement
    # forwards the assertion, and a config that deliberately says more
    # than the profile runs as its own suite, unnamed.
    if profile is not None:
        from gpuwm.prepared_single_domain_forecast import (
            named_profile_config_conflicts)
        if named_profile_config_conflicts(
                config.read_text(encoding="utf-8"),
                source=source, profile=profile):
            profile = None

    # No "is the runner on disk?" gate any more.  It used to be here
    # because the runner was a script under tools/ that a wheel install
    # does not carry, and the honest answer was "go clone the
    # repository".  The runner is part of the package now, so if this
    # module imported, so did it.

    # [tiles] is HONORED, and recorded rather than refused.  The refusal
    # that used to stand here brought the runner's admission refusal
    # forward, to the near side of the download -- and the runner no longer
    # refuses, because both prepared routes wire a streamed-domain builder
    # (streaming.builders_for_tree): the authority stage carries the
    # config's [tiles] table into the hash-bound experiment.toml byte for
    # byte, and the forecast stage reads it there and streams.  What
    # remained wrong after the refusal went was the SILENCE: the plan
    # recorded nothing, so a config that asked to stream planned six
    # commands that never said so, and the only evidence was one stage
    # line on a captured stdout.  The decision is recorded on the plan
    # here, where the banner and the dry run say it out loud.  The one
    # [tiles] shape this chain genuinely cannot run -- a coupling edge
    # with BOTH ends streamed -- was already refused by load_experiment
    # above, with the core's own sentence, relayed before the fetch.
    tiles_plan = None
    tiles_options = getattr(experiment, "tiles", None)
    if tiles_options is not None and tiles_options.enabled:
        from gpuwm.core import streaming

        asked = [
            f"d{int(dc.grid_id):02d}" for dc in experiment.domains
            if streaming.options_for_domain(dc, tiles_options).enabled]
        if tiles_options.mode == "on":
            outcome = (
                f"{'/'.join(asked)} integrate(s) out of a pinned "
                f"{tiles_options.store} store, tile by tile")
        else:
            outcome = (
                f"{'/'.join(asked)} stream(s) out of a pinned "
                f"{tiles_options.store} store wherever the planner says "
                "the domain does not fit resident, decided against the "
                "card at the forecast stage")
        tiles_plan = {
            "mode": tiles_options.mode,
            "store": tiles_options.store,
            "asked": asked,
            "sentence": (
                f"[tiles] mode = '{tiles_options.mode}' "
                f"({tiles_options.store} store) rides the hash-bound "
                f"experiment config into the forecast stage: {outcome}"),
        }
    for key in ("cycle", "hours", "area"):
        if key not in fetch_table:
            raise GoRefusal(
                f"{config}'s [fetch] table has no {key!r}; re-emit the "
                "config with `gpuwm domain`")

    base = config.parent
    # The CASE root: the directory the reader named (or the one derived
    # from the config's own name).  It holds the download cache and one
    # folder per run; the run tree itself is the stamped child claimed
    # below.
    case_root = (Path(outdir) if outdir is not None
                 else base / f"{config.stem}-go")
    root = case_root
    # `[fetch].out` is deliberately NOT used, and this is not an
    # oversight in either direction.
    #
    # The wizard writes that key relative to the directory IT was run in
    # (`_relative_or_absolute(data_dir, Path.cwd())`), not relative to
    # the config file, so a config emitted from one directory records a
    # download path that means something else -- or nothing -- read from
    # another.  A config emitted from a checkout root and read from
    # /tmp resolved to `C:/AppData/...`, six `..` hops having walked
    # past the drive root and clamped there.
    #
    # The table declares itself "advisory ... validated, not executed",
    # and the values that are load-bearing are the ones the domain was
    # SIZED against -- cycle, hours, area -- which are absolute facts
    # and are honoured exactly.  Where the bytes land is this command's
    # own business, so they land under its own root unless `--data-dir`
    # names an existing download to reuse.
    # NOT stamped, on purpose.  This directory holds INPUTS -- the GRIB
    # the fetch stage downloads -- and inputs are cached across runs
    # while artifacts are separated by run.  Putting it inside the run
    # folder would re-download an identical forcing set on every re-run
    # of one config, which is the opposite of the complaint the stamping
    # answers.
    data = Path(data_dir) if data_dir is not None else case_root / "data"
    # E-07.  The default puts the download INSIDE the run root, so the
    # two are related by construction and only an explicit --data-dir can
    # make them equal.  When it does, the run claims the directory the
    # inputs live in: every stage is create-only, so a re-run then
    # refuses against a directory the reader thinks of as their download
    # cache, and the download and the run share a receipt.  Both flags
    # were the reader's, so neither is dropped -- this refuses and says
    # which one to move.
    if outdir is not None and data_dir is not None:
        try:
            same = Path(outdir).resolve() == Path(data_dir).resolve()
        except (OSError, RuntimeError):
            same = Path(outdir) == Path(data_dir)
        if same:
            raise GoRefusal(
                f"--outdir and --data-dir both name {root}.  The run "
                f"directory is create-only and the download directory is "
                f"meant to be reused, so one directory cannot be both: "
                f"the next run would refuse against your own cache.  "
                f"Point --outdir at a new directory and leave --data-dir "
                f"on the download, or drop --data-dir and let the "
                f"download land under the run root as {root / 'data'}.")
    # AFTER every refusal above: a plan that is going to be refused must
    # not leave a directory behind on the way out.
    root = run_stamp_module.resolve(
        case_root, init=fetch_table["cycle"], enabled=run_stamp,
        create=claim)
    return {
        "config": config,
        "wps_namelist": base / f"{config.stem}.namelist.wps",
        "source": source,
        "cycle": str(fetch_table["cycle"]),
        "hours": int(fetch_table["hours"]),
        # Load-bearing exactly like cycle/hours/area: the config's
        # start_time IS cycle + this lead, so a fetch that ignored it
        # would download a window the front door then refuses for not
        # carrying the lead the experiment starts from.
        "forecast_start_hour": int(fetch_table.get("forecast_start_hour", 0)),
        # Load-bearing for the same reason, and dropped until 1.4.1.  It
        # sets the LATERAL BOUNDARY interval: a config asking for hourly
        # forcing whose fetch runs without --cadence downloads f000/f003
        # and the run gets 3-hourly boundaries -- three times coarser
        # than the file says, with exit 0, "forecast validity PASS", and
        # nothing anywhere saying the request was not honoured.  A
        # measured case: [fetch] cadence = 1, report.json
        # boundary_interval_seconds = 10800.  The block is advisory in
        # the sense that this command chooses where bytes land; it is
        # not advisory about what is fetched.
        "cadence": (int(fetch_table["cadence"])
                    if fetch_table.get("cadence") is not None else None),
        "area": str(fetch_table["area"]),
        "data": data,
        "profile": profile,
        # Two roots, and the distinction is the whole scheme.  ``root``
        # is THIS RUN's tree, which every stage writes into and every
        # existing consumer already reads; ``case_root`` is the folder
        # the reader named, which holds the shared download and one
        # ``run-...`` child per run.
        "case_root": Path(case_root),
        "root": Path(root),
        "authority": Path(root) / "authority",
        "prepared": Path(root) / "prepared",
        "run": Path(root) / "run",
        # Set by a caller that wants a subset; `gpuwm go` itself never
        # sets it, so its render stage is unchanged.
        "render_products": render_products,
        # None for the resident run every config without a [tiles] table
        # is -- the same emptiness contract the receipts keep -- and the
        # recorded routing decision otherwise.  Nothing is forwarded on a
        # flag: the forecast stage reads the table off the hash-bound
        # experiment config the authority stage publishes.
        "tiles": tiles_plan,
        "render": Path(root) / "png",
        "domains": domain_count,
        # Preparation does not branch; the forecast does.
        "runner": (TREE_RUNNER_MODULE if domain_count > 1
                   else RUNNER_MODULE),
        # A config that declares a [relocation] follow source needs the
        # sealed statics corridor prepared, or the forecast stage will
        # refuse the very bundle stage 4 just built.  Derived from the
        # config here so the chain stays config-driven end to end, and
        # through the corridor module's own predicate so this door, the
        # printed rw-wps line and run-plan's refusal all read the same
        # sentence out of one place.
        "statics_corridor": config_declares_follow_source(experiment),
    }


# ---------------------------------------------------------------------------
# The five commands, composed from the plan
# ---------------------------------------------------------------------------

def claim_run_root(plan: dict) -> dict:
    """Create this run's folder and re-point the plan's trees into it.

    Separate from :func:`plan_from_config`, and called at the moment the
    first stage is about to write, for two reasons that pull the same
    way.  The gates refuse without spending anything and a refused run
    must leave the disk exactly as it found it, so naming the folder and
    making it cannot be one step.  And the making has to be
    create-exclusive at the moment it happens: two chains launched
    inside one second both PREDICT the same stamp, and only an
    allocation that fails on an existing directory separates them.

    ``plan`` is updated in place, because the observer, the render stage
    and the closing report all hold this same dict.
    """

    root = Path(plan["root"])
    if root == Path(plan["case_root"]):
        # Either --run-stamp off, or --outdir named a run folder and it
        # was honoured verbatim.  Both leave the run root EQUAL to the
        # case root, which is the one test that separates "the plan
        # predicted a stamped child" from "the caller chose the folder"
        # -- the predicted child's NAME is a run stamp too, so asking
        # the name instead would never allocate anything.
        root.mkdir(parents=True, exist_ok=True)
        return plan
    claimed = run_stamp_module.allocate(plan["case_root"],
                                        init=plan.get("cycle"))
    plan["root"] = claimed
    plan["authority"] = claimed / "authority"
    plan["prepared"] = claimed / "prepared"
    plan["run"] = claimed / "run"
    plan["render"] = claimed / "png"
    return plan


def _run_folder_note(plan: dict) -> str:
    """The one line saying where THIS run's tree is, before it exists.

    Printed by the dry run and by the real run, from the same function,
    so the path a reader plans against is the path they get.  Says which
    of the three cases they are in rather than leaving them to compare
    two directory names: a fresh stamped folder, a folder they named, or
    the unstamped tree the workaround restores.
    """

    root, case_root = Path(plan["root"]), Path(plan["case_root"])
    # The EFFECTIVE download directory, which is `<case root>/data` only
    # when the reader named none.  Deriving it here instead re-computed
    # the default and announced a path `--data-dir` had already moved:
    # every stage that touches the download -- fetch, manifest, prepare
    # -- composes `plan["data"]`, so this is the one value that can be
    # said out loud without the line drifting from the run it describes.
    data = Path(plan["data"])
    if root == case_root:
        return (f"go: run folder {root} (--run-stamp off: this run "
                "writes straight into the output directory, so a second "
                "run of this config will be refused against it)")
    if not run_stamp_module.is_run_folder(root):    # pragma: no cover
        return f"go: run folder {root}"
    note = (f"go: run folder {root.name} under {case_root} -- "
            f"authority/, prepared/, run/ and png/ all land inside it, "
            f"and the download is cached at {data}")
    return note


def _lead_note(plan: dict) -> str:
    """The forecast lead, said in the plan line, or nothing at lead 0."""

    lead = plan.get("forecast_start_hour") or 0
    if not lead:
        return ""
    return (f", initialized from forecast lead f{lead:03d} "
            f"(a {lead} h forecast, not an analysis)")


def _area_flag(area: str) -> str:
    """``--area=VALUE`` when the value leads with a minus, else split.

    The same rule the wizard's printed command follows: argparse reads a
    leading-minus token as an option unless it is joined with ``=``.
    """

    return f"--area={area}"


def _profile_flags(plan: dict) -> list[str]:
    """``--physics-profile <id>`` when the config is bound to one.

    Empty for a config matching no shipped profile: the runner executes
    the hash-bound experiment's own suite as written, and inventing a
    name here would re-create the WSM6-substitution defect this chain
    already met once.
    """

    profile = plan.get("profile")
    return [] if profile is None else ["--physics-profile", str(profile)]


def authority_command(plan: dict) -> list[str]:
    # RUNNER_MODULE, not plan["runner"].  Materializing the physics
    # authority is SOURCE-level work -- it writes the experiment.toml
    # and namelist.wps every later stage binds -- and it lives in the
    # single-domain module for every route, which is what FIRST-LIGHT
    # step 2 spells even for a chain that ends in the tree runner.  The
    # tree runner has no --materialize-authorities at all, so keying
    # this off the plan's runner sent stage one to a module that would
    # refuse the flag.
    return [sys.executable, "-m", RUNNER_MODULE,
            "--materialize-authorities",
            "--source", plan["source"],
            "--base-experiment-config", str(plan["config"]),
            "--base-wps-namelist", str(plan["wps_namelist"]),
            *_profile_flags(plan),
            "--output-directory", str(plan["authority"])]


def fetch_command(plan: dict) -> list[str]:
    command = [sys.executable, "-m", "gpuwm.cli", "fetch",
               "--source", plan["source"], "--cycle", plan["cycle"],
               "--hours", str(plan["hours"]), _area_flag(plan["area"]),
               "--out", str(plan["data"])]
    if plan.get("cadence") is not None:
        command.extend(("--cadence", str(plan["cadence"])))
    if plan.get("forecast_start_hour"):
        command.extend(
            ("--forecast-start-hour", str(plan["forecast_start_hour"])))
    return command


def manifest_command(plan: dict, bridge: Path) -> list[str]:
    return [sys.executable, "-m", "gpuwm.cli", "fetch",
            "--source", plan["source"], "--author-front-door-manifest",
            "--out", str(plan["data"]), "--bridge", str(bridge),
            "--wps-namelist", str(plan["authority"] / "namelist.wps"),
            "--experiment-config", str(plan["authority"] / "experiment.toml")]


def prepare_command(plan: dict, bridge: Path, *, manifest: Path,
                    manifest_sha256: str, cycle_stamp: str,
                    geog_root: Path) -> list[str]:
    """The rw-wps invocation, composed from the artifacts on disk.

    Every value here is either from the plan or read back from a file a
    previous stage wrote -- the manifest's path and its digest.  This is
    the same command ``gpuwm fetch --author-front-door-manifest``
    prints; composing it from the same inputs rather than parsing that
    printed line is what keeps it working when the printed line moves
    behind ``--explain``.
    """

    return [sys.executable, "-m", "gpuwm.source_cli",
            "--source", plan["source"],
            "--gfs-series", str(plan["data"] / f"{plan['source']}-series.tsv"),
            "--cycle", cycle_stamp,
            "--bridge", str(bridge),
            "--wps-namelist", str(plan["authority"] / "namelist.wps"),
            "--experiment-config",
            str(plan["authority"] / "experiment.toml"),
            "--source-manifest", str(manifest),
            "--source-manifest-sha256", manifest_sha256,
            # Forwarded only when the config is bound to one: the GFS
            # front door no longer substitutes WSM6 for an absent
            # profile (the defect that once made this flag mandatory in
            # practice) -- an unnamed config's own suite is prepared as
            # written and its verification status reported.
            *_profile_flags(plan),
            "--geog-root", str(geog_root),
            # The config declared a follow source, so the bundle must
            # carry the sealed statics corridor or stage 5 refuses it.
            *(["--statics-corridor"] if plan.get("statics_corridor")
              else []),
            "--output-root", str(plan["prepared"])]


#: How `go` asks the forecast stage for its per-step progress.
#:
#: The runner's OWN default is `text`: one WRF-shaped `Timing for main:`
#: line per model time step on stdout, which is what a user driving that
#: door from a script asked for and gets.  `go` is the other caller, and
#: for it the sentences must not go to stdout, for two reasons:
#:
#: * the SUBPROCESS arm captures the stage's stdout in memory and
#:   discards it on success, so a 4-domain day-long run would buffer tens
#:   of megabytes of lines nobody will ever read;
#: * the IN-PROCESS arm is `gpuwm run-plan` hosting the runner, and that
#:   front door's contract is that stdout carries its event stream and
#:   NOTHING else.
#:
#: `jsonl` keeps every one of those lines -- they land in
#: `<run>/progress.jsonl`, beside the frame-ready markers -- and keeps
#: them off the channel neither caller can afford to have written on.
_PROGRESS_FLAGS = ("--progress-format", "jsonl")


def _early_render_products(plan: dict) -> str | None:
    """What the forecast stage should draw as the first frame lands.

    TIME TO FIRST PLOT, on the door people use.  The machinery exists
    (:mod:`gpuwm.first_products`) and `gpuwm run-plan` proves it, but it
    is armed on an OBSERVER -- and arming it from `go` would mean
    hosting the forecast in this process, which is the one thing `go`
    must not start doing for telemetry's sake.  The runner owns an
    identical arming of its own, reachable from its command line, so
    this chain asks for it there and keeps its subprocess.

    ``"all"`` when the plan named no products, because ``"all"`` IS what
    the finalize stage renders when nobody passes ``--products``
    (``gpuwm render --products`` defaults to it).  The two must agree
    exactly: finalize skips the early frame only when the receipt's
    product spec matches its own, and byte-identity between the early
    picture and the late one is a property of composing both commands
    from this one plan dict.

    ``None`` for a run that asked for no pictures -- ``none`` is passed
    through verbatim so the runner's own ``early_render_requested``
    makes that call, and there is no second place that can disagree
    about what "asked for pictures" means.
    """

    if "render" not in plan:
        # A caller driving this function with a hand-built plan (the
        # tree route's stub, a test) named no render directory, and the
        # early render must land in the same tree the finalize stage
        # renders into or the two layouts diverge.  No directory, no
        # early render, and nothing guessed.
        return None
    if render_extra_missing() is not None:
        # This install cannot draw at all, and the finalize stage below
        # will say so with a remedy.  Asking the runner to draw anyway
        # would spend a subprocess to reach the same answer, in a place
        # the reader is not looking.
        return None
    products = plan.get("render_products")
    return "all" if products is None else str(products)


def forecast_command(plan: dict, digests: dict, *,
                     early_render: str | None = None) -> list[str]:
    return [sys.executable, "-m", str(plan["runner"]),
            "--source", plan["source"],
            "--prepared-root", str(plan["prepared"]),
            "--proof-sha256", digests["proof"],
            "--source-manifest-sha256", digests["source_manifest"],
            "--prepared-content-sha256", digests["prepared_content"],
            "--experiment-config",
            str(plan["authority"] / "experiment.toml"),
            "--wps-namelist", str(plan["authority"] / "namelist.wps"),
            *_profile_flags(plan),
            *_PROGRESS_FLAGS,
            # Same output directory the finalize stage renders into, so
            # the early picture and the late one land in one tree under
            # one layout rather than two.
            *([] if early_render is None else
              ["--render-products", str(early_render),
               "--render-dir", str(plan["render"])]),
            "--io-mode", "history", "--outdir", str(plan["run"])]


def tree_forecast_command(plan: dict) -> list[str]:
    """The fifth stage for a DOMAIN TREE.

    The tree runner binds ONE digest where the single-domain runner
    binds three: the sha256 of the hierarchy document rw-wps left in the
    prepared root (``proof.json`` for gfs/era5, ``receipt.json`` for
    hrrr -- the runner accepts whichever filename carries a schema it
    knows, and pins the digest exactly).  Plus the experiment config's
    own digest, which the single-domain runner takes on trust from the
    proof.

    Both are read off the files on disk here, which is the same relay
    rule :func:`proof_digests` follows: `go` transports a digest, it
    never computes one the runner will then check against itself.  The
    manual chain has a person copy these two out of what rw-wps printed;
    nothing is printed-and-parsed here, the artifacts are the contract.
    """

    from gpuwm.fetch import sha256_file

    prepared = Path(plan["prepared"])
    receipt = _hierarchy_document(prepared)
    config = Path(plan["authority"]) / "experiment.toml"
    if not config.is_file():
        raise GoRefusal(
            f"the authority stage wrote no {config}, so the tree "
            "forecast has no experiment config to bind")
    return [sys.executable, "-m", str(plan["runner"]),
            "--prepared-root", str(prepared),
            "--preparation-receipt-sha256", sha256_file(receipt),
            "--experiment-config", str(config),
            "--experiment-config-sha256", sha256_file(config),
            *_PROGRESS_FLAGS,
            "--io-mode", "history", "--outdir", str(plan["run"])]


def _hierarchy_document(prepared_root: Path) -> Path:
    """The tree preparation's top-level document, whichever wrote it.

    The candidate filenames are the tree runner's own
    ``_HIERARCHY_DOCUMENTS`` table, read from it rather than restated:
    which files count as a hierarchy document is that runner's
    question, and a second list here is the enumeration drift this tree
    keeps paying for.
    """

    from gpuwm.prepared_domain_tree_forecast import _HIERARCHY_DOCUMENTS

    names: list[str] = []
    for entry in _HIERARCHY_DOCUMENTS:
        if entry["filename"] not in names:
            names.append(entry["filename"])
    # Matched on SCHEMA, not on filename order.  Four sources write
    # `proof.json` and one writes `receipt.json`; picking by position
    # would hand the runner a digest of the wrong file, and the runner
    # would then refuse with "carries no hierarchy document matching" --
    # true, unhelpful, and this function's fault.
    accepted = {entry["schema"] for entry in _HIERARCHY_DOCUMENTS}
    present: list[str] = []
    for name in names:
        candidate = Path(prepared_root) / name
        if not candidate.is_file():
            continue
        present.append(name)
        try:
            schema = json.loads(
                candidate.read_text(encoding="utf-8")).get("schema")
        except (OSError, ValueError, AttributeError):
            continue
        if schema in accepted:
            return candidate
    if present:
        raise GoRefusal(
            f"{prepared_root} carries {present}, but none of them is a "
            "hierarchy document this build knows; the preparation may "
            "have written a single-domain product, whose runner is "
            f"{RUNNER_MODULE}")
    raise GoRefusal(
        f"the preparation wrote none of {names} into {prepared_root}, so "
        "the tree forecast has no preparation receipt to bind")


def _run_forecast(plan: dict, digests: dict, *, explain: bool,
                  observer=None) -> None:
    """The forecast stage: subprocess by default, in-process for an observer.

    Same command either way.  ``forecast_command`` composes it once and
    both arms consume that one list, so the flags a run-plan run
    executes are byte-identical to the ones ``gpuwm go`` prints and
    spawns -- there is no second argument set to keep in step.

    Why the second arm exists at all: a subprocess can only be observed
    through what it publishes, and ``progress.json`` is republished on
    the runner's own throttle (every sixtieth step).  That is a fine
    heartbeat and a poor event stream, and it can never say "this
    wrfout just landed" at the moment it did.  In-process, the runner's
    per-domain writer raises ``output_committed`` on the thread that
    published the file, and every step reaches the observer.

    The subprocess arm stays the default and stays untouched, because
    process isolation is what keeps a CUDA failure inside one stage.
    A caller that asks for the observer is asking to host the forecast,
    and run-plan does exactly that -- it IS the supervising process.
    """

    # A host is an observer that says it hosts.  Every observer used to
    # mean "run the forecast in this process", which was fine while the
    # only observer was run-plan -- but `gpuwm go` now always carries a
    # stage-event observer, and telemetry must not be the reason a bare
    # chain gives up the process isolation that keeps a CUDA failure
    # inside one stage.  Defaults to True, so every existing caller
    # (which is run-plan, and which does host) is unchanged.
    hosted = observer is not None and getattr(observer, "hosts_forecast", True)
    command = (tree_forecast_command(plan) if plan.get("domains", 1) > 1
               else forecast_command(plan, digests,
                                     early_render=None if hosted
                                     else _early_render_products(plan)))
    progress = plan["run"] / "progress.json"
    if not hosted:
        _run_stage("forecast", command, explain=explain, progress=progress,
                   observer=observer)
        return

    # `python -m MODULE ...` -> the module, and the argv it would have
    # been handed.  Sliced off the composed command rather than rebuilt.
    module_name, argv = command[2], command[3:]
    _notify(observer, "stage_begin", label="forecast", command=list(command))
    started = time.monotonic()
    print("  .. forecast (in process)", flush=True)
    import importlib

    runner = importlib.import_module(module_name)
    code = runner.main(argv, observer=observer)
    ok = not code
    _notify(observer, "stage_end", label="forecast", exit_code=code, ok=ok,
            elapsed_seconds=time.monotonic() - started,
            progress=_progress_payload(progress))
    if not ok:
        print(f"  FAILED  forecast (exit {code})")
        print("go: stopped at forecast; every later stage consumes this "
              "one's output, so nothing after it ran.")
        raise GoStageFailed(code)
    print(f"  ok      forecast "
          f"({_elapsed_words(time.monotonic() - started)})")


# ---------------------------------------------------------------------------
# Reading relayed values back out of the artifacts
# ---------------------------------------------------------------------------

def proof_digests(prepared_root: Path) -> dict:
    """The three digests the forecast stage binds, from ``proof.json``.

    Exactly the values ``gpuwm.gfs_direct`` composes its printed
    next-command from, read from the same document: the digest OF the
    proof, and the two digests carried INSIDE it.  Reading the file is
    what makes this a relay rather than a re-derivation -- ``go`` must
    not compute a digest the runner will then compare against itself.
    """

    from gpuwm.fetch import sha256_file

    proof_path = Path(prepared_root) / "proof.json"
    if not proof_path.is_file():
        raise GoRefusal(
            f"preparation finished without writing {proof_path}, so the "
            "forecast stage has nothing to bind against")
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    cache = payload.get("prepared_cache")
    content = cache.get("content_sha256") if isinstance(cache, dict) else None
    manifest_digest = payload.get("input_manifest_sha256")
    if not isinstance(content, str) or not isinstance(manifest_digest, str):
        raise GoRefusal(
            f"{proof_path} carries no single prepared-cache identity, "
            "which is what a multi-domain hierarchy product looks like.  "
            "Its runner is gpuwm-prepared-tree-forecast (module form: "
            "python -m gpuwm.prepared_domain_tree_forecast); see "
            f"{MANUAL_CHAIN}")
    return {"proof": sha256_file(proof_path),
            "source_manifest": manifest_digest,
            "prepared_content": content}


def wrfout_frames(plan: dict) -> list[Path]:
    """Every history file the forecast stage published, in time order."""

    return sorted((plan["run"] / "wrfout").glob("wrfout_d*"))


def render_command(plan: dict, frames: list[Path] | None = None) -> list[str]:
    """The sixth stage: turn the forecast into pictures.

    ``go`` used to stop after the forecast and print this line for the
    reader to paste.  That is one more baton pass, and a baton pass is
    where a first run ends -- a command printed instead of run reads as
    "it stopped", not "your turn".  Every other handoff in this chain is
    internal now; this one has no reason to be the exception, and the
    reader who typed ``gpuwm go`` wanted a forecast they can look at.

    ``gpuwm render`` takes FILES.  The documented line spells them
    ``.../wrfout/*`` and relies on a shell to expand that; a subprocess
    has no shell, and handing the directory over produced "unreadable
    wrfout (not a regular file)" on a run that had otherwise finished.
    So the frames are enumerated here.  ``frames=None`` keeps the
    glob spelling, which is what ``--dry-run`` should print: at that
    point the directory is empty and naming its contents would be a
    guess.

    The run folder is claimed ONCE, by the chain, and this stage draws
    into it: :func:`gpuwm.run_stamp.stage_flags` is what says so.  This
    command is composed twice for one run -- once by
    :mod:`gpuwm.first_products` on the first committed frame and once by
    :func:`_render_stage` at the end -- so a render that stamped for
    itself would put those two halves of one run's pictures in two
    wall-clock folders, and the finalize stage could no longer prove the
    early frame had already been drawn.
    """

    targets = ([str(frame) for frame in frames] if frames is not None
               else [str(plan["run"] / "wrfout" / "*")])
    command = [sys.executable, "-m", "gpuwm.cli", "render", *targets,
               "--out", str(plan["render"]),
               *run_stamp_module.stage_flags()]
    # `products` is `gpuwm render --products`' own spec, passed through
    # verbatim: a comma-separated list of product names, or `all`.  It
    # is NOT parsed or validated here -- the render front door owns that
    # vocabulary, and a second copy of it in this module is the
    # enumeration drift render.py's own catalog code already refuses to
    # pay for.  Absent leaves the default set exactly as it was.
    products = plan.get("render_products")
    if products:
        command += ["--products", str(products)]
    return command


def render_extra_missing() -> str | None:
    """Why rendering cannot run here, or ``None`` when it can.

    Asked of the RENDER FRONT DOOR, not of the Python import table,
    because the stage this gates is ``gpuwm render`` and that command
    chooses between two engines.  This function used to import ``wrf``
    and skip the stage when it was absent -- so a bare install with
    staged bridges, on which ``gpuwm render`` selects the rust engine
    and writes its full catalog (measured: 161 PNGs, 127 MB, no
    ``render`` extra installed), finished a whole forecast and then
    declined to draw it.  The gate named a package the stage it gates
    does not need.

    :func:`gpuwm.render.drawable_engine` is that one answer, shared, so
    "can this install draw?" cannot be answered differently by the chain
    and by the command the chain runs.
    """

    from gpuwm.render import drawable_engine

    engine, why = drawable_engine()
    return None if engine is not None else why


def _render_stage(plan: dict, *, explain: bool,
                  observer=None) -> bool:
    """Run the render stage; return whether anything was rendered.

    ``plan["render_products"] == "none"`` skips the stage outright.
    That spelling is deliberate: it lives in the same field as a product
    list rather than in a separate boolean, so "which products" has one
    answer and not two that can disagree.  `none` is not a product name
    in the render catalog, so it cannot collide with one.

    A run with an early render (see :mod:`gpuwm.first_products`) has
    already published its first frame while the forecast was still
    going.  That render is collected here, before anything is drawn, and
    the frame it claims is dropped from this stage's list -- but only
    after its receipt has been checked, digest by digest, against what is
    actually on disk.  An unproven claim is simply not used, and the
    frame is rendered again.

    The receipt is read off disk whether or not THIS process armed the
    render that wrote it, because on `gpuwm go` it never does.
    """

    if str(plan.get("render_products") or "").strip().lower() == "none":
        print("  -- render skipped: this run asked for no products "
              "(render_products = none).")
        return False
    missing = render_extra_missing()
    if missing is not None:
        # An ANNOUNCED absence of imagery, which is the lawful shape
        # here: the render law reserves weather fields for rw_wrfbatch
        # and names one fallback that serves none of these products, so
        # a chain with no usable renderer draws nothing and says so.  It
        # used to offer `pip install 'gpuwm[render]'` as a second way
        # out, which installed the matplotlib engine -- the second
        # fallback the law forbids (audit F7).
        print(f"  -- render skipped: {missing}")
        print("     remedy: gpuwm setup")
        print("     # stages the rust render engine, which draws the full "
              "catalog")
        print("     then:")
        print(f"       {printable(render_command(plan))}")
        return False
    frames = wrfout_frames(plan)
    if not frames:
        print("  -- render skipped: the forecast stage published no "
              f"wrfout frame under {plan['run'] / 'wrfout'}.")
        return False
    from gpuwm.first_products import published_frames

    already: list[Path] = []
    trigger = getattr(observer, "first_products", None)
    if trigger is not None:
        # Collected first.  The early render writes into this stage's
        # own output directory, so drawing before it has finished would
        # be two writers on one directory for no reason -- and its
        # receipt, which is what licenses the skip below, is the last
        # thing it publishes.
        trigger.wait()
    # The receipt is read off DISK, whether or not this process armed the
    # render that wrote it.
    #
    # THE DEFECT THIS CLOSES, measured on both 3080 walks: `gpuwm go`
    # does not host the early render -- it asks the runner subprocess for
    # one on its command line (`forecast_command`) -- so there is no
    # trigger object here, and the skip below never ran on the front door
    # people use.  This stage then redrew the frame the early render had
    # already published, overwriting those PNGs, and the published tree's
    # earliest picture carried THIS stage's timestamp (2m 45s) while the
    # receipt said 46s.  The receipt licenses nothing until
    # `published_frames` has re-checked the frame and every picture it
    # names against their recorded digests, so reading it off disk is
    # exactly as safe as reading it off a trigger this process happens to
    # own -- and it is what keeps the early picture, and its instant, in
    # the tree.
    frames, already, note = published_frames(frames, plan)
    if note is not None:
        print(f"  -- render: {note}")
    if not frames:
        # Every frame was published early.  Distinguished from the empty
        # case above because "nothing to do because it is done" and
        # "nothing to do because nothing was produced" are opposite
        # outcomes and used to print the same sentence.
        print(f"  -- render complete: {len(already)} frame(s) were "
              "published by the early render and verified by digest; "
              "nothing was left to draw.")
        return True
    _run_stage("render", render_command(plan, frames), explain=explain,
               observer=observer)
    return True


def resolve_bridge() -> Path:
    """The built ``gfs_grib2_bridge``, through the one resolver.

    :mod:`gpuwm.bridges` is where every other consumer looks, so a
    bridge staged by ``gpuwm setup`` or built in a checkout is found
    the same way here as it is by ``gpuwm doctor``.
    """

    from gpuwm import bridges

    found = bridges.find_bridge("gfs_grib2_bridge")
    if found is None:
        raise GoRefusal(
            "no built gfs_grib2_bridge, which every stage of this chain "
            "decodes through.\n"
            "  remedy: gpuwm setup\n"
            "  # or `gpuwm doctor --explain` for the build route on a "
            "platform with no published bundle")
    return found


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

def _elapsed_words(seconds: float) -> str:
    """``3m 07s`` -- an elapsed time a person reads without converting."""

    whole = int(seconds)
    return f"{whole // 60}m {whole % 60:02d}s"


def _progress_payload(progress: Path | None) -> dict | None:
    """The running stage's published progress file, as a dict.

    What :func:`_progress_note` renders for a person, handed to a
    machine unrendered.  Same source, same silence on a missing or
    half-written file -- a stage that has not got there yet is not an
    error.
    """

    if progress is None:
        return None
    try:
        payload = json.loads(Path(progress).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _progress_note(progress: Path | None) -> str:
    """What the running stage says about itself, if it says anything.

    The runner publishes ``progress.json`` beside its output and updates
    it as the forecast advances, which is a better answer than a
    stopwatch: "2m 40s elapsed" and "2m 40s elapsed, INTEGRATING, 1200
    model seconds" differ by whether the reader can tell it is moving.
    A missing or half-written file is not an error here -- it is a stage
    that has not got there yet -- so every failure to read one is
    silent.
    """

    if progress is None:
        return ""
    try:
        payload = json.loads(Path(progress).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    status = payload.get("status")
    model = payload.get("model_elapsed_seconds")
    parts = [str(status)] if isinstance(status, str) else []
    if isinstance(model, (int, float)) and model > 0:
        parts.append(f"{float(model):.0f} model seconds")
    return (", " + ", ".join(parts)) if parts else ""


def _notify(observer, event: str, **fields) -> None:
    """Tell an optional stage observer something, never failing the run.

    ``go`` is a chain of integrity checks; telemetry is not one of them,
    so an observer that raises must not take a forecast down.
    """

    if observer is None:
        return
    hook = getattr(observer, event, None)
    if hook is None:
        return
    try:
        hook(**fields)
    except Exception:  # noqa: BLE001 - telemetry never fails a stage
        pass


def _run_stage(label: str, command: list[str], *, explain: bool,
               progress: Path | None = None,
               heartbeat_seconds: float = HEARTBEAT_SECONDS,
               observer=None) -> None:
    """Run one stage; replay everything it said and stop if it failed.

    Output is captured so the default is one line per stage, and
    released in full the moment a stage fails -- the reason a stage
    refused is never summarized, because a chain that stops without
    saying why leaves the reader with five commands to re-run by hand
    to find out.

    Captured output means a quiet stage is a quiet terminal, and two of
    these stages run for minutes.  An owner watching one of them
    concluded the tool had hung.  So while a stage runs, this prints a
    line every ``heartbeat_seconds`` naming the stage, how long it has
    been going, and -- when the stage publishes a progress file -- what
    it says it is doing.  The heartbeat is a property of waiting, not of
    verbosity: it is not behind ``--explain``, because the reader who
    needs it is the one who did not pass any flags.
    """

    print(f"  .. {label}", flush=True)
    _notify(observer, "stage_begin", label=label, command=list(command))
    started = time.monotonic()
    # The subprocess still runs through ``subprocess.run`` -- one call,
    # one place, the same capture semantics -- and the waiting moves to
    # a worker thread so this one stays free to say that it is waiting.
    # Doing it the other way round (Popen plus a poll loop) would have
    # split the "how a stage is run" answer across two code paths.
    box: dict[str, object] = {}

    def _wait() -> None:
        try:
            # Popen + communicate is exactly what subprocess.run does;
            # spelling it out is what publishes the child's pid, which
            # the interrupt path has to be able to NAME (it does not
            # signal it -- see GoInterrupted).
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, errors="replace", cwd=str(_stage_cwd()),
                env=_stage_env())
            box["pid"] = proc.pid
            out, err = proc.communicate()
            box["completed"] = subprocess.CompletedProcess(
                command, proc.returncode, out, err)
        except BaseException as error:  # re-raised on this thread below
            box["error"] = error

    worker = threading.Thread(target=_wait, name=f"go-stage-{label}",
                              daemon=True)
    worker.start()
    next_beat = started + heartbeat_seconds
    try:
        while worker.is_alive():
            worker.join(timeout=0.2)
            now = time.monotonic()
            if worker.is_alive() and now >= next_beat:
                print(f"     .. {label}, {_elapsed_words(now - started)}"
                      f"{_progress_note(progress)}", flush=True)
                # The same beat the terminal gets, as fields.  The
                # payload is the stage's own published progress file --
                # the artifact, never the prose, which is this module's
                # standing rule for relayed values.
                _notify(observer, "stage_heartbeat", label=label,
                        elapsed_seconds=now - started,
                        progress=_progress_payload(progress))
                next_beat = now + heartbeat_seconds
        worker.join()
    except KeyboardInterrupt:
        # Ctrl-C.  The wait happens on this thread, so the interrupt
        # lands here; the stage subprocess is on the worker.  Hand the
        # stage name and that pid up and let go_main say it in one
        # sentence.  Nothing is signalled from here: in a terminal the
        # child already received its own SIGINT with the rest of the
        # foreground process group, and a `kill` issued by gpuwm at a
        # pid it merely observed is how a tool ends up stopping
        # something that was never its to stop.
        raise GoInterrupted(label, box.get("pid")) from None
    if "error" in box:
        raise box["error"]
    completed = box["completed"]
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        print(f"  FAILED  {label} (exit {completed.returncode})")
        # The stage's own refusal is usually its last few lines; replay
        # a readable tail by default and everything under --explain.
        lines = output.splitlines()
        tail = lines if explain else lines[-_FAILURE_TAIL_LINES:]
        if len(tail) < len(lines):
            print(f"    ... ({len(lines) - len(tail)} earlier line(s); "
                  "re-run with --explain to replay all of them)")
        for line in tail:
            print(f"    {line}")
        print(f"go: stopped at {label}; every later stage consumes this "
              "one's output, so nothing after it ran.")
        _notify(observer, "stage_end", label=label,
                exit_code=completed.returncode, ok=False,
                elapsed_seconds=time.monotonic() - started,
                progress=_progress_payload(progress))
        raise GoStageFailed(completed.returncode)
    _notify(observer, "stage_end", label=label, exit_code=0, ok=True,
            elapsed_seconds=time.monotonic() - started,
            progress=_progress_payload(progress))
    print(f"  ok      {label} ({_elapsed_words(time.monotonic() - started)})")
    if explain:
        for line in output.splitlines():
            print(f"    {line}")
    else:
        # Warnings must not need a flag to be seen: a passing stage's
        # warning lines surface, one line each, nothing else.
        #
        # `note:` joins `warning:` here, and for the same reason rather
        # than for tidiness.  This tree uses `note:` for "true, and worth
        # knowing, and not a fault" -- the render stage says it when a
        # frame's declared inputs are absent and a product was therefore
        # not drawn.  That sentence exists precisely so a skipped product
        # is not a silent success; a chain that captured it and printed
        # `ok render` would have re-created the silence one level up.
        for line in output.splitlines():
            head = line.lstrip().lower()
            if head.startswith("warning:") or head.startswith("note:"):
                print(f"    {line.strip()}")


#: Failure-replay tail length: enough to carry any refusal message this
#: tree prints plus a Python traceback's tail, small enough to read.
_FAILURE_TAIL_LINES = 30


#: The chain-stage primitive, under a public name.
#:
#: `go` owns the GFS chain, but "run one documented command, capture it,
#: heartbeat while it waits, replay everything it said if it failed, and
#: tell an observer" is not GFS-specific -- it is what running ANY stage
#: of ANY documented chain looks like here.  The HRRR route reuses it
#: rather than growing a second copy with slightly different capture and
#: failure-replay semantics, which is how two chains end up refusing
#: differently for the same reason.
run_stage = _run_stage


class GoStageFailed(Exception):
    """A stage exited nonzero; its output has already been replayed."""

    def __init__(self, code: int):
        super().__init__(f"stage exited {code}")
        self.code = code


def _tree_bytes(root) -> int:
    """Bytes under ROOT, counting what can be counted.

    Best-effort by construction: a tree a failed stage was mid-write in
    can lose a file between the walk and the stat, and a size is not
    worth an exception on an error path.
    """
    total = 0
    try:
        for path in Path(root).rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _partial_tree_note(root) -> str:
    """What a failed run left on disk, and that removing it is safe."""
    root = Path(root)
    if not root.exists():
        return (f"go: {root} was not created, so there is nothing on disk "
                "from this run.")
    return (f"go: {root} is a partial tree with no certification capsule, "
            f"holding {_human_bytes(_tree_bytes(root))}.  It is kept "
            f"because it is the evidence of what failed; nothing later "
            f"reads it, so it is safe to remove once you are done with "
            f"it.  Every stage is create-only, so a re-run needs a new "
            f"--outdir either way.")
def _experimental_labels(plan: dict) -> tuple[str, ...]:
    """(label, reason) for the experimental options this plan selects.

    Read from the registry through the config, never from a scheme name,
    so registering a second experimental option surfaces it here without
    a further edit.  Any failure to read the config is silent by design:
    the banner is not the place a broken config is diagnosed, and the
    loader that IS that place runs a moment later.
    """
    try:
        from gpuwm.experiment import is_experiment_toml, load_experiment
        from gpuwm.physics_compat import _experimental_components

        path = plan.get("config")
        if path is None:
            return ()
        if is_experiment_toml(path):
            configs = [dc.run for dc in load_experiment(path).domains]
        else:
            from gpuwm.config import load_config
            configs = [load_config(path)]
        clauses: dict[str, str] = {}
        for cfg in configs:
            for label, reason in _experimental_components(cfg):
                clauses.setdefault(label, reason)
        return tuple(sorted(clauses.items()))
    except Exception:
        return ()


#: Ctrl-C's exit code, by the shell convention 128 + SIGINT.
INTERRUPT_EXIT_CODE = 130


class GoInterrupted(Exception):
    """Ctrl-C arrived while ``label``'s subprocess (``pid``) was running.

    Carries the pid so the message can NAME the process gpuwm was
    waiting on, and deliberately carries no authority to kill it.  The
    bounded contract, shared with the multi-GPU orchestration lane:
    Ctrl-C returns 130, kills no child and no unrelated process, and an
    unobserved child pid is printed rather than acted on.
    """

    def __init__(self, label: str, pid: int | None):
        super().__init__(f"interrupted during {label}")
        self.label = label
        self.pid = pid


def _physics_words(plan: dict) -> str:
    """The banner's physics clause: profile id, or one honest sentence."""

    experimental = _experimental_labels(plan)
    if experimental:
        # An experimental option outranks the profile id in the banner.
        # "supported, not yet WRF-verified" would be the wrong sentence
        # for a scheme WRF does not have -- it promises a comparison
        # that is merely outstanding, and there is none to be
        # outstanding.  Warn-not-block: this is the wording, not a gate.
        # The REASON comes from physics_compat so the banner and the two
        # runners cannot say different things; only the shout is local.
        return ", ".join(
            f"{label}: {reason.replace('experimental,', 'EXPERIMENTAL,', 1)}"
            for label, reason in experimental)
    if plan.get("profile") is not None:
        return f"profile {plan['profile']}"
    return ("physics from the config as written (supported, not yet "
            "WRF-verified)")


def memory_refusal_text(gate: dict) -> str:
    """The go memory refusal, with remedies that can actually succeed.

    The 2026-08-19 3080 walk followed the previous remedy verbatim --
    ``gpuwm domain --vram-gib 8`` -- and was refused at every grid size,
    because the flag names a CARD and the number fed to it was a
    free-VRAM figure, pointed back into the same accounting that had
    just refused.  A refusal's remedy must be reachable: the bare wizard
    measures this card itself, a lighter suite is ranked by the priced
    envelope inside the wizard's own refusal, and a bigger card is a
    bigger card.  The measured free-VRAM sentence stays: it is the one
    number in this refusal read directly off the machine.
    """

    free = gate.get("free_bytes")
    free_words = ("" if free is None else
                  f", and the card has {free / (1024 ** 3):.2f} GiB "
                  "free right now")
    return (
        f"this configuration will not fit: {gate['verdict']}{free_words}."
        "  Refusing here, BEFORE the fetch stage downloads the forcing "
        "data, rather than in preprocessing after it.\n"
        "  remedy: re-size against this machine -- gpuwm domain ... "
        "(bare, it measures this card) -- or pick a lighter "
        "--physics-profile (the wizard's refusal ranks them by priced "
        "envelope), or free VRAM and re-run, or use a larger card\n"
        "  # gpuwm go CONFIG --no-memory-gate runs it anyway")


def memory_gate(plan: dict, *, vram_gib: float | None = None) -> dict:
    """Price every phase of ``plan`` against this card, before the fetch.

    The whole point is WHERE this runs.  A configuration that cannot fit
    used to discover that in preprocessing -- after `gpuwm fetch` had
    pulled 81 GFS files down -- because the only memory estimate anyone
    computed described the forecast.  Both phases are priced here, at
    planning time, and the verdict names the binding phase and its
    number in one sentence.

    Returns a dict with ``verdict`` (always), ``refuse`` (a genuine OOM
    prediction: the binding phase does not fit even the whole card's
    free VRAM) and ``warn`` (it fits free VRAM but not the reserved
    budget -- said out loud, never blocking).
    """

    from gpuwm.core.preflight import (ReservePolicy, _load_experiment_any,
                                      config_forcing_source,
                                      device_memory_probe_reason,
                                      device_memory_probe_subprocess,
                                      estimate_phases,
                                      profile_from_device_probe,
                                      unpriced_ingest_note)

    config = Path(plan["config"])
    source = config_forcing_source(config)
    if source is None:
        return {"verdict": unpriced_ingest_note(config, plan.get("source")),
                "refuse": False, "warn": True, "free_bytes": None}
    exp = _load_experiment_any(config)
    # This gate is about THIS machine -- it is the last thing between the
    # user and a download -- so the non-pool terms are priced against the
    # card that is actually here, not against the 170-SM reference the
    # module was calibrated on.  `gpuwm check` reads the same numbers off
    # the same device; the two must not disagree about one card.
    #
    # Both device questions -- the free VRAM the budget subtracts from
    # and the card's local-memory profile -- are asked in ONE short-lived
    # SUBPROCESS, never in-process.  An in-process memGetInfo or
    # deviceGetLimit stands up a CUDA primary context, and this process
    # outlives its gate as the chain's stage runner and progress printer:
    # that context sat on the card for the entire run -- measured
    # 0.486 GiB on the RTX 5090 -- as a consumer no term of the budget
    # computed below names.  The probe's context dies with the probe, so
    # the printer holds no device memory at all, and the forecast stage
    # (whose own context IS priced, in the reserve's device-overhead
    # term) gets the card the verdict described.
    # THE seam the outside world enters through, and the only one: every
    # caller and every test pins this function, so the gate must keep
    # asking it rather than the private helper underneath.  The reason is
    # asked for SEPARATELY and only when there are no numbers, which is
    # the one path where a second probe costs nothing anybody is waiting
    # on -- and where saying "no card here" for an uninstalled runtime
    # was the whole defect.
    probe = device_memory_probe_subprocess()
    probe_reason = None if probe is not None else device_memory_probe_reason()
    profile = profile_from_device_probe(probe)
    # The cadence the domain was SIZED against, from the same [fetch]
    # table the fetch stage reads, so the two cannot disagree.
    cadence_h = plan.get("cadence")
    # [tiles] mode = "auto" with no pinned tiling is the planner's decision,
    # and the planner needs a card.  Build its Machine from the SUBPROCESS
    # probe already taken above rather than letting autoplan.Machine.detect
    # stand a CUDA context up in this process -- the same reason the probe
    # is a subprocess at all (0.486 GiB held for the whole run, MEASURED).
    machine = _planner_machine(probe)
    phases = estimate_phases(
        exp, source=source, vram_gib=vram_gib, profile=profile,
        machine=machine,
        ingest_forcing_interval_seconds=(
            float(cadence_h) * 3600.0 if cadence_h else None))

    if probe is None:
        # No numbers: price the phases and print the verdict, but never
        # refuse on a card we cannot see.
        #
        # The verdict now SAYS WHY there are no numbers.  It used to say
        # "no card here" for every cause including an uninstalled CuPy,
        # which is how this gate swallowed the one gap that stops the
        # chain dead.  The front door refuses that case before this
        # function is reached; the reason is still carried here so a
        # caller reading the gate's own dict is not told a fiction.
        return {"verdict": f"{phases.verdict(None)} ({probe_reason})"
                           if probe_reason else phases.verdict(None),
                "refuse": False, "warn": False, "free_bytes": None,
                "probe_reason": probe_reason, "phases": phases}
    free = int(probe["free_bytes"])
    # The budget the ENVELOPE is compared against, from the wizard's own
    # seam so the two doors cannot disagree about one card.  It is free
    # VRAM minus what this process's envelope does not model -- other
    # processes -- and NOT the allocation reserve, which carries the CUDA
    # context and the local-memory backing store that
    # ``peak_envelope_bytes`` already contains.  Charging both warned
    # about a card the run fits (task 206).
    from gpuwm.domain_wizard import sizing_budget_bytes

    budget = sizing_budget_bytes(
        exp, free_bytes=free, vram_gib=vram_gib,
        forcing_interval_seconds=(float(cadence_h) * 3600.0
                                  if cadence_h else None),
        profile=profile)
    peak = phases.peak_envelope_bytes
    verdict = phases.verdict(budget)
    refuse = peak > free
    # THE PINNED STORE IS A REFUSAL TOO, and only for a streamed run: the
    # domain lives in host RAM there, so a config whose store cannot be
    # page-locked dies at attach -- after the download, which is exactly
    # what this gate exists on this side of.  Priced only when the host
    # total could be read; unknown host RAM never refuses.
    if phases.streamed_forecast:
        env = phases.streamed
        if (env.host_budget_bytes is not None
                and env.host_bytes > env.host_budget_bytes):
            refuse = True
            verdict += (
                f"; and the pinned host store is "
                f"{env.host_bytes / (1024 ** 3):.2f} GiB against a "
                f"{env.host_budget_bytes / (1024 ** 3):.2f} GiB page-locking "
                "budget, which is where a streamed domain actually lives")
    return {
        "verdict": verdict,
        "refuse": refuse,
        "warn": peak > budget,
        "free_bytes": free,
        "budget_bytes": budget,
        # Present on every return of this function, so a caller never
        # has to distinguish "the key is absent" from "there was no
        # reason"; None means the probe answered.
        "probe_reason": None,
        "phases": phases,
    }


def _planner_machine(probe):
    """A :class:`tilestream.autoplan.Machine` from this gate's own probe.

    The arithmetic lives in :func:`gpuwm.core.streaming.planner_machine`,
    beside the envelope it feeds, because ``gpuwm check`` needs the same
    Machine built from a DECLARED budget rather than a probe and two
    copies of this would be two answers about one card.
    """
    from gpuwm.core.streaming import planner_machine

    return planner_machine(
        vram_bytes=None if probe is None else int(probe["free_bytes"]),
        name="gpuwm go probe")


def geography_refusal(geog_root: Path) -> str | None:
    """Why the prepare stage cannot build static fields, or ``None``.

    The prepare stage builds every domain's terrain, land use, soil and
    greenness out of the staged WPS_GEOG tree.  Without it there is no
    run -- and what a first-time reader used to get for that was three
    successful stages, a download, and then this, verbatim:

        FAILED  prepare (exit 2)
          rw-wps --source gfs: /.../WPS_GEOG/topo_gmted2010_30s/index.

    No verb, no remedy, and a sentence that ends in a path and a full
    stop.  Meanwhile ``gpuwm doctor`` had the whole answer, including
    the command, and ``go`` never asked it.

    Asked here for the same reason the memory gate is: BEFORE the fetch
    stage spends the user's bandwidth on forcing data for a run that
    cannot preprocess.

    This is a hard refusal and stays one.  It is not the warn-not-block
    case -- nothing here is a prediction that might be wrong.  The
    input is absent, the stage that needs it is next, and it was
    measured failing.  ``gpuwm doctor`` correctly exits 0 on the same
    gap, because a ~16 GB download nobody opted into is not a broken
    install; that is a statement about the INSTALL.  This is a
    statement about a RUN, and the two answers differ honestly.
    """

    from gpuwm.doctor import geography_gaps

    gaps = geography_gaps(Path(geog_root))
    if not gaps:
        return None
    return layered(
        f"the staged WPS_GEOG tree is not usable: "
        + "; ".join(gap.brief or gap.detail for gap in gaps) + ".\n"
        "  Every domain's static fields (terrain, land use, soil, "
        "greenness) are built from it by the prepare stage, so no "
        "stage after it can run.\n"
        "  remedy: gpuwm fetch-geog\n"
        f"  # stages the nine required datasets into {geog_root}\n"
        "  # gpuwm go CONFIG --geog-root DIR uses a tree staged elsewhere",
        "Refusing here, before the fetch stage downloads the forcing "
        "data, rather than in preprocessing after it.\n\n"
        "`gpuwm doctor` reports this same gap and still exits 0.  That "
        "is not a disagreement: the download is an explicit opt-in, so "
        "an install that did everything its documentation asked is not "
        "a broken install, and doctor is describing the install.  This "
        "message describes a RUN, which cannot proceed without the "
        "tree.\n\n"
        + "\n".join(f"  {gap.name}: {gap.detail}" for gap in gaps))


def go_main(args, *, observer=None, allow_tree=False) -> int:
    """The documented GFS chain, run in order.

    ``observer`` is an optional structured consumer of the same stage
    boundaries the terminal gets: ``stage_begin``, ``stage_heartbeat``
    (carrying the running stage's own published progress file, never
    its prose) and ``stage_end``.  It also switches the forecast stage
    from a subprocess to an in-process call, which is the only way a
    consumer can be told about each wrfout AS it lands rather than at
    the next heartbeat -- the runner's per-domain writer raises that
    hook on the thread that publishes the file.

    ``None`` -- every existing caller, including `gpuwm go` itself --
    changes nothing: no notifications, and the forecast stays the
    subprocess it has always been.

    ``allow_tree`` lets a multi-domain config through.  Off by default,
    so `gpuwm go`'s refusal is byte-identical: that refusal protects an
    interactive reader from a chain whose forecast stage would not run
    their config.  A caller that dispatches to the tree runner instead
    -- run-plan does -- is not the reader that refusal is for.
    """

    from gpuwm import chain_events
    from gpuwm.fetch import sha256_file
    from gpuwm.geog_assets import default_geog_root
    from gpuwm.gfs_direct import prepare_progress_path

    explain = explain_enabled(args)
    # DEFAULT-ON.  A caller that brought its own observer owns its own
    # telemetry -- `gpuwm run-plan` writes this exact grammar into its
    # own run directory and a second writer would interleave two runs'
    # sequences in one file.  Everybody else, which is everybody who
    # types `gpuwm go`, gets the stage timings on disk without asking.
    chain = chain_events.GoChainEvents() if observer is None else None
    if chain is not None:
        observer = chain
    # THE first thing this chain does, before it reads a config, resolves
    # a bridge, downloads a byte or writes a directory.
    #
    # `gpuwm.cli.main` has normally already asked the same question
    # through the same table; it is asked again here because this
    # function is also `gpuwm run-plan`'s go route, which enters without
    # that boundary.  Two doors, one preflight, one message.
    #
    # What it replaces was measured: the memory gate below reads the
    # card through a subprocess that exits 3 for ANY failure, so a
    # missing CuPy was indistinguishable from a machine with no card and
    # the gate said "no card here" and waved the chain on.  It then ran
    # authority -> fetch (gigabytes) -> manifest -> prepare (large disk
    # writes) and died at the forecast stage relaying the child's raw
    # traceback, in which no extra was ever named.
    unmet = capabilities.missing(
        capabilities.COMMAND_REQUIREMENTS.get("go", ()))
    if getattr(args, "dry_run", False):
        # A dry run spends nothing, so it prints rather than refuses --
        # but it must not print six stages as though they would run.
        for requirement in unmet:
            print(f"go: WARNING -- this install cannot run stage 5 "
                  f"(forecast): {requirement.label} is not installed.",
                  file=sys.stderr)
            print(requirement.remedy, file=sys.stderr)
    else:
        capabilities.require_for_command("go")
    plan = plan_from_config(args.config,
                            outdir=getattr(args, "outdir", None),
                            data_dir=getattr(args, "data_dir", None),
                            render_products=getattr(
                                args, "render_products", None),
                            allow_tree=allow_tree,
                            run_stamp=run_stamp_module.run_stamp_enabled(args))
    bridge = resolve_bridge()
    geog_root = (Path(args.geog_root) if getattr(args, "geog_root", None)
                 else default_geog_root())

    manifest = plan["data"] / "gfs-input-manifest.json"
    cycle_stamp = _cycle_stamp(plan["cycle"])

    if getattr(args, "dry_run", False):
        print(f"go: {plan['config']} -- source {plan['source']}, cycle "
              f"{plan['cycle']}, {plan['hours']} h, "
              f"{_physics_words(plan)}{_lead_note(plan)}")
        print(_run_folder_note(plan))
        if plan.get("tiles"):
            print(f"go: {plan['tiles']['sentence']}")
        print("")
        for label, command in (
                ("1. authority", authority_command(plan)),
                ("2. fetch", fetch_command(plan)),
                ("3. manifest", manifest_command(plan, bridge)),
        ):
            print(f"{label}\n     {printable(command)}")
        # The last two carry values that do not exist yet.  Naming the
        # FILE each one is read from beats printing a plausible-looking
        # hash: the point of a dry run is to show what will happen, and
        # what will happen is a read from that artifact.
        print("4. prepare\n     " + printable(prepare_command(
            plan, bridge, manifest=manifest,
            manifest_sha256="<sha256 of " + manifest.name + ", after step 3>",
            cycle_stamp=cycle_stamp, geog_root=geog_root)))
        print("5. forecast\n     " + printable(forecast_command(plan, {
            "proof": "<sha256 of prepared/proof.json, after step 4>",
            "source_manifest": "<proof.json input_manifest_sha256>",
            "prepared_content":
                "<proof.json prepared_cache.content_sha256>"})))
        print("6. render\n     " + printable(render_command(plan)))
        return 0

    # Re-running `gpuwm go` is the second thing anyone does, and the
    # first stage is create-only: the runner will not materialize an
    # authority into a directory an earlier run owns, because the
    # receipt it writes has to describe one run.  That refusal is right
    # and its message names --output-directory, a flag this caller never
    # typed.  Answer it here, in the caller's own vocabulary, before
    # spending a stage to reach it.
    if plan["authority"].exists():
        # Reachable in exactly two ways now that every run claims its own
        # stamped folder: --outdir named an EXISTING run folder (which is
        # honoured verbatim, so the caller has pointed a second chain at
        # a finished run's tree), or --run-stamp off put this run back in
        # the shared directory the last one used.  Both are named, in
        # that order, because the remedy differs.
        named = run_stamp_module.is_run_folder(plan["root"])
        raise GoRefusal(
            f"{plan['authority']} already exists, so this chain has been "
            f"run into {plan['root']} before.  Every stage is create-only "
            "on purpose -- merging two runs into one tree would publish "
            "receipts describing neither.\n"
            + (f"  {plan['root'].name} is a run folder you named on the "
               "command line, so it was used as given rather than "
               "getting a stamped child of its own.\n"
               f"  remedy: gpuwm go {_quote(plan['config'])} --outdir "
               f"{_quote(str(plan['case_root']))}, which claims a fresh "
               "run folder beside it\n"
               if named else
               "  remedy: drop --run-stamp off and each run claims its "
               f"own {run_stamp_module.RUN_PREFIX}... folder under "
               f"{plan['case_root']}, or pass --outdir <a new "
               "directory>\n")
            + f"  # or remove {plan['root']} if that run is finished with")

    print(f"go: {plan['config']} -- source {plan['source']}, cycle "
          f"{plan['cycle']}, {plan['hours']} h, {_physics_words(plan)}"
          f"{_lead_note(plan)}")
    # The run-folder line, BEFORE the gates -- the same line, from the
    # same function, that --dry-run prints second.  It used to print
    # only after the memory and geography gates, so both of the
    # 2.4.1-upgrader walk's real attempts refused without ever naming
    # the 2.5.0 layout, and no cheap door revealed the new folder
    # naming (UX finding N18).  Announced here it still spends nothing:
    # the folder is only CLAIMED after the gates, below, and if the
    # claim lands on a different stamp the line prints again corrected.
    predicted_root = plan["root"]
    print(_run_folder_note(plan))
    # The routing a [tiles] table chose, said before any stage runs.  The
    # forecast stage prints its own decision line too, but that stage's
    # stdout is captured -- a reader watching this terminal would learn
    # their run streamed only from report.json, after the fact.
    if plan.get("tiles"):
        print(f"go: {plan['tiles']['sentence']}")

    # The interrupt contract covers the gate too: the memory probe
    # is a real subprocess (first-run staging), and a Ctrl-C landing
    # there deserves the same one sentence and rc 130 as any stage.
    try:
        # BEFORE the download, not after it.  `gpuwm fetch` is the stage that
        # costs the user gigabytes and minutes; a configuration that cannot
        # fit has to be told so on this side of it.
        if not getattr(args, "no_memory_gate", False):
            gate = memory_gate(plan)
            print(f"go: memory -- {gate['verdict']}")
            if gate["refuse"]:
                raise GoRefusal(memory_refusal_text(gate))
            if gate["warn"]:
                print("go: WARNING -- that is above the reserved budget, "
                      "though it fits the free VRAM measured just now.  "
                      "Proceeding; a driver/other-process spike could still "
                      "OOM this run.")

        # Same rule as the memory gate, same side of the download.
        geography = geography_refusal(geog_root)
        if geography is not None:
            raise GoRefusal(geography)

        # THE STAGE TIMINGS START LANDING ON DISK HERE, by default, for
        # a caller that passed no flags and asked for no observer.
        #
        # Here and not at the top of the function on purpose: the gates
        # above refuse without spending anything, and a run they refuse
        # must leave the disk exactly as it found it.  Opening the
        # stream creates the run root.  The boot stage is not lost by
        # waiting -- `chain.open` emits it from the launch anchor.
        #
        # And the run folder is CLAIMED here, on the same side of the
        # gates and for the same reason: the plan above only named it.
        claim_run_root(plan)
        if plan["root"] != predicted_root:
            # The claim landed on a different stamp than the plan named
            # (two chains racing the same second allocate neighbours):
            # correct the announcement above, once.
            print(_run_folder_note(plan))
        if chain is not None:
            chain.open(plan["root"] / chain_events.CHAIN_EVENTS_FILENAME,
                       plan=plan)

        _run_stage("authority", authority_command(plan), explain=explain,
                   observer=observer)
        _run_stage("fetch", fetch_command(plan), explain=explain,
                   observer=observer)
        _run_stage("manifest", manifest_command(plan, bridge),
                   observer=observer,
                   explain=explain)
        if not manifest.is_file():
            raise GoRefusal(
                f"the manifest stage wrote no {manifest}, so the "
                "preparation stage has nothing to bind against")
        _run_stage("prepare", prepare_command(
            plan, bridge, manifest=manifest,
            manifest_sha256=sha256_file(manifest),
            cycle_stamp=cycle_stamp, geog_root=geog_root), explain=explain,
            # Not `prepared/progress.json`: the stage builds its whole
            # product in a staging directory and publishes it with one
            # rename, so nothing can live inside `prepared/` until
            # there is nothing left to report.  The path is asked of
            # the publisher rather than spelled here, so the writer and
            # this poller cannot disagree.
            progress=prepare_progress_path(plan["prepared"]),
            observer=observer)
        # A hierarchy proof carries no single prepared-cache identity,
        # and proof_digests says so by refusing.  The tree arm reads its
        # own one digest instead, so it is not asked for here.
        digests = ({} if plan.get("domains", 1) > 1
                   else proof_digests(plan["prepared"]))
        # The first frame the forecast commits is the analysis, and an
        # observer that wants it rendered as it lands is told so before
        # the forecast starts -- with THIS plan, the one the render
        # stage below runs on.  `_notify` because an observer without
        # the hook (there is no such thing in tree, but `go` takes any
        # duck) must not be a reason a chain stops.
        _notify(observer, "arm_first_products", render_plan=plan)
        _run_forecast(plan, digests, explain=explain, observer=observer)
        rendered = _render_stage(plan, explain=explain,
                                 observer=observer)
    except GoRefusal:
        # A refusal reached from INSIDE the chain -- a manifest stage
        # that wrote nothing, a proof with no single cache identity.
        # The stream is open by then and would otherwise end mid-run,
        # which reads exactly like a killed process.
        if chain is not None:
            chain.finish(status="REFUSED", exit_code=2)
        raise
    except GoStageFailed as failure:
        # D-05.  The interrupted arm below has always told the reader what
        # is on disk; this one raised and said nothing, so a failed
        # prepare left a scratch tree of a few hundred MB with no mention
        # anywhere that it existed, was incomplete, or could be removed.
        # Nothing is deleted here: the tree is the only evidence of what
        # went wrong, and a tool that tidies away the failure a person is
        # about to report is worse than one that leaves it.
        if chain is not None:
            chain.finish(status="FAILED", exit_code=failure.code)
        print(_partial_tree_note(plan["root"]), file=sys.stderr)
        return failure.code
    except GoInterrupted as stop:
        # One sentence, 130, and the truth about what is on disk.  The
        # partial tree has no certification capsule, which is what makes
        # its incompleteness visible to every receipt reader.
        child = ("" if stop.pid is None else
                 f"  # the {stop.label} process was pid {stop.pid}; gpuwm "
                 "signalled nothing and killed nothing\n")
        print(render(layered(
            f"go: interrupted during {stop.label}; no later stage ran and "
            f"{plan['root']} is a partial tree with no certification "
            "capsule.\n"
            f"{child}"
            f"  remedy: gpuwm go {_quote(plan['config'])} --outdir "
            "<a new directory>   # every stage is create-only",
            "Ctrl-C sends SIGINT to the whole foreground process group, "
            "so the stage subprocess received it directly and gpuwm did "
            "not (and will not) signal any pid itself -- a tool that "
            "kills pids it merely observed is a tool that eventually "
            "kills the wrong one. If the stage was launched into the "
            "background by a shell, SIGINT is SIG_IGN for the whole job "
            "and neither process can see a Ctrl-C at all; send SIGTERM "
            "there instead."),
            explain=explain, command="gpuwm go"), file=sys.stderr)
        if chain is not None:
            chain.finish(status="INTERRUPTED",
                         exit_code=INTERRUPT_EXIT_CODE)
        return INTERRUPT_EXIT_CODE

    # The forecast writes its own validity verdict (health, stability,
    # input-identity gates) into report.json; say it here so the chain
    # ends with the one line a reader is looking for.
    try:
        report = json.loads(
            (plan["run"] / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    verdict = report.get("status")
    delivered = _boundary_interval_refusal(plan, report)
    if delivered is not None:
        if chain is not None:
            chain.finish(status="FAILED", exit_code=1)
        print(f"go: {delivered}", file=sys.stderr)
        return 1
    if verdict:
        print(f"go: forecast validity {verdict}")
    print(f"go: wrote {plan['run']}")
    if rendered:
        print(f"go: rendered {plan['render']}")
    if chain is not None:
        summary = chain.finish(status="SUCCESS")
        # THE HEADLINE NUMBER, said out loud at the end of the run that
        # produced it.  It existed as an engine measurement and was
        # reachable only from `gpuwm run-plan`; a reader of `gpuwm go`
        # had to compare PNG mtimes against when they hit return.
        ttfp = summary.get("time_to_first_plot_seconds")
        if ttfp is not None:
            print(f"go: time to first plot {_elapsed_words(ttfp)} "
                  f"({summary['time_to_first_plot_source']})")
        # The path is stated only when there IS one.  A stream that
        # could not be opened is not a reason to fail a finished
        # forecast, and it is certainly not a reason to point a reader
        # at a file that does not exist.
        where = ("" if chain.path is None
                 else f"; per-stage timings in {chain.path}")
        print("go: launch to done "
              f"{_elapsed_words(summary['wall_seconds'])}{where}")
    return 0


def _boundary_interval_refusal(plan: dict, report: dict) -> str | None:
    """Refuse a run whose lateral boundaries are not the cadence asked for.

    The forecast records ``input.boundary_interval_seconds`` -- the real
    spacing of the lateral boundary times it integrated against.  When
    the ``[fetch]`` table names a cadence, that number is what the
    config asked for, and the two disagreeing means the run tracked the
    driving model on a different clock from the one the file declares.

    Checked against the RECEIPT rather than against the fetch command,
    because the defect this closes was a fetch command that looked right
    and a run that was three times coarser: exit 0, "forecast validity
    PASS", and nothing anywhere naming the substitution.  A verdict that
    can only be reached by reading report.json by hand is not a verdict.
    """

    cadence = plan.get("cadence")
    if cadence is None:
        return None
    observed = report.get("input", {}).get("boundary_interval_seconds")
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        return None
    requested = float(cadence) * 3600.0
    if float(observed) == requested:
        return None
    return (
        f"the run integrated against {float(observed) / 3600.0:g} h lateral "
        f"boundaries, but [fetch] cadence = {cadence} in "
        f"{plan['config'].name} asks for {cadence:g} h.  The boundary clock "
        "is what ties a limited-area run to its driving model, so this is "
        "not a rounding difference and the run above is not the run the "
        "config describes.  Re-run after making the two agree; "
        f"{plan['run'] / 'report.json'} records what was delivered.")


def _cycle_stamp(cycle: str) -> str:
    """``YYYY-MM-DD_HH:MM:SS`` -- the stamp the front door's --cycle takes.

    The config records the cycle as ``YYYY-MM-DDTHH``; the front door
    wants the WRF-style stamp.  One conversion, in one place, rather
    than a second cycle spelling in the config.
    """

    from gpuwm.fetch import parse_cycle

    return f"{parse_cycle(cycle, 'gfs'):%Y-%m-%d_%H:%M:%S}"


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "go",
        help="run the documented GFS chain for a wizard-authored config "
             "-- authority, fetch, front-door manifest, rw-wps, forecast "
             "-- carrying each stage's digests to the next instead of "
             "asking you to copy them")
    parser.add_argument("config", type=Path, metavar="CONFIG",
                        help="a single-domain GFS experiment TOML emitted "
                             "by `gpuwm domain --source gfs` -- the "
                             "wizard's default emission is exactly this "
                             "shape (--physics-profile optional: an "
                             "unbound config's own suite runs as written)")
    parser.add_argument("--outdir", type=Path, default=None, metavar="DIR",
                        help="the case directory: it holds the cached "
                             "download and one timestamped run folder "
                             "per run, each carrying that run's "
                             "authority, prepared, run and png trees "
                             "(default <config-stem>-go beside the "
                             "config).  Point it at an existing "
                             "run-... folder and that folder is used as "
                             "given")
    run_stamp_module.add_argument(
        parser, artifacts="authority, prepared, run and png trees",
        option="--outdir")
    parser.add_argument("--data-dir", type=Path, default=None, metavar="DIR",
                        dest="data_dir",
                        help="reuse an existing `gpuwm fetch` download "
                             "instead of fetching into <outdir>/data")
    parser.add_argument("--geog-root", type=Path, default=None, metavar="DIR",
                        help="staged WPS_GEOG tree (default: the one "
                             "`gpuwm fetch-geog` stages into)")
    # `plan_from_config` has always carried `render_products`, and
    # `gpuwm run-plan` has always been able to set it; `go` -- the door
    # a reader actually types -- had no spelling for it, so the only way
    # to run this chain against a NAMED product set was to drive the
    # stages by hand.  `gpuwm speedrun` needs exactly that (a course's
    # product set is part of what makes two records comparable), and so
    # does anyone who wants four charts instead of the whole catalog.
    parser.add_argument("--products", default=None, dest="render_products",
                        metavar="LIST",
                        help="which products the render stage draws: a "
                             "comma-separated list of catalog slugs, 'all' "
                             "(the default -- the renderer's whole "
                             "catalog), or 'none' to stop after the "
                             "forecast.  The same spelling `gpuwm render "
                             "--products` takes")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="print the six commands, filled in, and "
                             "exit without running any of them")
    parser.add_argument("--no-memory-gate", action="store_true",
                        dest="no_memory_gate",
                        help="skip the before-the-fetch memory check that "
                             "refuses a configuration whose binding phase "
                             "cannot fit this card's free VRAM")
    parser.set_defaults(func=go_main)
    return parser


__all__ = [
    "GoRefusal", "GoStageFailed", "HEARTBEAT_SECONDS", "MANUAL_CHAIN",
    "ORCHESTRATED_SOURCES", "RUNNER_MODULE", "RUNNER_RELATIVE",
    "TREE_RUNNER_MODULE", "tree_forecast_command",
    "authority_command", "fetch_command", "forecast_command", "go_main",
    "manifest_command", "memory_gate", "plan_from_config",
    "prepare_command", "printable",
    "proof_digests", "register_cli", "render_command",
    "render_extra_missing", "resolve_bridge", "run_stage",
    "wrfout_frames",
]
