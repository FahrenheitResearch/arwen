"""``gpuwm go``: launch the config through its declared native route.

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

The configuration chooses the execution route. Declared `[case_data]`
inputs use the experiment runner; other sources use their registered
native preparation chain. Both routes share the human progress and render
stages, and each retains its own input and physics validation.

"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from gpuwm import capabilities
from gpuwm import run_stamp as run_stamp_module
from gpuwm.explain import explain_enabled, layered, render
# The default product spec, declared where the early render declares it
# and re-exported below rather than written twice.  Safe at module
# scope: first_products reaches back into this module only from inside
# its functions.
from gpuwm.first_products import (
    DEFAULT_RENDER_PRODUCTS as _first_products_default)
# The two-stage route is spelled in ONE place (the seam that IS those
# two stages), and every door that points a reader at it renders from
# there -- this refusal included.

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

#: Which products a front door draws when its caller named none.
#:
#: ONE literal in this language, and it is not here: the early render
#: already had to spell it (:data:`gpuwm.first_products.DEFAULT_RENDER_PRODUCTS`)
#: because it is handed a command line, and the finalize stage compares
#: the two specs before it may skip a frame.  This is that name, read
#: rather than repeated, so `gpuwm go` and `gpuwm downscale` cannot
#: default to two catalogs.  The terminal declares its own copy in Rust
#: and says so where it does.
DEFAULT_RENDER_PRODUCTS = _first_products_default

#: The history frames a run publishes, as a glob.
#:
#: Both readers of "which files are this run's frames" spell it here:
#: :func:`wrfout_frames`, which enumerates them, and
#: :func:`render_command`, which prints the pattern for a reader to
#: expand.  They used to disagree -- the command globbed ``*`` -- and on
#: a route whose frames live in the RUN ROOT rather than in a
#: ``wrfout/`` subdirectory (a downscaled child does) that swept
#: ``events.jsonl``, ``report.json``, the checkpoints and the picture
#: tree into the renderer, which stopped on "NetCDF: Unknown file
#: format" at exit 2.
WRFOUT_GLOB = "wrfout_d*"

#: Compatibility export for callers of the former GFS-only go interface.
#: Dispatch uses runplan.prepared_chain_for_source, not this historical tuple.
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


def checked_config_fetch_cycle(fetch_table: dict, *, start_time=None):
    """Keep a saved experiment and its download on the same fixed clock."""
    from datetime import datetime, timedelta
    from gpuwm.fetch import parse_cycle

    raw = str(fetch_table["cycle"])
    if raw.strip().lower() == "latest":
        raise GoRefusal(
            "A saved [fetch].cycle must be a concrete UTC cycle; 'latest' "
            "would leave the download free to disagree with the saved "
            "[experiment].start_time. Regenerate the config with "
            "`gpuwm domain --cycle latest` and the same domain options, "
            "which resolves the cycle and experiment start together.")
    cycle = parse_cycle(raw, str(fetch_table["source"]))
    if isinstance(start_time, datetime) and start_time.tzinfo is None:
        lead = fetch_table.get("forecast_start_hour", 0)
        if isinstance(lead, bool) or not isinstance(lead, int) or lead < 0:
            raise GoRefusal("[fetch].forecast_start_hour must be a nonnegative integer.")
        expected = cycle + timedelta(hours=lead)
        if start_time != expected:
            raise GoRefusal(
                f"[fetch].cycle plus forecast_start_hour starts at "
                f"{expected:%Y-%m-%dT%H:%M:%S} UTC, but "
                f"[experiment].start_time is {start_time:%Y-%m-%dT%H:%M:%S} UTC. "
                "Regenerate the config with the intended --cycle and "
                "--forecast-start-hour so the download and experiment agree.")
    return cycle


def fetch_request(fetch_table, *, p_top=None) -> dict:
    """The download a ``[fetch]`` table makes for a run whose top is ``p_top``.

    The table as written, plus ``p_top_pa`` when the source's certified
    ladder stops below the run's own model top and its fetch can reach
    it (:func:`gpuwm.source_adapters.fetch_model_top_pa`, read from the
    registry row).  Without it a GFS run with a 50 hPa top fetched the
    100 hPa ladder and was refused at preparation for the two levels the
    download never asked for.  The key is ``gpuwm fetch``'s own flag
    name, so every consumer that spells a request as fetch flags --
    the fetch stage, the managed download folder's identity and the
    download price -- carries it without a second spelling.  A table
    that already states a top or asks for every level is left as it is.
    """

    from gpuwm.source_adapters import fetch_model_top_pa

    request = dict(fetch_table or {})
    if request.get("p_top_pa") is not None or request.get("all_levels"):
        return request
    top = fetch_model_top_pa(request.get("source"), p_top)
    if top is not None:
        request["p_top_pa"] = top
    return request


#: Which statement named the host a run's fetch stage pins, as its plan
#: records it.
TRANSPORT_FROM_FLAG = "--transport"
TRANSPORT_FROM_TABLE = "[fetch] transport"


def pinned_transport(fetch_table, flag: str | None = None
                     ) -> tuple[str | None, str | None]:
    """The host a run's fetch stage pins, and which statement named it.

    ``gpuwm go --transport`` wins over the config's ``[fetch] transport``:
    the flag is the later and more specific statement.  A flag the
    config's source cannot pin is refused here, in the fetch's own words,
    because the fetch stage would refuse it only after the chain started.
    ``(None, None)`` when there is no flag and the table names no host, so
    the fetch walks the source's ladder.

    ``auto`` names no host (:func:`gpuwm.fetch.pinned_host`): it is the
    ladder written out, so it pins nothing, and a flag saying it over a
    table that pins a host unpins the run.  Read as a host, a config that
    spelled out the default was announced as pinning host "auto" and its
    download was keyed apart from the identical unpinned request, so the
    cycle was fetched again into a second folder.  Callers therefore drop
    a request's own ``transport`` whenever this returns no pin.

    A flag of ``auto`` still decided the host: it returns
    ``(None, "--transport")``, so :func:`transport_note` can say that the
    flag, not the table, is why the fetch walks the ladder.
    """

    from gpuwm.fetch import pinned_host, transport_refusal
    from gpuwm import fetch_endpoints

    table = fetch_table if isinstance(fetch_table, dict) else {}
    if fetch_endpoints.policy_uses_aws(str(table.get("source", ""))):
        return (fetch_endpoints.policy_transport(str(table["source"])),
                fetch_endpoints.FETCH_POLICY_ENV)
    if flag is None:
        value = table.get("transport")
        host = None if value is None else pinned_host(str(value))
        return (None, None) if host is None else (host, TRANSPORT_FROM_TABLE)
    refusal = transport_refusal(str(table.get("source", "")), flag)
    if refusal is not None:
        raise GoRefusal(f"--transport {flag}: {refusal}")
    return pinned_host(flag), TRANSPORT_FROM_FLAG


def pin_request(request: dict, pinned: str | None) -> dict:
    """``request`` asking the host ``pinned``, or no host when it is None.

    The request's own ``transport`` (the table's copy) is replaced either
    way: a flag of ``auto`` over a pinned table walks the ladder, and a
    table spelling ``auto`` keys the same download as one saying nothing.
    """

    pinned_request = {key: value for key, value in request.items()
                      if key != "transport"}
    if pinned is not None:
        pinned_request["transport"] = pinned
    return pinned_request


def transport_note(transport: str | None, basis: str | None,
                   table_value=None) -> str | None:
    """The one line a plan prints about its download host, or None.

    A pinned host is always named.  An unpinned fetch is named only when
    ``--transport auto`` overrode a table that pins a host: that run asks
    other hosts than its config says, and a silent plan left the reader
    believing the table's host was asked.
    """

    from gpuwm.fetch import pinned_host

    if transport is None:
        if (basis != TRANSPORT_FROM_FLAG or table_value is None
                or pinned_host(str(table_value)) is None):
            return None
        return ("go: the fetch walks the host ladder, from --transport auto, "
                f"over [fetch] transport = {str(table_value)!r}")
    note = f"go: the fetch pins host {transport}, from {basis}"
    if basis == TRANSPORT_FROM_FLAG and table_value is not None:
        note += f" (over [fetch] transport = {str(table_value)!r})"
    return note


def config_fetch_request(payload) -> dict:
    """:func:`fetch_request` for a whole configuration's tables."""

    shared = payload.get("shared") if isinstance(payload, dict) else None
    return fetch_request(
        (payload or {}).get("fetch"),
        p_top=shared.get("p_top") if isinstance(shared, dict) else None)


def _managed_download_request(fetch_table: dict):
    """The existing normalized acquisition identity, independent of output path."""
    import hashlib

    from types import SimpleNamespace
    from gpuwm import fetch, source_adapters

    request = {key: value for key, value in fetch_table.items() if key != "out"}
    # ``auto`` is the unpinned request written out, so it keys the folder
    # the unpinned request fills rather than a second copy of the cycle.
    if "transport" in request and fetch.pinned_host(request["transport"]) is None:
        del request["transport"]
    try:
        source = source_adapters.get_source_adapter(str(request["source"])).source_id
        cycle = checked_config_fetch_cycle(request | {"source": source})
        radius = request.get("radius_km")
        area = fetch._resolve_area(SimpleNamespace(
            area=request.get("area"), point=request.get("point"),
            radius_km=None if radius is None else float(radius)))
    except (KeyError, TypeError, ValueError) as error:
        raise GoRefusal(f"The forecast download settings are invalid: {error}") from error
    request["source"] = source
    request["cycle"] = cycle.strftime("%Y-%m-%dT%H:%M:%SZ")
    request["area"] = None if area is None else area.as_manifest()
    request.pop("point", None)
    request.pop("radius_km", None)
    try:
        canonical = json.dumps(request, sort_keys=True, separators=(",", ":"),
                               allow_nan=False)
    except (TypeError, ValueError) as error:
        raise GoRefusal(f"The forecast download settings are invalid: {error}") from error
    key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return request, source, cycle, area, key


def managed_download_key(fetch_table: dict) -> str:
    """Use the native managed cache's complete canonical request key elsewhere."""
    return _managed_download_request(fetch_table)[-1]


def managed_download_dir(case_root: Path, fetch_table: dict) -> Path:
    """Select a reusable cache for this request without changing existing data.

    Quick Forecast can launch different cycles and areas into one workspace.
    Sharing one flat download directory made the second request hit fetch's
    correct wrong-input refusal. Keep requests separate here; the downloader
    still locks its selected directory and verifies every reused payload.
    """
    from gpuwm import fetch, fetch_guard, fetch_routes
    request, source, cycle, area, key = _managed_download_request(fetch_table)
    # A separate namespace preserves the flat data directory created by older
    # versions, including a prior request that the user may still need.
    cache_root = Path(case_root) / "downloads"
    attempt = 0
    while True:
        name = key if attempt == 0 else f"{key}-{attempt}"
        candidate = cache_root / name
        if not candidate.is_symlink():
            if not candidate.exists():
                return candidate
            if candidate.is_dir():
                try:
                    if fetch_guard.active_writer("fetch-out", candidate):
                        return candidate
                    if not any(candidate.iterdir()):
                        return candidate
                    if (source in fetch_routes.route_ids()
                            and fetch_routes.has_recovery_request(candidate)):
                        fetch_routes.check_prior_request(
                            candidate, source=source, cycle=cycle,
                            host=request.get("transport"), member=request.get("member"))
                        return candidate
                    manifest = candidate / fetch.FETCH_MANIFEST_NAME
                    if manifest.is_file():
                        payload = json.loads(manifest.read_text(encoding="utf-8"))
                        table_route = source in fetch_routes.route_ids()
                        schema = (fetch_routes.ROUTE_MANIFEST_SCHEMA if table_route
                                  else fetch.FETCH_MANIFEST_SCHEMA)
                        if not isinstance(payload, dict) or payload.get("schema") != schema:
                            raise ValueError("The cache has no recognized fetch receipt")
                        if table_route:
                            if not isinstance(payload.get("request"), dict):
                                raise ValueError("The cache has no recorded request")
                            fetch_routes.check_prior_request(
                                candidate, source=source, cycle=cycle,
                                host=request.get("transport"), member=request.get("member"))
                        else:
                            fetch.check_prior_request(candidate, source=source,
                                                      cycle=cycle, area=area)
                        return candidate
                except (OSError, ValueError):
                    pass
        # A cache without either a complete or recovery receipt cannot
        # establish input identity. Preserve it and select another slot.
        attempt += 1


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

    :data:`gpuwm.progress.PREP_EVENT_PARENT_ENV` tells a stage that hosts
    a preparer program (``gpuwm.source_cli`` on the GFS chain) that this
    process reads step records off its output (:func:`_read_stage_output`),
    so it passes each one on instead of keeping it to its own log.
    """

    from gpuwm.progress import PREP_EVENT_PARENT_ENV

    return _with_git_handle({**os.environ, "PYTHONSAFEPATH": "1",
                             PREP_EVENT_PARENT_ENV: "1"})


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
                     render_section: str | None = None,
                     run_stamp: bool = run_stamp_module.DEFAULT_RUN_STAMP,
                     claim: bool = False,
                     transport: str | None = None,
                     devices: int | None = None,
                     as_posted: bool | None = None,
                     late_after_minutes: float | None = None) -> dict:
    """Everything the five stages need, or a refusal saying why not.

    ``transport`` is ``gpuwm go --transport``: it wins over the config's
    ``[fetch] transport``, and the plan records which one it took.

    ``render_section`` is the line every ``xsec:`` product of this run
    is cut along (``gpuwm render --section``), carried to all three of
    its renders: the one the forecast draws as each frame lands, its
    early first frame, and the end-of-run batch.

    ``run_stamp`` puts this run's whole tree -- authority, prepared,
    run, png -- in its own timestamped folder under the output
    directory, which is the default and the reason two runs of one
    config no longer collide. Downloads use a request-specific cache shared
    across matching runs; changing the cycle or area selects another cache
    automatically without replacing the previous inputs.

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
        # The remedy is spelled by the shared shell rule, and `gpuwm run`
        # starts only when `gpuwm check` passed: a literal `&&` here was a
        # Windows PowerShell 5.1 parser error.  The path is quoted for
        # that same shell, because a config under a folder with a space
        # split into two arguments in either shell.
        from gpuwm.bridges import run_if_first_succeeds, shell_line
        raise GoRefusal(
            f"{config} declares a [case_data] table, which is the ERA5 "
            "config-driven route -- `gpuwm run` executes that one "
            "directly and needs no chain.\n"
            "  remedy: " + run_if_first_succeeds(
                shell_line("gpuwm", "check", config),
                shell_line("gpuwm", "run", config)))

    domains = payload.get("domain")
    domain_count = len(domains) if isinstance(domains, list) else 1
    # A DOMAIN TREE IS NOT REFUSED HERE.  It used to be, on the stated
    # premise that "the single-domain runner this chain drives takes
    # one" -- but the chain has not driven only that runner for some
    # time: `runner` below is already TREE_RUNNER_MODULE when
    # ``domain_count > 1``, :func:`_run_forecast` already composes
    # :func:`tree_forecast_command` on that arm, and :func:`go_main`
    # already skips the three-digest relay a tree has no single
    # prepared-cache identity for.  The refusal's remedy was "re-emit
    # the config without --ladder", i.e. throw away the nests the
    # README's first paragraph sells, to reach a runner the very next
    # statement of this function was willing to select.
    #
    # This stage composer accepts only its declared decoder contract.
    # go_main dispatches every other native chain before reaching it.

    fetch_table = payload.get("fetch")
    if not isinstance(fetch_table, dict) or "source" not in fetch_table:
        raise GoRefusal(
            f"{config} carries no [fetch] table, so the cycle and area "
            "this domain was sized against are not recorded in it.\n"
            "  remedy: re-emit the config with `gpuwm domain`, which "
            "writes that table")

    from gpuwm.runplan import prepared_chain_for_source

    source = str(fetch_table["source"])
    if prepared_chain_for_source(source) != "prepared:go":
        raise GoRefusal(
            f"The rw-wps stage composer cannot prepare --source {source}. "
            f"Next: gpuwm go {_quote(config)}")

    try:
        experiment = load_experiment(config)
        if len(experiment.domains) == 1:
            from gpuwm.experiment import refuse_unrouted_perturbation
            refuse_unrouted_perturbation(experiment, "single-domain prepared forecast")
        profile = identify_single_domain_profile(experiment.root.run)
    except Exception as error:  # a config that will not load is the finding
        raise GoRefusal(f"{config} does not load as an experiment: "
                        f"{error}") from error
    # WHAT THIS MACHINE CANNOT PREPARE IS REFUSED HERE, on the near side
    # of the fetch.  Every precondition in the inventory is a question
    # about the INSTALL -- is a 225 MB dataset staged here -- answerable
    # the moment the experiment loads.  The floor that raises the same
    # sentence, ``initialize_real``, sits in the PREPARE stage below:
    # after the authority stage, after the whole cycle has been
    # downloaded, after the manifest of those bytes has been verified.
    # Asking there and only there is the "wizard says PASS, fetch 10-15
    # GB, then refuse" shape, and it is the shape the route table this
    # precondition replaced was written against.
    from gpuwm.config import experiment_preparation_refusals

    unmet = experiment_preparation_refusals(experiment)
    if unmet:
        raise GoRefusal("\n".join(
            f"{label}: {sentence}" for label, sentence in unmet))
    # A root the source's grid does not reach is refused here too, from
    # the source row's declared coverage, rather than at the prepare
    # stage after the whole cycle is downloaded.
    from gpuwm.source_coverage import config_source_coverage_refusal

    uncovered = config_source_coverage_refusal(experiment, source)
    if uncovered is not None:
        raise GoRefusal(uncovered)
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
    # does not carry, and the accurate answer was "go clone the
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
    from gpuwm.core.devices import override_device_count, refuse_unrouted_devices
    from dataclasses import replace
    experiment = replace(experiment, devices=override_device_count(
        experiment.devices, devices)) if devices is not None else experiment
    if len(experiment.domains) != 1:
        # A split tree runs through the tree runner (--devices relayed by
        # tree_forecast_command, the table riding the hash-bound config);
        # what that runner cannot split is refused here by name, before
        # the download: moving nests, late grids, a split parent over a
        # resident nest.
        from gpuwm.core.devices import validate_tree_devices
        validate_tree_devices(experiment)
    elif getattr(experiment.relocation, "enabled", False):
        refuse_unrouted_devices(experiment, "gpuwm go moving nests")
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
    checked_config_fetch_cycle(fetch_table, start_time=experiment.start_time)
    # What the fetch stage asks for is the [fetch] table PLUS the model
    # top this config's own ladder needs.  The table records the cycle,
    # hours and area the domain was sized against; the top is recorded
    # once, in [shared].p_top, and read from there rather than asked
    # for a second time.
    pinned, pinned_from = pinned_transport(fetch_table, transport)
    # The download cache is keyed on the host the fetch will ask.
    request = pin_request(
        fetch_request(fetch_table, p_top=experiment.vertical.p_top), pinned)

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
    # and the values that are essential are the ones the domain was
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
    data = (Path(data_dir) if data_dir is not None
            else managed_download_dir(case_root, request))
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
                "download use an automatically managed request cache.")
    # AFTER every refusal above: a plan that is going to be refused must
    # not leave a directory behind on the way out.
    #
    # ONE launch instant, taken here and carried in the plan, so the
    # folder this plan names and the folder claim_run_root makes are the
    # same name.  The claim used to read the clock again after the
    # memory and geography gates, which take over a second, so every
    # real run announced a run-...Z folder that was never created and
    # then a second one a few seconds later.
    launch = run_stamp_module.utcnow()
    root = run_stamp_module.resolve(
        case_root, init=fetch_table["cycle"], launch=launch,
        enabled=run_stamp, create=claim)
    plan = {
        "config": config,
        "wps_namelist": base / f"{config.stem}.namelist.wps",
        "source": source,
        "cycle": str(fetch_table["cycle"]),
        "hours": int(fetch_table["hours"]),
        # Essential exactly like cycle/hours/area: the config's
        # start_time IS cycle + this lead, so a fetch that ignored it
        # would download a window the front door then refuses for not
        # carrying the lead the experiment starts from.
        "forecast_start_hour": int(fetch_table.get("forecast_start_hour", 0)),
        # Essential for the same reason, and dropped until 1.4.1.  It
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
        "era5_provider": fetch_table.get("era5_provider"),
        "era5_product": fetch_table.get("era5_product"),
        "member": fetch_table.get("member"),
        "retrieve": fetch_table.get("retrieve", False),
        # Essential too: the model top the download must reach, from
        # this config's own [shared].p_top.  None when the source's
        # certified ladder already reaches it, which leaves the request
        # byte for byte what it always was.
        "p_top_pa": request.get("p_top_pa"),
        "all_levels": bool(request.get("all_levels")),
        # The one host the fetch pins, None to walk the source's ladder,
        # and which statement named it: --transport wins over the table.
        "transport": pinned,
        "transport_from": pinned_from,
        "transport_table": fetch_table.get("transport"),
        # How the fetch stage waits for its source (DESIGN A136 3.2): the
        # flag over the table, None for the default (as posted, the
        # source row's budget).
        "as_posted": (as_posted if as_posted is not None
                      else fetch_table.get("as_posted")),
        "late_after_minutes": (
            None if (as_posted if as_posted is not None
                     else fetch_table.get("as_posted")) is False
            else late_after_minutes if late_after_minutes is not None
            else fetch_table.get("late_after_minutes")),
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
        # The instant the stamp above was named from; the claim reuses it.
        "launch": launch,
        "authority": Path(root) / "authority",
        "prepared": Path(root) / "prepared",
        "run": Path(root) / "run",
        # Set by a caller that wants a subset; `gpuwm go` itself never
        # sets it, so its render stage is unchanged.
        "render_products": render_products,
        # The line the section products are cut along; None draws no
        # section, and every render command stays what it was.
        "render_section": render_section,
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
    if getattr(getattr(experiment, "devices", None), "enabled", False) or devices is not None:
        from gpuwm.core.devices import describe_split
        options = experiment.devices
        plan["devices"] = options.to_mapping()
        plan["devices_sentence"] = describe_split(experiment, options)
        if devices is not None:
            plan["devices_count_override"] = devices
    return plan



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
    # The plan's own launch instant, not the clock now: the stamp names
    # when this run was launched, and the gates between naming and
    # claiming must not move it.  Only a folder of that exact name made
    # in the meantime (another chain launched in the same second) bumps
    # the ordinal, which is the one case the caller's correction line is
    # for.
    claimed = run_stamp_module.allocate(plan["case_root"],
                                        init=plan.get("cycle"),
                                        launch=plan.get("launch"))
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
    for key in ("era5_provider", "era5_product", "member"):
        if plan.get(key) is not None:
            command.extend(("--" + key.replace("_", "-"), str(plan[key])))
    if plan.get("retrieve"):
        command.append("--retrieve")
    if plan.get("p_top_pa") is not None:
        command.extend(("--p-top-pa", f"{float(plan['p_top_pa']):g}"))
    if plan.get("all_levels"):
        command.append("--all-levels")
    if plan.get("transport") is not None:
        command.extend(("--transport", str(plan["transport"])))
    from gpuwm import fetch_endpoints
    from gpuwm.fetch import GFS_CONTAINER_SOURCES
    if (fetch_endpoints.policy_uses_aws(plan["source"])
            and plan["source"] in GFS_CONTAINER_SOURCES):
        command.extend(("--mode", "full-file"))
    if plan.get("as_posted") is not None:
        command.append("--as-posted" if plan["as_posted"] else "--whole-cycle")
    if plan.get("late_after_minutes") is not None:
        command.extend(("--late-after-minutes",
                        f"{float(plan['late_after_minutes']):g}"))
    return command


def front_door_manifest(plan: dict) -> Path:
    """This run's own GFS front-door manifest, beside its preparation.

    Not ``<data>/gfs-input-manifest.json``: the download is shared by
    every run of the case (and by any run given the same --data-dir),
    while the manifest binds this run's own namelist and experiment, so
    one shared file let a second run replace a first run's binding while
    the first was still preparing.  Read after :func:`claim_run_root`,
    which may move the run into a neighbouring folder.
    """

    from gpuwm.fetch import preparation_manifest_path

    return preparation_manifest_path(Path(plan["prepared"]))


def manifest_command(plan: dict, bridge: Path) -> list[str]:
    return [sys.executable, "-m", "gpuwm.cli", "fetch",
            "--source", plan["source"], "--author-front-door-manifest",
            "--out", str(plan["data"]), "--bridge", str(bridge),
            "--wps-namelist", str(plan["authority"] / "namelist.wps"),
            "--experiment-config", str(plan["authority"] / "experiment.toml"),
            "--manifest-out", str(front_door_manifest(plan))]


def prepare_command(plan: dict, bridge: Path, *, manifest: Path | None,
                    manifest_sha256: str | None, cycle_stamp: str,
                    geog_root: Path,
                    as_posted: Path | None = None) -> list[str]:
    """The rw-wps invocation, composed from the artifacts on disk.

    Every value here is either from the plan or read back from a file a
    previous stage wrote -- the manifest's path and its digest.  This is
    the same command ``gpuwm fetch --author-front-door-manifest``
    prints; composing it from the same inputs rather than parsing that
    printed line is what keeps it working when the printed line moves
    behind ``--explain``.

    ``as_posted`` is the posting folder of a fetch running beside this
    preparation (DESIGN A136 2.5): the preparation then waits for each
    lead's marker there and its seal writes the manifest, so it is given
    no manifest to bind.
    """

    preprocessing = []
    if plan["source"] == "gfs":
        import tomllib
        from gpuwm.config_authority import read_config_authority
        from gpuwm.preprocess_policy import preprocess_backend_choice
        tables = tomllib.loads(read_config_authority(plan["config"]).payload.decode("utf-8-sig"))
        backend, reason = preprocess_backend_choice(source="gfs", tables=tables)
        if backend == "cpu":
            # The reason rides with the choice so the preparation receipt
            # names the declaration that moved preparation off the card.
            preprocessing = ["--preprocess-backend", "cpu",
                             *(("--preprocess-backend-reason", reason)
                               if reason is not None else ())]

    return [sys.executable, "-m", "gpuwm.source_cli",
            "--source", plan["source"],
            "--gfs-series", str(plan["data"] / f"{plan['source']}-series.tsv"),
            "--cycle", cycle_stamp,
            "--bridge", str(bridge),
            "--wps-namelist", str(plan["authority"] / "namelist.wps"),
            "--experiment-config",
            str(plan["authority"] / "experiment.toml"),
            *(["--as-posted", str(as_posted)] if as_posted is not None
              else ["--source-manifest", str(manifest),
                    "--source-manifest-sha256", manifest_sha256]),
            # Forwarded only when the config is bound to one: the GFS
            # front door no longer substitutes WSM6 for an absent
            # profile (the defect that once made this flag mandatory in
            # practice) -- an unnamed config's own suite is prepared as
            # written and its verification status reported.
            *_profile_flags(plan),
            "--geog-root", str(geog_root),
            *preprocessing,
            # The config declared a follow source, so the bundle must
            # carry the sealed statics corridor or stage 5 refuses it.
            *(["--statics-corridor"] if plan.get("statics_corridor")
              else []),
            "--output-root", str(plan["prepared"])]


def announce_policy_backend(command: list[str]) -> None:
    """One line when the configuration, not the reader, moved preparation to the CPU.

    ``auto`` announces its own fallbacks inside the preparation; a policy
    choice arrives there as an explicit ``cpu`` and would otherwise run
    without a word in the go log.
    """

    if "--preprocess-backend-reason" in command:
        reason = command[command.index("--preprocess-backend-reason") + 1]
        print(f"go: preparation runs on the CPU backend: {reason}",
              flush=True)


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
    return (DEFAULT_RENDER_PRODUCTS if products is None
            else str(products))


def _early_render_flags(plan: dict, early_render: str | None) -> list[str]:
    """The runner's render flags: products, directory and section line.

    Both runners arm their renders off these, the every-frame render and
    its early first frame, so the line a section is cut along reaches
    them here or not at all.  Joined with ``=``: a line in the southern
    or western hemisphere starts with a minus sign, and a separate token
    that does is read as an option by the runner's parser.
    """

    if early_render is None:
        return []
    flags = ["--render-products", str(early_render),
             "--render-dir", str(plan["render"])]
    section = plan.get("render_section")
    if section:
        flags.append(f"--render-section={section}")
    return flags


def forecast_command(plan: dict, digests: dict, *,
                     early_render: str | None = None) -> list[str]:
    # A chained preparation binds its HEAD (the forecast starts before the
    # later boundary intervals exist); a sealed one binds proof and cache.
    binding = (["--prepared-head-sha256", digests["prepared_head"]]
               if "prepared_head" in digests else
               ["--proof-sha256", digests["proof"],
                "--source-manifest-sha256", digests["source_manifest"],
                "--prepared-content-sha256", digests["prepared_content"]])
    # An as-posted head names no manifest: its seal writes one, held to
    # the input plan the head digest binds (head_digests).
    if "prepared_head" in digests and digests.get("source_manifest"):
        binding[2:2] = ["--source-manifest-sha256",
                        digests["source_manifest"]]
    return [sys.executable, "-m", str(plan["runner"]),
            "--source", plan["source"],
            "--prepared-root", str(plan["prepared"]),
            *binding,
            "--experiment-config",
            str(plan["authority"] / "experiment.toml"),
            "--wps-namelist", str(plan["authority"] / "namelist.wps"),
            *_profile_flags(plan),
            *(["--devices", str(plan["devices_count_override"])]
              if "devices_count_override" in plan else []),
            *_PROGRESS_FLAGS,
            # Same output directory the finalize stage renders into, so
            # the early picture and the late one land in one tree under
            # one layout rather than two.
            *_early_render_flags(plan, early_render),
            "--io-mode", "history", "--outdir", str(plan["run"])]


def tree_forecast_command(plan: dict, *,
                          digests: dict | None = None,
                          early_render: str | None = None,
                          prepared_head_sha256: str | None = None
                          ) -> list[str]:
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

    ``digests`` exists for ``--dry-run`` ONLY, which prints the chain
    before any stage has written anything: neither file is on disk
    then, so the caller supplies the two placeholder strings that name
    the file each value will be read from.  Left ``None`` -- every real
    run -- the two are read off disk exactly as before, so this
    function's output on the live path is byte-identical.

    ``prepared_head_sha256`` binds a chained tree at its head instead of
    its sealed document (``--prepared-head-sha256``): the runner restores
    from the head and binds the seal at the end.
    """

    from gpuwm.fetch import sha256_file

    prepared = Path(plan["prepared"])
    config = Path(plan["authority"]) / "experiment.toml"
    if prepared_head_sha256 is not None:
        if not config.is_file():
            raise GoRefusal(
                f"the authority stage wrote no {config}, so the tree "
                "forecast has no experiment config to bind")
        binding = ["--prepared-head-sha256", str(prepared_head_sha256)]
        config_digest = sha256_file(config)
    elif digests is None:
        receipt = _hierarchy_document(prepared)
        if not config.is_file():
            raise GoRefusal(
                f"the authority stage wrote no {config}, so the tree "
                "forecast has no experiment config to bind")
        binding = ["--preparation-receipt-sha256", sha256_file(receipt)]
        config_digest = sha256_file(config)
    else:
        binding = ["--preparation-receipt-sha256",
                   digests["preparation_receipt"]]
        config_digest = digests["experiment_config"]
    return [sys.executable, "-m", str(plan["runner"]),
            "--prepared-root", str(prepared),
            *binding,
            "--experiment-config", str(config),
            "--experiment-config-sha256", config_digest,
            *_profile_flags(plan),
            # `gpuwm go TREE --devices N`: the count reaches the tree runner
            # as it reaches the single-domain one (forecast_command).
            *(["--devices", str(plan["devices_count_override"])]
              if "devices_count_override" in plan else []),
            *_PROGRESS_FLAGS,
            # The tree draws as it goes too, every grid of it, into the
            # directory the finalize stage renders into: the runner arms
            # its own renders off these two flags, exactly as the
            # single-domain runner does (``forecast_command``).  Without
            # them a nested `gpuwm go` drew nothing until the forecast
            # ended.
            *_early_render_flags(plan, early_render),
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


@contextlib.contextmanager
def _checkpoint_retention(keep: int | None):
    """Hand the forecast ``keep`` as its checkpoint retention, then put back what was there.

    The forecast's checkpoint writer reads the number of sets to keep from
    :data:`gpuwm.resume.KEEP_CHECKPOINTS_ENV` (0 keeps every set), in this
    process when it is hosted here and in the stage subprocess, which
    inherits this environment, when it is not.  ``None`` leaves whatever
    the caller set, which is how a hosting ``gpuwm run-plan`` keeps its own
    policy.
    """
    from gpuwm.resume import KEEP_CHECKPOINTS_ENV

    if keep is None:
        yield
        return
    previous = os.environ.get(KEEP_CHECKPOINTS_ENV)
    os.environ[KEEP_CHECKPOINTS_ENV] = str(int(keep))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(KEEP_CHECKPOINTS_ENV, None)
        else:
            os.environ[KEEP_CHECKPOINTS_ENV] = previous


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

    The subprocess arm stays the default and carries a watchdog, because
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
    early = None if hosted else _early_render_products(plan)
    command = (tree_forecast_command(
                   plan, early_render=early,
                   prepared_head_sha256=digests.get("prepared_head"))
               if plan.get("domains", 1) > 1
               else forecast_command(plan, digests, early_render=early))
    progress = plan["run"] / "progress.json"
    if plan.get("domains", 1) > 1:
        progress = plan["run"] / "evidence" / "progress.json"
    if not hosted:
        if early is None:
            _run_stage("forecast", command, explain=explain, progress=progress,
                       observer=observer)
            return
        # THIS STAGE DRAWS, so it owns working stores exactly as the
        # render stage does and is swept on the same terms.  The early
        # render runs inside the forecast subprocess and writes into
        # ``plan["render"]``, so a forecast that dies mid-frame leaves
        # tiled hour files beside a delivery that published nothing --
        # and a sweep can only reach them if this stage's stores carry a
        # token this door minted, because the plain prefix matches a
        # concurrent render's live store too.
        from gpuwm.render import SCRATCH_PREFIX_ENV, stage_scratch_prefix

        stage_prefix = stage_scratch_prefix()
        try:
            _run_stage("forecast", command, explain=explain, progress=progress,
                       observer=observer,
                       env={SCRATCH_PREFIX_ENV: stage_prefix})
        except GoStageFailed:
            _close_stage_scratch(plan, prefix=stage_prefix, stage="forecast",
                                 observer=observer, failed=True)
            raise
        # A passing forecast stage has exited too, so what its early
        # render could not remove goes now rather than never.
        _close_stage_scratch(plan, prefix=stage_prefix, stage="forecast",
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


def head_digests(prepared_root: Path, head_sha256: str) -> dict:
    """The two digests a forecast bound to a prepared head carries.

    Read from ``boundary-stream/head.json``, which carries its own digest
    and is checked against it on the read: the same relay rule as
    :func:`proof_digests`, for a preparation whose proof does not exist
    yet.  An as-posted head carries one: it binds the input plan, and the
    manifest does not exist until its seal (``source_manifest`` None).
    """

    from gpuwm.ingest.boundary_stream import BoundaryStreamError, bind_head

    try:
        head = bind_head(prepared_root, head_sha256, require_manifest=True)
    except BoundaryStreamError as error:
        raise GoRefusal(str(error)) from None
    return {"prepared_head": str(head["head_sha256"]),
            "source_manifest": head["basis"].get("input_manifest_sha256")}


def wrfout_frames(plan: dict) -> list[Path]:
    """Every history file the forecast stage published, in time order."""

    from gpuwm.io.wrfout import iter_wrfout_files

    root = Path(plan.get("wrfout_dir", plan["run"] / "wrfout"))
    return sorted(iter_wrfout_files(root, WRFOUT_GLOB,
                                   include_temporaries=False),
                  key=lambda path: (path.name, path.as_posix()))


def render_command(plan: dict, frames: list[Path] | None = None, *,
                   context_frames: list[Path] | None = None,
                   inputs_file: Path | None = None) -> list[str]:
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

    That glob is :data:`WRFOUT_GLOB` -- the FRAMES, not everything in
    the directory.  It used to be ``*``, which is the same set only on
    a route that keeps its frames in a ``wrfout/`` subdirectory of its
    own.  A downscaled child writes them in the run root beside
    ``events.jsonl``, ``report.json``, its checkpoints and its picture
    tree, so the remedy line this function prints fed all of those to
    the renderer and stopped at "NetCDF: Unknown file format".

    And when frames are ALREADY on disk, ``frames=None`` names them
    rather than the pattern, because the line this composes is pasted
    into a shell: :func:`printable` quotes any token holding a ``*``,
    so the pattern reaches the renderer's argv literally on a shell
    that would have expanded it -- and PowerShell, where a reader on
    this platform pastes it, expands nothing for a native command at
    all.  The pattern is printed when the directory holds no frame yet,
    which is the ``--dry-run`` case and the only one where naming its
    contents would be a guess.

    The run folder is claimed ONCE, by the chain, and this stage draws
    into it: :func:`gpuwm.run_stamp.stage_flags` is what says so.  This
    command is composed twice for one run -- once by
    :mod:`gpuwm.first_products` on the first committed frame and once by
    :func:`_render_stage` at the end -- so a render that stamped for
    itself would put those two halves of one run's pictures in two
    wall-clock folders, and the finalize stage could no longer prove the
    early frame had already been drawn.

    ``inputs_file`` is for a command this process RUNS whose frames do
    not fit on a command line (:func:`run_render_pass` decides): the
    frames and their context are written into that file and the command
    names the file (``--inputs-from``).  One path per frame on the
    command line is how a 48 h nested forecast asked Windows for a 36,519
    character command, 32,767 being the most it starts, and never drew a
    picture.  A printed line keeps the frames spelled out, because the
    reader pasting it has no such file.
    """

    if frames is None:
        frames = wrfout_frames(plan) or None
    listed = frames is not None and inputs_file is not None
    if listed:
        from gpuwm.render import write_render_inputs

        write_render_inputs(inputs_file, frames, context_frames or ())
        targets = ["--inputs-from", str(inputs_file)]
    else:
        targets = ([str(frame) for frame in frames] if frames is not None
                   else [str(Path(plan.get("wrfout_dir",
                                           plan["run"] / "wrfout"))
                             / WRFOUT_GLOB)])
    command = [sys.executable, "-m", "gpuwm.cli", "render", *targets, "--series",
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
    # The line every `xsec:` product is cut along, `gpuwm render
    # --section`'s own value.  Without it the render front door drops each
    # section product before the renderer starts, so a run that asked for
    # one drew none.  Joined with `=` because a southern or western line
    # starts with a minus sign.
    section = plan.get("render_section")
    if section:
        command.append(f"--section={section}")
    if not listed:
        for frame in context_frames or ():
            command += ["--context-wrfout", str(frame)]
    return command


#: The names of the render stage's frame files (:func:`render_inputs_file`)
#: in the system temporary folder: ``<prefix><random><suffix>``.
RENDER_INPUTS_PREFIX = "gpuwm-render-inputs-"
RENDER_INPUTS_SUFFIX = ".json"

#: How long a failed stage's frame file is kept for the "run ... --explain"
#: line its refusal printed: a week, so a person who comes back to a
#: failed forecast after a weekend can still paste it.  The next long
#: render stage removes the older ones, so the temporary folder holds the
#: files of the last week's failed long stages and no more (one file each
#: used to build up for good).
RENDER_INPUTS_KEEP_SECONDS = 7 * 24 * 3600


def sweep_kept_render_inputs(folder=None, *, now: float | None = None) -> list[Path]:
    """Remove frame files failed stages kept longer than a week; the removed.

    Only a file named as :func:`render_inputs_file` names one and last
    written more than :data:`RENDER_INPUTS_KEEP_SECONDS` ago: a running
    stage's file was written when that stage started, and the renderer
    reads it as it starts.  Never raises: a file another account owns, or
    one that is gone by the time it is reached, is left to its owner.
    """

    folder = Path(tempfile.gettempdir() if folder is None else folder)
    cutoff = (time.time() if now is None else now) - RENDER_INPUTS_KEEP_SECONDS
    removed: list[Path] = []
    try:
        candidates = sorted(folder.glob(
            f"{RENDER_INPUTS_PREFIX}*{RENDER_INPUTS_SUFFIX}"))
    except OSError:
        return removed
    for path in candidates:
        try:
            if not path.is_file() or path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
        except OSError:
            continue
        removed.append(path)
    return removed


@contextlib.contextmanager
def render_inputs_file():
    """A private file for one render stage's frame lists, removed after.

    In the system temporary folder rather than the run folder: the run
    folder is what a person opens, and a stage that dies leaves nothing
    there that is not the forecast's own.

    Kept when the stage ran and failed.  Its refusal's "run ... --explain"
    line names this file rather than spelling out a series too long for
    any shell (:func:`gpuwm.render._respell_invocation`), and that line
    must run as printed.  A stage that passed, was stopped, or never
    started printed no such line, and the file goes.  A kept file goes a
    week later, when a later long stage opens its own
    (:func:`sweep_kept_render_inputs`).
    """

    sweep_kept_render_inputs()
    try:
        handle, name = tempfile.mkstemp(prefix=RENDER_INPUTS_PREFIX,
                                        suffix=RENDER_INPUTS_SUFFIX)
        os.close(handle)
    except OSError:
        # No writable temporary folder: the frames go on the command
        # line as they always did, which a short series fits, and a
        # series too long for it fails as a stage that could not start.
        yield None
        return
    path = Path(name)
    kept = False
    try:
        yield path
    except GoStageFailed as failure:
        kept = failure.code != _STAGE_START_FAILED
        raise
    finally:
        if not kept:
            try:
                path.unlink()
            except OSError:
                pass


def run_render_pass(plan: dict, frames: list[Path], *,
                    context_frames: list[Path] | None = None,
                    **stage) -> None:
    """Run one render stage over ``frames``.

    A command that fits the command-line budget runs exactly as it always
    did, every frame on it.  One that does not (a long nested forecast
    under an ordinary Documents folder) hands its frames to the process
    in a private file (:func:`render_inputs_file`); everything that
    records or shows that command gets the frames spelled out
    (``shown``), since the file is gone once the stage ends.  The budget
    is the renderer launch's own (:data:`gpuwm.rustwx.COMMAND_LINE_BUDGET`),
    the same on every platform so the long-series route is the code the
    tests run wherever they run.  ``stage`` is passed through to
    :func:`_run_stage`.
    """

    from gpuwm.rustwx import COMMAND_LINE_BUDGET

    spelled = render_command(plan, frames, context_frames=context_frames)
    length = len(subprocess.list2cmdline([str(part) for part in spelled]))
    if length <= COMMAND_LINE_BUDGET:
        _run_stage("render", spelled, **stage)
        return
    with render_inputs_file() as inputs:
        if inputs is None:
            # No temporary folder to write the list into: the spelled
            # command is tried, and a system that will not start it is a
            # failed stage that says how long the command was.
            _run_stage("render", spelled, **stage)
            return
        _run_stage("render",
                   render_command(plan, frames, context_frames=context_frames,
                                  inputs_file=inputs),
                   shown=spelled, **stage)


def _start_failure_words(label: str, command: list[str],
                         error: OSError) -> str:
    """Why a stage's process never started, as one sentence."""

    too_long = (getattr(error, "winerror", None) == 206
                or error.errno == errno.E2BIG)
    if too_long:
        length = len(subprocess.list2cmdline([str(part) for part in command]))
        return (f"the {label} stage could not start: its command line is "
                f"{length:,} characters, more than this system will start "
                "a program with")
    reason = error.strerror or type(error).__name__
    return f"the {label} stage could not start: {reason}"


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


def unknown_render_products(spec) -> list[str]:
    """Which tokens of a ``--products`` spec this renderer cannot draw.

    Asked of the RENDERER'S OWN catalog
    (:func:`gpuwm.runplan.render_catalog`, which runs
    ``rw_wrfbatch --list-products``), so a door can refuse a misspelled
    slug at plan review instead of integrating a whole forecast and then
    dying in the render stage -- which is what an unknown slug does: the
    renderer exits nonzero, and a stage that exits nonzero is a
    :class:`GoStageFailed`, not a sentence.

    Empty when there is nothing to say: ``none``, an empty spec, a spec
    containing ``all`` (which IS the catalog), a catalog this install
    cannot ask for (no renderer -- a separate refusal, with its own
    remedy), and any token carrying a ``:``, which is the ``var:`` and
    ``xsec:`` grammar the renderer resolves per file rather than from a
    fixed list.  Group keywords and the four shared short names
    (:data:`gpuwm.render.RUST_PRODUCT_ALIASES`) are known names too.

    The spec is read with the engine's own tokenizer
    (:func:`gpuwm.rustwx.product_spec_terms`), so a section's level list,
    and the term that closes it (``0.1/wa`` in
    ``xsec:QCLOUD=0.01,0.1/wa``), stay inside that section and are never
    asked about as products.
    """

    from gpuwm.rustwx import product_spec_terms

    text = str(spec or "").strip()
    if not text or text.casefold() == "none":
        return []
    tokens = product_spec_terms(text)
    if any(token.casefold() == "all" for token in tokens):
        return []
    from gpuwm.runplan import render_catalog

    catalog = render_catalog()
    products = catalog.get("products")
    if not isinstance(products, list) or not products:
        # No catalog to check against.  Saying "unknown" here would
        # refuse every spelling on a box that simply has no renderer
        # staged, which is a different refusal with a different remedy.
        return []
    from gpuwm.render import RUST_PRODUCT_ALIASES

    known = {str(entry.get("name") or "") for entry in products
             if isinstance(entry, dict)}
    known |= set(RUST_PRODUCT_ALIASES)
    # The renderer matches ``all`` and its group keywords without regard
    # to case (rw-wrfbatch section.rs, split_product_spec); product names
    # are matched as written.
    groups = {str(word).casefold() for word in catalog.get("group_keywords") or ()}
    return [token for token in tokens
            if ":" not in token and token not in known
            and token.casefold() not in groups]


def render_section_value(section, *, base: Path | None = None) -> str | None:
    """A ``--section`` value as the plan records it and every render reads it.

    A ``lat,lon,lat,lon`` line is kept as written.  A file is made
    absolute against ``base`` (the caller's directory when omitted): the
    plan is recorded, printed as a command to paste and read by renders
    that run in other processes, and a relative name means something
    else from any of them.
    """

    if section is None:
        return None
    from gpuwm.rustwx import is_section_line

    text = str(section).strip()
    if not text or is_section_line(text):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() if base is None else Path(base)) / path
    return str(path.resolve())


def admit_render_products(spec, *, section=None) -> None:
    """Refuse, by name, a ``--products`` spec the renderer cannot draw.

    Asked at plan review, dry run included, before anything is fetched
    or created.  WHAT BREAKAGE THIS PREVENTS (gate law): an unknown slug
    reaches the renderer only after the whole forecast has integrated,
    and the renderer refuses it for the WHOLE invocation, so the run
    ends in a failed render stage with no picture of any product it
    asked for.  :func:`unknown_render_products` decides; a box with no
    renderer to ask says nothing here and meets its own refusal.

    ``section`` is the line the ``xsec:`` products are cut along
    (``--section``).  An ``xsec:`` term with none is refused here by
    name, because the render stage can only drop it: the forecast ran in
    full and the term drew nothing, and a request of only section terms
    ran the whole forecast to end on "nothing left to draw".  A line the
    renderer cannot read is refused here too
    (:func:`gpuwm.rustwx.section_line_problem`), because the renderer
    refuses it for the whole invocation, every product of it, and only
    after the forecast.
    """

    unknown = unknown_render_products(spec)
    if unknown:
        raise GoRefusal(
            "--products names " + ", ".join(repr(slug) for slug in unknown)
            + ", which the renderer's catalog does not carry. Next: "
            "gpuwm render --list-products names every product this "
            "install can draw; repeat this command with names from it, "
            "or 'all'.")
    from gpuwm.rustwx import section_line_problem, split_section_spec

    text = str(spec or "").strip()
    sections = ([] if not text or text.casefold() == "none"
                else split_section_spec(text)[1])
    lined = section is not None and str(section).strip() != ""
    if sections and not lined:
        raise GoRefusal(
            "--products names " + ", ".join(repr(term) for term in sections)
            + ", a vertical section, and this command names no line to cut "
            "it along, so it would draw nothing after the whole forecast. "
            "Next: repeat this command with --section=lat,lon,lat,lon (or "
            "--section FILE.json holding {start, end} or a {points, "
            "extend_km} polyline), or drop the term from --products.")
    problem = section_line_problem(section)
    if problem is not None:
        raise GoRefusal(
            f"{problem}. The renderer refuses the whole render for it, "
            "after the whole forecast. Next: repeat this command with "
            "--section=lat,lon,lat,lon (ends at least 1 km apart) or "
            "--section FILE.json holding {start, end} or a {points, "
            "extend_km} polyline.")
    if lined and not sections and text.casefold() != "none":
        print("go: note: --section names a line, but --products names no "
              "xsec: product, so no section is drawn; add "
              "xsec:<field>[/<overlay>...] to --products to draw one.",
              file=sys.stderr)


def _render_stage(plan: dict, *, explain: bool,
                  observer=None, door: str = "go",
                  windows: bool = True) -> bool:
    """Run the render stage; return whether anything was rendered.

    ``windows=False`` runs no windowed pass: the caller has asked the
    engine and its frames hold nothing a window folds, so a pass would
    draw nothing and exit 1 (:meth:`gpuwm.cycle.pictures.BoundaryPictures.finish`).

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
              f"wrfout frame under {plan.get('wrfout_dir', plan['run'] / 'wrfout')}.")
        return False
    from gpuwm.first_products import published_frames
    from gpuwm import live_products
    from gpuwm.render import announce_missing_basemap

    # The render subprocess prints its own map-asset warning into output
    # this stage captures, so the run is told here, beside the finished
    # pictures, where a status that reads only its last events finds it.
    announce_missing_basemap(_stage_warn(observer), plan["render"],
                             stage="finalize")

    # A run that reaches this stage through another door (run-plan's
    # config route, downscale, a WRF-input run) has no start-of-run
    # sweep of its own, so what an earlier run in the same folder marked
    # goes as this one starts drawing.  Only marked stores are touched.
    _sweep_earlier_render_scratch(_scratch_folder(plan), observer=observer)
    already: list[Path] = []
    live = getattr(observer, "live_products", None)
    if live is not None:
        # Stop taking frames and finish the queue before anything is
        # decided, so this stage is never a second writer into the same
        # folder: a render the bounded wait could not finish is told to
        # publish nothing, and every frame the wait did not reach is
        # drawn below.
        live.stop()
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
    every_frame = list(frames)
    from gpuwm.restart_render import history_before_restart
    current_paths = {Path(frame).resolve() for frame in every_frame}
    prior_frames = [frame for frame in history_before_restart(plan.get("restart"))
                    if frame.resolve() not in current_paths]
    frames, already, note = published_frames(frames, plan)
    if note is not None:
        print(f"  -- render: {note}")
    # Every frame drawn while the forecast ran, on every grid, off disk
    # for the same reason: the record licenses nothing until each picture
    # is re-checked against its digest.
    frames, drawn_live, live_note = live_products.published_frames(frames, plan)
    already += drawn_live
    if live_note is not None:
        print(f"  -- render: {live_note}")
    # A request made only of windows has nothing to draw on a frame no
    # window ends on (a grid's first, or one between its whole hours),
    # and a render that draws nothing exits 1: handed those frames, the
    # end-of-run batch stopped the stage before the windowed pass drew
    # the frames that do close windows.  They are left out of the batch
    # and stay its baselines.
    idle: list[Path] = []
    if frames and windows and live_products.windows_only(
            plan.get("render_products"), live_products.catalog_windowed_slugs):
        idle = [frame for frame in frames
                if not live_products.closes_a_window(
                    frame, [*prior_frames, *every_frame])]
        frames = [frame for frame in frames if frame not in idle]
    from gpuwm.render_receipts import SUMMARY_FILENAME
    from gpuwm.render import SCRATCH_PREFIX_ENV, stage_scratch_prefix

    # THIS STAGE'S OWN working stores, named before it opens any: the
    # token goes down to the render subprocess, every store that
    # subprocess opens carries it, and nothing another door spawned can.
    # A sweep below therefore cannot reach a concurrent render's live
    # store, whenever that store was created -- which a "what appeared
    # while this stage ran" test cannot promise, because the second
    # render into one delivery usually opens its store a beat AFTER this
    # one started, and on POSIX removing it takes the live render's work
    # with it.
    stage_prefix = stage_scratch_prefix()

    listed: list = []

    def windowed_slugs(frame):
        # The catalog listing imports a frame into a store of its own, in
        # THIS process.  It takes the stage's token and the stage's
        # scratch root, so the sweep that closes the stage reaches it too:
        # parked beside the frame under the plain prefix, as it was, it
        # sat in `run/wrfout.render-scratch/` where no door could ever
        # claim it, and a successful 24 h run kept it (30 MB, finding F7).
        # Any frame lists every windowed slug of this build, so it is
        # asked once however many questions this stage has.
        if not listed:
            listed.append(live_products.engine_windowed_slugs(
                frame, beside=plan["render"], prefix=stage_prefix))
        return listed[0]

    # The live pass closes each whole hour's own windows (qpf_1h, the
    # 1 h maxima) as the frame lands, over every frame of that hour.
    # What only the whole series holds (qpf_6h, run maxima) is drawn here
    # over every frame of each grid; a frame the live pass did not finish
    # is drawn in the batch below beside its grid's whole series.
    # ``windows=False`` is a caller that already asked the engine and
    # learned no window can be drawn from these frames, so there is no
    # windowed pass at all.
    windowed = (live_products.windowed_passes(
        every_frame, drawn_live, plan.get("render_products"),
        windowed_slugs=windowed_slugs,
        recorded=live_products.recorded_products(plan["render"]),
        live_held=live_products.live_held(plan["render"]),
        prior_frames=prior_frames)
        if windows else [])
    if not frames and not windowed:
        _close_stage_scratch(plan, prefix=stage_prefix, observer=observer)
        # Every frame was published early.  Distinguished from the empty
        # case above because "nothing to do because it is done" and
        # "nothing to do because nothing was produced" are opposite
        # outcomes and used to print the same sentence.
        print(f"  -- render complete: {len(already)} frame(s) were "
              "published while the forecast ran and verified by digest; "
              "nothing was left to draw.")
        summary = _print_closing_notes(plan, explain=explain)
        if summary is not None:
            _notify(observer, "stage_end", label="render", exit_code=0, ok=True,
                    elapsed_seconds=0., progress=summary)
        return True
    # A digest-verified early frame remains an accumulation baseline even
    # though its already published pictures must not be rewritten.
    try:
        if frames:
            # Baselines only where they draw something: every drawn frame
            # of a grid with a frame left to draw on a whole hour, so its
            # windows fold the grid's whole series, and none at all for a
            # request that holds no window.  Anything else is imported for
            # nothing (about 6 s a frame at 1 km).
            baselines = (live_products.baseline_frames(
                frames, [*prior_frames, *already, *idle])
                if windows and live_products.requests_windows(
                    plan.get("render_products"),
                    live_products.catalog_windowed_slugs) else [])
            run_render_pass(
                plan, frames,
                context_frames=baselines,
                explain=explain,
                progress=Path(plan["render"]) / SUMMARY_FILENAME,
                observer=observer, door=door,
                env={SCRATCH_PREFIX_ENV: stage_prefix})
        for wanted, context, products in windowed:
            print(f"  -- render: the windowed pictures of {len(wanted)} frame(s), "
                  "over every frame of their grid")
            try:
                run_render_pass(
                    {**plan, "render_products": products}, wanted,
                    context_frames=context,
                    explain=explain,
                    progress=Path(plan["render"]) / SUMMARY_FILENAME,
                    observer=observer, door=door,
                    env={SCRATCH_PREFIX_ENV: stage_prefix})
            except GoStageFailed as error:
                # Every other picture of these frames is drawn and
                # verified; a failed window pass is said, not a failed run.
                _close_stage_scratch(plan, prefix=stage_prefix,
                                     observer=observer, failed=True)
                print(f"render: warning: the windowed pictures of "
                      f"{len(wanted)} frame(s) were not drawn ({error}); "
                      "every other picture of those frames is in place",
                      file=sys.stderr)
    except GoStageFailed:
        # A render that died mid-store leaves its working tree in the
        # sibling `<case>.render-scratch/` -- tens of GiB of tiled hour
        # files beside a delivery that published nothing, and nothing
        # after this point will ever read them.  The delivered tree is
        # evidence and is kept (see the GoStageFailed arm in go_main);
        # working scratch is not, so it goes, and the line that says the
        # stage failed says what went with it.
        _close_stage_scratch(plan, prefix=stage_prefix, observer=observer,
                             failed=True)
        raise
    # AND WHEN IT SUCCEEDED.  Every render this stage spawned has exited
    # by here, so any store still carrying its token is one a render
    # could not remove itself, and nothing will ever read it.  This used
    # to run only on a failure, so a successful run kept whatever its
    # renders left: 741 MB on one 24 h 3 km forecast (finding F7).
    _close_stage_scratch(plan, prefix=stage_prefix, observer=observer)
    _print_closing_notes(plan, explain=explain)
    return True


def _print_closing_notes(plan: dict, *, explain: bool):
    """The run's closing notes, once, after the last render pass; returns the published summary.

    First the products that drew no picture on any grid, with the first
    reason each was skipped (:func:`_print_undrawn_note`), then the
    products one grid has and another has none of
    (:func:`_print_unpictured`), which never names a product no grid
    drew.
    """

    from gpuwm.render_receipts import read_summary
    try:
        summary = read_summary(plan["render"])
    except (OSError, ValueError):
        summary = None
    _print_undrawn_note(summary, explain=explain)
    _print_unpictured(plan)
    return summary


def _print_unpictured(plan: dict) -> None:
    """Name every product one grid has and another grid has no picture of.

    The line each render prints covers its own invocation only, so a
    nest with no ``qpf_1h`` at all used to finish under a note that read
    as one frame's skip.  A listing that cannot be read says nothing
    rather than fail a finished render.
    """

    from gpuwm import live_products

    try:
        note = live_products.unpictured_note(plan["render"],
                                             plan.get("render_products"))
    except Exception:  # noqa: BLE001 - a closing note never fails the stage
        return
    if note is not None:
        print(f"    {note}")


def _print_undrawn_note(summary, *, explain: bool) -> None:
    """The run's closing note: every product that drew NO picture.

    Printed once, after the last render pass, from the published summary
    -- which counts the early frame, every frame drawn while the forecast
    ran and every end-of-run pass together.  Each pass's own note speaks
    only for that pass: the end-of-run pass of a run whose frames were all
    drawn live named ``qpf_1h`` (24 pictures, skipped at F000) and never
    the three products the live renders had skipped on every frame.
    """

    from gpuwm.render_receipts import undrawn_note

    note = undrawn_note(summary)
    if note is None:
        return
    headline, detail = note
    print(render(layered(headline, detail), explain=explain,
                 command="gpuwm render"))


def _stage_warn(observer):
    """``(code, message, **fields)`` onto the run's events, or stderr.

    An observer with a ``warn`` hook owns an event stream and a reader
    who sees it (the desktop, the terminal workspace, the web page);
    without one this stage belongs to a terminal command, and stderr is
    what that reader sees.
    """

    hook = getattr(observer, "warn", None)

    def warn(code: str, message: str, **fields) -> None:
        if callable(hook):
            hook(code, message, **fields)
        else:
            print(f"render: warning: {message}", file=sys.stderr, flush=True)

    return warn


def _close_stage_scratch(plan: dict, *, prefix: str, stage: str = "render",
                         observer=None, failed: bool = False) -> None:
    """Remove the working stores one ended stage left, and say so.

    Called once every process the stage spawned has exited, whether the
    stage passed or failed; ``failed`` only chooses the words.  ``stage``
    names it, because two stages draw: the render stage, and the
    forecast stage when it carries the early render.  Both open their
    stores under a token this door minted, and only that token's stores
    are touched -- anything else beside the delivery belongs to another
    render.

    The stores a render left are a warning in the run's events as well
    as on the terminal (code ``render_scratch_left``): the render's own
    "left behind" line goes to a stderr this door captures and, on a
    passing stage, discards.  A store that STILL cannot be removed is
    marked, so the next run in the same folder removes it
    (:func:`gpuwm.render.sweep_marked_scratch`).  Never raises: a cleanup
    never replaces the stage's own outcome.
    """
    from gpuwm.render import (mark_abandoned_scratch, owned_scratch_stores,
                              scratch_root_for, sweep_abandoned_scratch)
    from gpuwm.render_layout import fs_path

    try:
        root = scratch_root_for(plan["render"])
        found = owned_scratch_stores(plan["render"], prefix=prefix)
        if not found:
            return
        sizes = {store: _tree_bytes(fs_path(store, descend=True))
                 for store in found}
        removed = sweep_abandoned_scratch(plan["render"], prefix=prefix)
        left = [store for store in found if store not in removed]
        if left:
            mark_abandoned_scratch(left, run=plan.get("root"))
    except Exception:            # noqa: BLE001 - see the docstring
        return
    held = sum(sizes.values())
    kept = sum(sizes[store] for store in left)
    if failed:
        what = f"the failed {stage} left"
    else:
        what = f"the {stage} stage passed but its renders left"
    message = (f"{what} {len(found)} working store(s) "
               f"({_human_bytes(held)}) in {root}; they held no product "
               "and nothing later reads them, so ")
    if not left:
        message += "they were removed once every render had exited."
    elif len(left) == len(found):
        message += (f"they were to be removed, but none could be yet. They "
                    f"are marked, and the next run in {_scratch_folder(plan)} "
                    "removes them.")
    else:
        message += (f"they were removed once every render had exited, "
                    f"except {len(left)} ({_human_bytes(kept)}) that could "
                    f"not be removed yet. Those are marked, and the next "
                    f"run in {_scratch_folder(plan)} removes them.")
    if failed:
        message += (" Only this stage's own stores were touched; anything "
                    "else beside the delivery belongs to another render "
                    "and was left alone.")
    print(f"{stage}: warning: {message}", file=sys.stderr)
    _notify(observer, "stage_warning", label=stage,
            code="render_scratch_left", message=message,
            scratch_root=str(root), stores=len(found), bytes=held,
            removed=len(found) - len(left), kept=len(left),
            kept_bytes=kept, stage_failed=bool(failed))


def _scratch_folder(plan: dict) -> Path:
    """The folder whose next run removes what this run's renders left.

    The case folder a run folder sits in, or the folder the run was
    written straight into; :func:`gpuwm.render.sweep_marked_scratch`
    looks two levels down from it, which is where every door's
    ``<delivery>.render-scratch`` root lands.
    """
    from gpuwm.render import scratch_root_for

    run = run_stamp_module.owning_run(plan["render"])
    if run is not None:
        return run.parent
    return scratch_root_for(plan["render"]).parent


def _sweep_earlier_render_scratch(folder, *, observer=None) -> None:
    """Remove the working stores earlier runs in ``folder`` marked, and say so.

    The second half of :func:`_close_stage_scratch`: a store another
    program still held when its own run ended was marked, and a run
    starting in the same folder removes it.  Only marked stores are
    touched, so a run still going in the same folder keeps its own.
    Never raises.
    """
    from gpuwm.render import sweep_marked_scratch

    try:
        removed = sweep_marked_scratch(folder)
    except Exception:            # noqa: BLE001 - housekeeping only
        return
    if not removed:
        return
    message = (f"removed {len(removed)} render working store(s) an "
               f"earlier run in {folder} could not remove when it ended")
    print(f"go: {message}", file=sys.stderr)
    _notify(observer, "stage_warning", label="render",
            code="render_scratch_swept", message=message,
            folder=str(folder), stores=len(removed))


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
    from gpuwm.source_posting import SCHEDULE_SCHEMA

    # The as-posted fetch's schedule is the progress file of a fetch run
    # beside the preparation.
    if payload.get("schema") == SCHEDULE_SCHEMA:
        return _posting_note(payload)
    status = payload.get("status")
    model = payload.get("model_elapsed_seconds")
    parts = [str(status)] if isinstance(status, str) else []
    if isinstance(model, (int, float)) and model > 0:
        parts.append(f"{float(model):.0f} model seconds")
    return (", " + ", ".join(parts)) if parts else ""


def _posting_note(schedule: dict) -> str:
    """The fetch's beat line as posted: leads in, and the lead it waits for.

    Said from the fetch's own schedule, so the terminal says the run waits
    on its source, not that a stage is merely taking a while: before the
    first leads post nothing else of the run moves.
    """

    rows = [row for row in schedule.get("leads") or ()
            if isinstance(row, dict)]
    if not rows:
        return ""
    ready = sum(1 for row in rows if row.get("state") == "ready")
    note = f", {ready} of {len(rows)} leads in"
    waiting = next((row for row in rows if row.get("state") == "waiting"),
                   None)
    if waiting is not None:
        expected = str(waiting.get("expected_at") or "")
        when = (f" (scheduled from about {expected[11:16]}Z)"
                if len(expected) >= 16 else "")
        note += (f", waiting for {schedule.get('source')} "
                 f"f{int(waiting.get('lead', 0)):03d}, not posted yet{when}")
    return note


def _heartbeat_note(status: str) -> str | None:
    """The forecast heartbeat's phase for the stage beat line, or ``None``.

    Only the named phases: a preparation, a beat after the last step, and
    a write between two steps.  A write is worded as the forecast
    progress line words it (:func:`gpuwm.progress.write_phase_words`), so
    the read-back of a checkpoint reads "checking checkpoint" and not
    "writing verify checkpoint".  ``None`` for any other status, whose
    beat line keeps the stage's own progress note.
    """

    if status.startswith("writing:"):
        from gpuwm.progress import write_phase_words

        return ", " + write_phase_words(status.removeprefix("writing:"))
    if status.startswith(("preparing:", "finalizing:")):
        return ", " + status.replace(":", " ").replace("-", " ")
    if status.startswith("waiting:"):
        # A seam wait says what it waits on: a source lead not posted yet,
        # or the preparation still building the interval.
        return ", waiting on the " + status.removeprefix("waiting:")
    return None


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


#: Every stage subprocess :func:`_run_stage` is waiting on, so a door
#: that owns its own stop can end the one in flight
#: (:func:`end_stage_processes`).  Recording one changes nothing about
#: how it runs; ``gpuwm go`` never ends one (see :class:`GoInterrupted`).
_STAGE_PROCESSES: dict[int, subprocess.Popen] = {}
_STAGE_PROCESSES_LOCK = threading.Lock()

#: How long :func:`end_stage_processes` lets a stage answer its SIGINT
#: before killing it.  The desktop kills a stopped run 5 s after asking.
END_STAGE_GRACE_SECONDS = 2.0


def end_stage_processes(*, grace: float = END_STAGE_GRACE_SECONDS) -> bool:
    """End every stage subprocess this process is waiting on.

    For a door that records its own stop (``gpuwm downscale``).  THE
    BREAKAGE: a child stopped with ``kill -TERM <pid>`` while its
    finalize render ran left that render drawing into the run's folder
    after the child had exited, because the signal reached the child
    alone and the render was never told.  Each stage is sent SIGINT
    (``gpuwm render`` answers it by ending the renderer it runs and
    exiting 130) and killed if it has not exited within ``grace``; off
    POSIX it is terminated.  The stage is signalled by pid, never by
    group: it shares the caller's group.

    Returns whether no stage is left running.  Never raises.
    """

    with _STAGE_PROCESSES_LOCK:
        processes = list(_STAGE_PROCESSES.values())
    running = [process for process in processes if process.poll() is None]
    for process in running:
        try:
            if os.name == "posix":
                import signal

                process.send_signal(signal.SIGINT)
            else:
                process.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + grace
    for process in running:
        try:
            process.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(grace)
            except (OSError, subprocess.TimeoutExpired):
                pass
    return all(process.poll() is not None for process in running)


#: The ``warning:`` and ``note:`` lines one ``gpuwm go`` has shown, or
#: None outside one.  Every stage is its own process and loads the
#: configuration again, so a configuration warning is printed by each of
#: them; the terminal shows it once, and a launch that keeps a log keeps
#: every copy there.
_SHOWN_ADVISORIES: set[str] | None = None
_SHOWN_ADVISORIES_LOCK = threading.Lock()


def _first_showing(line: str) -> bool:
    """Whether this advisory line is new to the terminal of this launch."""

    said = " ".join(str(line).split())
    with _SHOWN_ADVISORIES_LOCK:
        if _SHOWN_ADVISORIES is None:
            return True
        if said in _SHOWN_ADVISORIES:
            return False
        _SHOWN_ADVISORIES.add(said)
        return True


def _stage_relay_line(line: str) -> bool:
    """Whether a passing stage's advisory line is printed.

    Printed whole when this process's output passes through a relay that
    logs every line and shows each advisory once (a registered launch's
    ``LaunchOutput``); otherwise this is the terminal, and a line it
    already showed is left out.
    """

    if getattr(sys.stdout, "relays_advisories_once", False):
        return True
    return _first_showing(line)


@contextlib.contextmanager
def _each_advisory_once():
    """Show each advisory line once for the launch inside this block.

    A warning this process prints itself counts as shown, so a stage that
    says the same sentence again is not relayed a second time.
    """

    global _SHOWN_ADVISORIES
    from gpuwm.explain import add_warning_observer, remove_warning_observer

    def observe(record) -> None:
        said = " ".join(f"warning: {record['action']}".split())
        with _SHOWN_ADVISORIES_LOCK:
            if _SHOWN_ADVISORIES is not None:
                _SHOWN_ADVISORIES.add(said)

    with _SHOWN_ADVISORIES_LOCK:
        previous, _SHOWN_ADVISORIES = _SHOWN_ADVISORIES, set()
    add_warning_observer(observe)
    try:
        yield
    finally:
        remove_warning_observer(observe)
        with _SHOWN_ADVISORIES_LOCK:
            _SHOWN_ADVISORIES = previous


def _run_stage(label: str, command: list[str], *, explain: bool,
               progress: Path | None = None,
               heartbeat_seconds: float = HEARTBEAT_SECONDS,
               observer=None, door: str = "go",
               env: dict | None = None,
               shown: list[str] | None = None,
               stopped: Callable[[], bool] | None = None,
               secondary_failure: Callable[[], bool] | None = None) -> None:
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

    ``door`` names the command the reader typed, for the one sentence
    below that addresses them directly.

    The output is read as the stage writes it (:func:`_read_stage_output`),
    and each preparation step record in it goes to the observer's ``warn``
    as a ``preparation_progress`` record while the stage runs, which is
    how a run page hears a preparer program's steps.

    ``env`` adds to the environment every stage gets (:func:`_stage_env`)
    the few names THIS stage needs -- the render stage's own scratch
    token, today.  Added here rather than spelled onto the command line
    because these lines are also printed for a reader to paste, and a
    pasted line carrying one run's private token would be wrong the
    moment it is reused.

    ``shown`` is the command as a reader would type it, when that is not
    the one run: the render stage hands its frames over in a private file
    it removes when the stage ends (:func:`render_inputs_file`).  The
    ``stage_begin`` event carries ``shown``, frames spelled out, because
    a recorded command naming that file names nothing once the stage is
    over.

    ``stopped`` answers whether the chain itself ended this stage (the
    as-posted fetch beside a preparation or forecast that failed,
    :meth:`_BesideStage.stop`).  A stage the chain stopped that then exits
    nonzero is said as stopped, not failed: no FAILED line, no "stopped
    at" sentence and no ``stage_failed`` or failed ``stage_end`` for the
    observer, and :class:`GoStageStopped` is raised.  The breakage this
    prevents: a preparation that failed while the fetch still ran was
    reported as a failed fetch with the fetch's exit 130 (the interrupt
    code), on the terminal and in a hosting run's failure record.

    ``secondary_failure`` says another callback already failed first.
    The process continues and keeps its actual error and timing, but its
    later failure closes through ``stage_secondary_end`` rather than
    announcing that this stage stopped the run.
    """

    print(f"  .. {label}", flush=True)
    _notify(observer, "stage_begin", label=label,
            command=list(command if shown is None else shown))
    started = time.monotonic()
    watchdog = None
    if label == "forecast" and "--outdir" in command and command[1:2] == ["-m"]:
        from gpuwm.forecast_supervisor import ForecastWatchdog

        watchdog = ForecastWatchdog(command)
        command = watchdog.command
        env = {**(env or {}), **watchdog.env}
    # The waiting runs on a worker thread so this one stays free to say
    # that it is waiting.
    box: dict[str, object] = {}

    def _step(record: dict) -> None:
        # A preparer program's step, onto the run's stream through the
        # observer, while the stage is still running.
        from gpuwm.progress import prep_record_event

        _notify(observer, "warn", **prep_record_event(record))

    def _wait() -> None:
        try:
            # Popen spelled out publishes the child's pid, which the
            # interrupt path has to be able to NAME (it does not signal
            # it -- see GoInterrupted).
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, errors="replace", cwd=str(_stage_cwd()),
                env={**_stage_env(), **(env or {})})
            box["pid"] = proc.pid
            box["process"] = proc
            with _STAGE_PROCESSES_LOCK:
                _STAGE_PROCESSES[id(box)] = proc
            try:
                out, err = _read_stage_output(proc, label=label, on_step=_step)
            finally:
                with _STAGE_PROCESSES_LOCK:
                    _STAGE_PROCESSES.pop(id(box), None)
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
            if watchdog is not None and worker.is_alive() and "process" in box:
                reason = watchdog.check(box["pid"])
                if reason is not None:
                    box["watchdog_failure"] = reason
                    watchdog.terminate(box["process"])
                    watchdog.failed()
                    break
            if worker.is_alive() and now >= next_beat:
                note = _progress_note(progress)
                if watchdog is not None and watchdog.last is not None:
                    note = _heartbeat_note(watchdog.last.status) or note
                print(f"     .. {label}, {_elapsed_words(now - started)}"
                      f"{note}", flush=True)
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
        error = box["error"]
        if not isinstance(error, OSError) or "pid" in box:
            raise error
        # The process never started (a missing interpreter, a command
        # line longer than the system takes).  This escaped as the
        # operating system's own exception, and the observer heard a
        # stage begin and never end: the desktop showed it running.  It
        # is a failed stage, said the way every other one is.
        reason = _start_failure_words(label, command, error)
        if secondary_failure is not None and secondary_failure():
            _notify(observer, "stage_secondary_end", label=label,
                    exit_code=_STAGE_START_FAILED, ok=False, diagnostic=reason,
                    elapsed_seconds=time.monotonic() - started,
                    progress=_progress_payload(progress))
            raise GoStageFailed(_STAGE_START_FAILED, reason)
        print(f"  FAILED  {label} (did not start)")
        print(f"    {reason}")
        print(f"{door}: stopped at {label}; every later stage consumes "
              "this one's output, so nothing after it ran.")
        _notify(observer, "stage_failed", label=label,
                exit_code=_STAGE_START_FAILED, diagnostic=reason)
        _notify(observer, "stage_end", label=label,
                exit_code=_STAGE_START_FAILED, ok=False,
                elapsed_seconds=time.monotonic() - started,
                progress=_progress_payload(progress))
        raise GoStageFailed(_STAGE_START_FAILED, reason)
    completed = box["completed"]
    if "watchdog_failure" in box:
        completed = subprocess.CompletedProcess(
            command, 124, completed.stdout,
            (completed.stderr or "").rstrip() + "\n" + box["watchdog_failure"])
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0 and stopped is not None and stopped():
        print(f"  stopped {label} (exit {completed.returncode}; ended by "
              f"{door} because a stage beside it failed)", flush=True)
        raise GoStageStopped(completed.returncode)
    if completed.returncode != 0:
        diagnostic = ("\n".join(_said_lines(completed.stderr)).strip()
                      or "\n".join(_said_lines(completed.stdout)).strip())
        tail_text = _diagnostic_tail(diagnostic)
        if secondary_failure is not None and secondary_failure():
            _notify(observer, "stage_secondary_end", label=label,
                    exit_code=completed.returncode, ok=False, diagnostic=tail_text,
                    elapsed_seconds=time.monotonic() - started,
                    progress=_progress_payload(progress))
            raise GoStageFailed(completed.returncode, tail_text)
        print(f"  FAILED  {label} (exit {completed.returncode})")
        # The stage's own refusal is usually its last few lines; replay
        # a readable tail by default and everything under --explain.
        # A step record is left out of the tail: the steps were said on
        # the run's stream as they ran, and a preparer's raw step lines
        # filled the tail and pushed its refusal out of it.
        lines = output.splitlines() if explain else _said_lines(output)
        tail = lines if explain else lines[-_FAILURE_TAIL_LINES:]
        if len(tail) < len(lines):
            print(f"    ... ({len(lines) - len(tail)} earlier line(s); "
                  "re-run with --explain to replay all of them)")
        for line in tail:
            print(f"    {line}")
        # The DOOR the reader typed, not a door they did not.  A
        # downscaled child runs this very stage, and a line reading
        # "go: stopped at render" sends its reader to `gpuwm go`'s
        # documentation for a failure that happened under
        # `gpuwm downscale`.
        print(f"{door}: stopped at {label}; every later stage consumes "
              "this one's output, so nothing after it ran.")
        # Carry the same diagnostic into machine-facing failures. Desktop and
        # remote clients cannot rely on a separate terminal's preceding lines.
        # One tail, three readers: the event stream, the exception a
        # calling door turns into its own refusal, and the terminal
        # above.  Composed once so they cannot disagree about which
        # lines the stage's failure was.
        _notify(observer, "stage_failed", label=label,
                exit_code=completed.returncode, diagnostic=tail_text)
        _notify(observer, "stage_end", label=label,
                exit_code=completed.returncode, ok=False,
                elapsed_seconds=time.monotonic() - started,
                progress=_progress_payload(progress))
        raise GoStageFailed(completed.returncode, tail_text)
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
            if ((head.startswith("warning:") or head.startswith("note:"))
                    and _stage_relay_line(line)):
                print(f"    {line.strip()}")


def _read_stage_output(process, *, label: str, on_step) -> tuple[str, str]:
    """A stage's stdout and stderr, read as it writes them.

    Returns the two texts whole, as ``communicate()`` did, so the failure
    tail and the replay are unchanged.  While the stage runs, each step
    record a preparer program writes (a ``GPUWM_PREP_EVENT`` line) goes to
    ``on_step`` the moment it is read.  The breakage this prevents: with
    ``communicate()`` a stage's output reached this process only when the
    stage ended, so a run page on the prepared HRRR and GFS routes showed
    no preparation step until the preparation was over.  Both pipes are
    read at once, so a child that fills one while this waits on the other
    cannot stall.
    """

    from gpuwm.prep_progress import step_record

    texts: dict[str, str] = {}
    failures: list[BaseException] = []

    def drain(name: str, pipe) -> None:
        kept: list[str] = []
        try:
            for line in iter(pipe.readline, ""):
                kept.append(line)
                record = step_record(line.rstrip("\r\n"))
                if record is not None:
                    on_step(record)
        except BaseException as error:  # noqa: BLE001 - raised after the join
            failures.append(error)
        finally:
            texts[name] = "".join(kept)
            try:
                pipe.close()
            except OSError:
                pass

    readers = [threading.Thread(target=drain, args=(name, pipe),
                                name=f"go-read-{label}-{name}", daemon=True)
               for name, pipe in (("stdout", process.stdout),
                                  ("stderr", process.stderr))]
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join()
    process.wait()
    if failures:
        raise failures[0]
    return texts["stdout"], texts["stderr"]


def _said_lines(text: str | None) -> list[str]:
    """A stage's output lines less the step records its preparer wrote.

    For a failed stage's tail: each step was said on the run's stream as
    it happened, and a tail of raw ``GPUWM_PREP_EVENT`` JSON lines hid the
    refusal the tail is shown for.  The whole text stays in the log.
    """

    from gpuwm.prep_progress import step_record

    return [line for line in (text or "").splitlines()
            if step_record(line.strip()) is None]


#: The exit code a stage whose process never started is reported with:
#: the shells' own "command could not be run".
_STAGE_START_FAILED = 127

#: A failed stage's diagnostic: its last lines, each at most this long.
_DIAGNOSTIC_LINES = 8
_DIAGNOSTIC_LINE_CHARS = 1_000


def _diagnostic_tail(text: str) -> str:
    """The ``stage_failed`` diagnostic: a failed stage's last lines.

    Each line is cut to :data:`_DIAGNOSTIC_LINE_CHARS`, keeping its start
    and its end, so no single line can crowd out the rest.  The tail used
    to be the last 8,192 characters of the last eight lines, and one line
    spelling out a long series' frames filled all of it: the desktop was
    shown the middle of a frame path and never the refusal above it.
    """

    lines = []
    for line in text.splitlines()[-_DIAGNOSTIC_LINES:]:
        if len(line) > _DIAGNOSTIC_LINE_CHARS:
            keep = (_DIAGNOSTIC_LINE_CHARS - len(" ... ")) // 2
            line = f"{line[:keep]} ... {line[-keep:]}"
        lines.append(line)
    return "\n".join(lines)


#: Failure-replay tail length: enough to carry any refusal message this
#: tree prints plus a Python traceback's tail, small enough to read.
_FAILURE_TAIL_LINES = 30


#: The chain-stage primitive, under a public name.
#:
#: The legacy stage composer owns "run one documented command, capture it,
#: heartbeat while it waits, replay everything it said if it failed, and
#: tell an observer" is not GFS-specific -- it is what running ANY stage
#: of ANY documented chain looks like here.  The HRRR route reuses it
#: rather than growing a second copy with slightly different capture and
#: failure-replay semantics, which is how two chains end up refusing
#: differently for the same reason.
run_stage = _run_stage


# ---------------------------------------------------------------------------
# The as-posted fetch, run beside the preparation (DESIGN A136 2.5)
# ---------------------------------------------------------------------------

#: How much earlier than this fetch's launch a posting schedule may be dated
#: and still be this fetch's: a filesystem that keeps modification times to
#: two seconds (FAT) dates a fresh write up to that much early.
_FRESH_SCHEDULE_SLACK_SECONDS = 2.0
#: How often the chain looks for the fetch's posting schedule.
_POSTING_POLL_SECONDS = 0.2
#: How long a fetch that has said it is failing (its own ``failed.json``,
#: ``source_behind``) is given to end on its own once the preparation has
#: failed on that record, before the chain stops it.  It exits right after
#: writing the record, but on a loaded host (node-4 under a full test
#: battery, 1.5 GiB of free RAM) it took longer than the 2 s every other
#: stop allows (:data:`END_STAGE_GRACE_SECONDS`), the chain stopped it, and
#: the preparation that failed because of it was named the run's failure.
_SAID_FAILURE_GRACE_SECONDS = 30.0
#: The failure codes this chain writes into the fetch's ``failed.json`` when
#: the fetch ended without writing one, so a preparation waiting on a lead
#: marker ends by name (``PostedLeads``) instead of waiting for a marker
#: nothing will write.  The fetch's own ``source_behind`` is never replaced.
FETCH_FAILED_CODE = "fetch_failed"
FETCH_INCOMPLETE_CODE = "fetch_incomplete"
#: The fetch was ended by the chain because a stage beside it failed: not a
#: failure of the fetch, and still an end to a wait on a marker, because a
#: preparation process can outlive its stage (a killed stage wrapper leaves
#: the preparer it started waiting on the next lead).
FETCH_STOPPED_CODE = "fetch_stopped"


def posting_folder(plan: dict) -> Path:
    """The ``posting/`` folder this chain's fetch publishes lead markers in."""

    from gpuwm.source_posting import POSTING_DIRNAME

    return Path(plan["data"]) / POSTING_DIRNAME


def posts_beside_preparation(plan: dict) -> bool:
    """Whether this chain fetches and prepares at once (DESIGN A136 2.5).

    As posted (the default; ``--whole-cycle`` or ``as_posted = false`` is
    the old order), for a source whose preparation waits on posted leads
    (:func:`gpuwm.source_cli.prepares_as_posted`), one domain or a tree: a
    tree's children bind the input plan at its head and its seal writes
    the one-shot tree (``gfs_direct._prepare_chained_gfs_tree``), so every
    nest shape, moving and delayed nests included, starts on its start
    leads.
    """

    from gpuwm.source_cli import prepares_as_posted

    return (plan.get("as_posted") is not False
            and prepares_as_posted(str(plan["source"])))


def _fresh_schedule(folder: Path, plan: dict, *, since: float) -> bool:
    """Whether ``folder`` holds this fetch's schedule of this window.

    A schedule left by an earlier fetch into the same download is not this
    run's: its window may have been fetched whole since, and its markers
    would then describe files the fetch has replaced.
    """

    from gpuwm.ingest.boundary_stream import read_replaced_json
    from gpuwm.source_posting import POSTING_SCHEDULE_NAME

    path = Path(folder) / POSTING_SCHEDULE_NAME
    try:
        if path.stat().st_mtime < since - _FRESH_SCHEDULE_SLACK_SECONDS:
            return False
        schedule = read_replaced_json(path)
    except (OSError, ValueError):
        return False
    if not isinstance(schedule, dict):
        return False
    cycle = str(plan.get("cycle") or "")[:13]
    return (schedule.get("source") == plan.get("source")
            and str(schedule.get("cycle") or "")[:13] == cycle)


def _record_fetch_end(folder: Path, plan: dict, *, since: float,
                      error: BaseException | None,
                      stopped: bool = False) -> None:
    """Tell a preparation waiting on lead markers that no more will come.

    Written only into this fetch's own schedule's folder, only when the
    fetch did not say why itself (its ``source_behind`` record stands),
    and on success only when a lead of the schedule has no marker.  Never
    raises: the chain reports the fetch's own failure either way.
    ``stopped``: the chain ended the fetch because a stage beside it
    failed, so the record says that (``fetch_stopped``), not a failure of
    the fetch.
    """

    from gpuwm.ingest.boundary_stream import (
        _write_json_atomic, posted_lead_marker_name, read_replaced_json,
    )
    from gpuwm.source_posting import (POSTING_FAILED_NAME,
                                      POSTING_SCHEDULE_NAME)

    folder = Path(folder)
    try:
        if (not _fresh_schedule(folder, plan, since=since)
                or (folder / POSTING_FAILED_NAME).exists()):
            return
        schedule = read_replaced_json(folder / POSTING_SCHEDULE_NAME)
        leads = [int(row["lead"]) for row in schedule.get("leads") or ()]
        missing = [lead for lead in leads
                   if not (folder / posted_lead_marker_name(lead)).is_file()]
        if (error is None or stopped) and not missing:
            return
        if stopped:
            code, message = FETCH_STOPPED_CODE, (
                "the fetch stage was stopped because a stage beside it "
                "failed; no later lead will be fetched for this run")
        elif error is not None:
            detail = getattr(error, "diagnostic", None) or str(error)
            code, message = FETCH_FAILED_CODE, (
                f"the fetch stage failed (exit "
                f"{getattr(error, 'code', '?')}): {detail}".strip())
        else:
            code, message = FETCH_INCOMPLETE_CODE, (
                "the fetch stage ended without a marker for "
                + ", ".join(f"f{lead:03d}" for lead in missing))
        _write_json_atomic(folder / POSTING_FAILED_NAME, {
            "code": code, "source": schedule.get("source"),
            "cycle": schedule.get("cycle"),
            "lead": missing[0] if missing else None,
            "leads_missing": missing, "message": message,
        })
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return


def _end_process(process, *, grace: float) -> None:
    """End one stage subprocess this chain started, as end_stage_processes does."""

    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            import signal

            process.send_signal(signal.SIGINT)
        else:
            process.terminate()
    except OSError:
        return
    try:
        process.wait(grace)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(grace)
        except (OSError, subprocess.TimeoutExpired):
            pass


class _BesideStage:
    """One chain stage run on its own thread, beside the stages after it.

    The as-posted fetch (DESIGN A136 2.5) publishes each lead's marker as
    the lead posts while the preparation waits on those markers and the
    forecast on the preparation, so it cannot finish before they start.
    It runs exactly as every other stage does (:func:`_run_stage`:
    captured, a heartbeat while it runs, replayed and told to the observer
    if it fails); only the waiting moves off the chain's thread.  When it
    ends without a marker for every lead it tells the preparation so
    (:func:`_record_fetch_end`).
    """

    def __init__(self, label: str, command: list[str], *, plan: dict,
                 explain: bool, observer=None, progress: Path | None = None):
        import contextvars

        self.label = label
        self.command = list(command)
        self.plan = plan
        self.progress = progress
        self.launched = time.time()
        self._box: dict[str, BaseException] = {}
        #: Set by :meth:`stop` before it signals the stage: the chain ended
        #: it, so its exit is not a failure of its own (``_run_stage``'s
        #: ``stopped``).
        self._stopped = threading.Event()
        #: A different stage failed first. This suppresses failure
        #: announcements without canceling the fetch or changing its exit.
        self._secondary_failure = threading.Event()
        context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=context.run, args=(self._run, explain, observer),
            name=f"go-beside-{label}", daemon=True)
        self._thread.start()

    def _run(self, explain: bool, observer) -> None:
        error = None
        try:
            _run_stage(self.label, self.command, explain=explain,
                       observer=observer, progress=self.progress,
                       stopped=self._stopped.is_set,
                       secondary_failure=self._secondary_failure.is_set)
        except BaseException as failure:  # noqa: BLE001 - see result()
            error = failure
            if not self._stopped.is_set():
                self._box["error"] = failure
        finally:
            # A fetch the chain stopped says so (fetch_stopped), not "the
            # fetch stage failed (exit 130)", which misnamed the failure
            # for a reader of posting/.  It still writes a record: a
            # preparer can outlive its stage (a killed stage wrapper leaves
            # the preparer it started), and without one it waited forever
            # for a marker nothing would write.
            _record_fetch_end(posting_folder(self.plan), self.plan,
                              since=self.launched, error=error,
                              stopped=self._stopped.is_set())

    def running(self) -> bool:
        return self._thread.is_alive()

    def suppress_failure_publication(self) -> None:
        """A later fetch failure remains secondary while preparation seals."""

        self._secondary_failure.set()

    def failure(self) -> BaseException | None:
        """The stage's own failure, once it has ended with one; else ``None``."""

        return self._box.get("error")

    def said_failure(self) -> bool:
        """Whether the fetch has failed on its own, or has said it is failing.

        Its own ``posting/failed.json`` (a lead past its budget writes
        ``source_behind`` and then exits 75) is written before the stage
        ends, so a preparation can fail on it first; the fetch clears the
        file when it starts, so one found here is this fetch's.
        """

        from gpuwm.source_posting import POSTING_FAILED_NAME

        return ("error" in self._box
                or (posting_folder(self.plan) / POSTING_FAILED_NAME).is_file())

    def settle(self, *, fetch_first: bool,
               grace: float = END_STAGE_GRACE_SECONDS,
               said_grace: float = _SAID_FAILURE_GRACE_SECONDS
               ) -> BaseException | None:
        """After the chain beside this stage failed: the chain's failure, if it is this one's.

        ``fetch_first``: the fetch had failed, or said it was failing,
        before the preparation or forecast failed (so they failed because
        of it).  Then the fetch is ending on its own; it is given
        ``said_grace`` to end, and its failure (its exit code and words) is
        the chain's.  Otherwise, or if it did not end with a failure, the
        fetch is stopped within ``grace`` (a download for a run that has
        ended) and ``None`` returned: the stage that failed first is the
        chain's failure.
        """

        if fetch_first:
            self._thread.join(said_grace)
            failed = self.failure()
            if failed is not None:
                return failed
        self.stop(grace=grace)
        return None

    def result(self) -> None:
        """Wait for the stage to end; raise its failure if it failed."""

        self._thread.join()
        if "error" in self._box:
            raise self._box["error"]

    def _processes(self) -> list:
        with _STAGE_PROCESSES_LOCK:
            return [process for process in _STAGE_PROCESSES.values()
                    if getattr(process, "args", None) == self.command]

    def pid(self) -> int | None:
        processes = self._processes()
        return getattr(processes[0], "pid", None) if processes else None

    def stop(self, *, grace: float = END_STAGE_GRACE_SECONDS) -> None:
        """End the stage's process when the chain failed before it ended.

        Not on an interrupt: Ctrl-C reached the stage with the rest of the
        foreground process group, and the interrupt contract kills no child
        (:class:`GoInterrupted`).  On a failure the chain returns, and a
        fetch left running would go on downloading for a run that has
        ended.
        """

        self._stopped.set()
        for process in self._processes():
            _end_process(process, grace=grace)
        self._thread.join(grace)


def _await_posting(beside: _BesideStage, plan: dict) -> Path | None:
    """The posting folder once the fetch beside the chain has scheduled its window.

    ``None`` when the fetch ended without waiting on any lead: its window
    was already here whole (a cached download), or its source is not
    waited on, so there are no lead markers and the preparation binds the
    whole window as before.  A fetch that failed first raises its failure.
    """

    folder = posting_folder(plan)
    try:
        while True:
            if _fresh_schedule(folder, plan, since=beside.launched):
                return folder
            if not beside.running():
                if _fresh_schedule(folder, plan, since=beside.launched):
                    return folder
                beside.result()
                return None
            time.sleep(_POSTING_POLL_SECONDS)
    except KeyboardInterrupt:
        raise GoInterrupted(beside.label, beside.pid()) from None


class GoStageFailed(Exception):
    """A stage exited nonzero; its output has already been replayed.

    "Replayed" means to the TERMINAL.  A door that turns this into its
    own refusal has no terminal to read -- the desktop's run view shows
    the refusal and the command, and nothing of the stage's own output
    -- so the same tail that goes to the terminal and onto the
    ``stage_failed`` event rides the exception as :attr:`diagnostic`.
    Without it a reader was handed a render command and an exit code and
    had to re-run the whole render to learn which product failed.
    """

    def __init__(self, code: int, diagnostic: str = ""):
        super().__init__(f"stage exited {code}")
        self.code = code
        #: The stage's own last lines, or ``""`` when it produced none.
        self.diagnostic = diagnostic


class GoStageStopped(GoStageFailed):
    """A stage the chain ended itself exited nonzero (``_run_stage``'s ``stopped``).

    Not the chain's failure: the stage beside it that failed is.  Nothing
    was said to the observer and no failure lines were replayed.
    """


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

    #: The exit code this interrupt is worth, on the exception itself so
    #: a front door that catches it as one of many ``BaseException``s
    #: (``gpuwm.runplan.execute_plan``) can tell a stop from a failure
    #: without importing this module to ask.
    exit_code = INTERRUPT_EXIT_CODE

    def __init__(self, label: str, pid: int | None):
        super().__init__(f"interrupted during {label}")
        self.label = label
        self.pid = pid


def _physics_words(plan: dict) -> str:
    """The banner's physics clause: profile id, or one accurate sentence."""

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

    if gate.get("devices") is not None:
        return ("[devices] memory admission refused before download: prevent "
                "a card or pinned host allocation failure\n" + gate["verdict"])
    free = gate.get("free_bytes")
    free_words = ("" if free is None else
                  f", and the card has {free / (1024 ** 3):.2f} GiB "
                  "free right now")
    # THE ARITHMETIC UNDER A STREAMED REFUSAL.  The verdict above is one
    # sentence with one total in it; a user reporting that total has no
    # way to say which term produced it, and the 2.7.2 report of a
    # streamed run priced at 2.4x its resident one was exactly that -- a
    # total nobody could take apart.  The envelope's own named terms go
    # under it, so the screenshot carries the sum and its parts.
    phases = gate.get("phases")
    env = getattr(phases, "streamed", None)
    lines = getattr(env, "terms_lines", None)
    terms = "" if lines is None else "".join(
        f"\n    {line}" for line in lines())
    streamed_remedy = ("" if env is None or getattr(env, "rows", None) is not None
                       or not getattr(env, "tile_nx", None) else
                       "\n  remedy, streamed: a smaller [tiles] tile_nx/tile_ny "
                       "shrinks the buffer terms above; nbuffers = 1 halves "
                       "them; the fixed floors do not move with the tile")
    # HOST RAM ALONE REFUSED: a CPU preparation larger than this machine's
    # RAM, on a card that holds the forecast.  Freeing VRAM, a lighter
    # suite or a larger card leaves that number where it is.
    remedy = ("  remedy: re-size against this machine -- gpuwm domain ... "
              "(bare, it measures this card) -- or pick a lighter "
              "--physics-profile (the wizard's refusal ranks them by priced "
              "envelope), or free VRAM and re-run, or use a larger card\n")
    if (gate.get("preparation_refusal") is not None
            and not gate.get("card_refuse", True)):
        remedy = ("  remedy: re-size against this machine -- gpuwm domain "
                  "... (bare, it weighs this machine's RAM as well as its "
                  "card) -- or prepare on a machine with more RAM; freeing "
                  "VRAM or a larger card does not move host RAM\n")
    # A BEP+BEM price here is the every-column-urban bound (A176); the
    # prepared door reads the land cover, so the refusal says which it is.
    from gpuwm.core.preflight import (ExperimentMemoryEstimate,
                                      bem_workspace_bound_note)
    forecast = getattr(phases, "forecast", None)
    bem_note = (bem_workspace_bound_note(forecast)
                if isinstance(forecast, ExperimentMemoryEstimate) else None)
    bem_words = "" if bem_note is None else f"\n  note: {bem_note}"
    return (
        f"this configuration will not fit: {gate['verdict']}{free_words}."
        "  Refusing here, BEFORE the fetch stage downloads the forcing "
        f"data, rather than in preprocessing after it.{terms}{streamed_remedy}"
        f"{bem_words}\n"
        f"{remedy}"
        "  # gpuwm go CONFIG --no-memory-gate runs it anyway")


#: The planner's memory resources.  A refusal carrying one of these was
#: computed against a card's numbers; the others ("geometry", None) are
#: statements about the configuration alone.
_PLANNER_MEMORY_RESOURCES = frozenset({"vram", "host", "memory"})


def planner_gate(tree_road, *, card_seen: bool) -> tuple[bool, str | None]:
    """``(refuse, note)`` from the tile-planning report, for the go gate.

    Three things arrive on ``tree_road`` and only one of them is a refusal:

    * ``report_error`` -- the pricing REPORT raised (a ``TypeError`` in the
      walk, say).  Never a refusal: a defect in a report is not a fact
      about the tree, and the run proceeds under the pre-existing
      admission rule with the failure said out loud.  This function exists
      because a bare ``except Exception`` used to fill the same attribute
      a genuine refusal filled, and ``gpuwm go`` hard-refused runnable
      trees on the report's own exception (ENG-014).
    * a refusal whose ``refusal_resource`` is a MEMORY resource -- computed
      against the card's numbers.  Refuses when the card was measured;
      when it was not, "never refuse on a card we cannot see" holds, as the
      comment beside the no-probe branch has always promised.
    * a refusal about the configuration itself (``geometry``, or the
      walk's non-memory route refusal) -- refuses regardless of the card,
      because the run door's own walk would refuse it after the download.
    """
    report_error = getattr(tree_road, "report_error", None)
    if report_error:
        return False, ("tile-planning report failed (" + str(report_error)
                       + "); admission decided by the resident price alone")
    refusal = getattr(tree_road, "refusal", None)
    if not refusal:
        return False, None
    resource = getattr(tree_road, "refusal_resource", None)
    if not card_seen and resource in _PLANNER_MEMORY_RESOURCES:
        return False, ("native tile planner refused against an unmeasured "
                       "card, not refusing on a card we cannot see: "
                       + str(refusal))
    return True, "native tile planner refused this configuration: " + str(refusal)


def memory_gate(plan: dict, *, vram_gib: float | None = None,
                experiment=None) -> dict:
    """Price every phase of ``plan`` against this card, before the fetch.

    The whole point is WHERE this runs.  A configuration that cannot fit
    used to discover that in preprocessing -- after `gpuwm fetch` had
    pulled 81 GFS files down -- because the only memory estimate anyone
    computed described the forecast.  Both phases are priced here, at
    planning time, and the verdict names the binding phase and its
    number in one sentence.

    ``experiment`` may carry the already validated run-plan resolution,
    including explicit launch options; other callers load the config here.

    Returns a dict with ``verdict`` (always), ``refuse`` (the binding phase
    exceeds free memory or the native planner explicitly refused the
    configured execution road) and ``warn`` (it fits free VRAM but not the reserved
    budget -- said out loud, never blocking).
    """

    from gpuwm.core.preflight import (ReservePolicy, _load_experiment_any,
                                      config_forcing_source,
                                      config_forcing_schedule,
                                      DEFAULT_FORCING_INTERVAL_SECONDS,
                                      device_memory_probe_reason,
                                      device_memory_probe_subprocess,
                                      estimate_phases,
                                      profile_from_device_probe,
                                      unpriced_ingest_note)

    config = Path(plan["config"])
    source = config_forcing_source(config, priced_only=False)
    exp = (_load_experiment_any(config) if experiment is None else experiment)
    if plan.get("devices") is not None:
        from dataclasses import replace
        from gpuwm.core.devices import DeviceOptions
        exp = replace(exp, devices=DeviceOptions.from_mapping(plan["devices"]))
    if getattr(getattr(exp, "devices", None), "enabled", False):
        from gpuwm.core.devices import probe_devices, validate_device_count
        from gpuwm.core.devices_memory import devices_gate, include_preparation
        from gpuwm.core.preflight import estimate_devices, host_available_bytes
        measured = probe_devices()
        if measured is not None:
            validate_device_count(exp.devices, measured["visible_count"])
        interval, intervals = config_forcing_schedule(
            config, exp, fetch_cadence_hours=plan.get("cadence"))
        profiles = (None if measured is None else
                    {int(dev): profile_from_device_probe(row)
                     for dev, row in measured["cards"].items()})
        if len(exp.domains) > 1:
            # A split tree: every grid on the cards it runs on, the
            # pricing the tree runner repeats before its first restore.
            from gpuwm.core.devices_memory import estimate_devices_tree
            estimate = estimate_devices_tree(
                exp, vram_gib=vram_gib, forcing_intervals=intervals,
                source=source, profiles=profiles,
                forcing_interval_seconds=interval or DEFAULT_FORCING_INTERVAL_SECONDS)
        else:
            estimate = estimate_devices(
                exp, vram_gib=vram_gib, forcing_intervals=intervals, source=source,
                profiles=profiles,
                forcing_interval_seconds=interval or DEFAULT_FORCING_INTERVAL_SECONDS)
        budgets = ({dev: int(vram_gib * 2**30) for dev in exp.devices.device_ids()}
                   if vram_gib is not None else None if measured is None else
                   {int(dev): int(row["free_bytes"])
                    for dev, row in measured["cards"].items()})
        gate = devices_gate(estimate, budgets=budgets,
                            host_budget=host_available_bytes())
        include_preparation(
            gate, exp, source=source, budgets=budgets, host_budget=host_available_bytes(),
            forcing_intervals=intervals, forcing_interval_seconds=interval or
            DEFAULT_FORCING_INTERVAL_SECONDS, ingest_forcing_interval_seconds=interval,
            vram_gib=vram_gib)
        gate["device_probe"] = measured
        return gate
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
    # Actual staged case metadata outranks fetch hints; wider supplied
    # windows retain all of their boundary intervals after the run starts.
    forcing_interval, forcing_intervals = config_forcing_schedule(
        config, exp, fetch_cadence_hours=plan.get("cadence"))
    # [tiles] mode = "auto" with no pinned tiling is the planner's decision,
    # and the planner needs a card.  Build its Machine from the SUBPROCESS
    # probe already taken above rather than letting autoplan.Machine.detect
    # stand a CUDA context up in this process -- the same reason the probe
    # is a subprocess at all (0.486 GiB held for the whole run, MEASURED).
    machine = _planner_machine(probe, profile)
    phases = estimate_phases(
        exp, source=source, vram_gib=vram_gib, profile=profile,
        machine=machine, forcing_intervals=forcing_intervals,
        forcing_interval_seconds=(forcing_interval if forcing_interval is not None
                                  else DEFAULT_FORCING_INTERVAL_SECONDS),
        ingest_forcing_interval_seconds=forcing_interval)
    # A PREPARATION THE CARD CANNOT HOLD PREPARES ON THE CPU.  The door
    # this chain runs (gpuwm prep, backend auto) prices its preparation
    # before its first device allocation and moves it to the CPU when the
    # card's free memory cannot hold it (A65), so refusing the whole run
    # here for the preparation's card price refused a run that completes.
    # The gate prices the road the door will take: the CPU preparation's
    # host term is then what binds that phase.  An explicit cuda keeps
    # the card price, and the refusal.  The note says "may": this gate
    # reads the ingest estimate, while the door decides from the decoded
    # preparation's own price when it starts, and a price under the card's
    # free memory keeps that preparation on the card (the safe direction).
    preparation_on_cpu_note = None
    if (probe is not None and phases.ingest_priced
            and getattr(phases, "preprocess_backend", "cuda") == "auto"
            and phases.ingest_envelope_bytes > int(probe["free_bytes"])):
        preparation_on_cpu_note = (
            f"the preparation's CUDA price "
            f"{phases.ingest_envelope_bytes / 2**30:.2f} GiB exceeds the "
            f"card's {int(probe['free_bytes']) / 2**30:.2f} GiB free, so "
            "it may prepare on the CPU (--preprocess-backend auto decides "
            "from the preparation's own price when it starts)")
        phases = estimate_phases(
            exp, source=source, vram_gib=vram_gib, profile=profile,
            machine=machine, forcing_intervals=forcing_intervals,
            forcing_interval_seconds=(
                forcing_interval if forcing_interval is not None
                else DEFAULT_FORCING_INTERVAL_SECONDS),
            ingest_forcing_interval_seconds=forcing_interval,
            preprocess_backend="cpu")
    tree_road = getattr(phases, "tree_road", None)
    planner_refuse, planner_note = planner_gate(tree_road, card_seen=probe is not None)
    # THE PREPARATION'S HOST TERM, the one the wizard sizes against.  A CPU
    # preparation holds its whole working set in host RAM, which no device
    # comparison here sees; one larger than the machine's RAM is killed
    # after the download.  Host RAM does not depend on the card, so it is
    # read even when no card could be.
    preparation_refusal = preparation_warning = None
    weigh_preparation = getattr(phases, "host_preparation_refusal", None)
    if weigh_preparation is not None:
        preparation_host = getattr(phases, "host_ram_bytes", None)
        if preparation_host is None:
            from gpuwm.core.streaming import _host_total_bytes
            preparation_host = _host_total_bytes()
        preparation_refusal = weigh_preparation(preparation_host)
        preparation_warning = getattr(
            phases, "host_preparation_warning", lambda _host: None)(
                preparation_host)

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
        verdict = f"{phases.verdict(None)} ({probe_reason})" if probe_reason else phases.verdict(None)
        if planner_note and planner_note not in verdict:
            verdict += "; " + planner_note
        if preparation_refusal is not None:
            verdict += "; " + preparation_refusal
        return {"verdict": verdict,
                "refuse": planner_refuse or preparation_refusal is not None,
                "warn": False, "free_bytes": None,
                "probe_reason": probe_reason, "phases": phases, "device_probe": probe,
                "planner_report_error": getattr(tree_road, "report_error", None),
                "preparation_refusal": preparation_refusal,
                "preparation_warning": preparation_warning,
                "card_refuse": planner_refuse}
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
        forcing_interval_seconds=forcing_interval,
        profile=profile)
    peak = phases.peak_envelope_bytes
    verdict = phases.verdict(budget)
    # A resident reference number cannot admit a configuration for which the
    # native tree planner explicitly refused the configured execution road.
    # A planning REPORT that died is not such a refusal (planner_gate).
    refuse = peak > free or planner_refuse
    card_refuse = refuse
    if preparation_on_cpu_note is not None:
        verdict += "; " + preparation_on_cpu_note
    if planner_note and planner_note not in verdict:
        verdict += "; " + planner_note
    if preparation_refusal is not None:
        refuse = True
        verdict += "; and " + preparation_refusal
    # THE PINNED STORE IS A REFUSAL TOO, and only for a streamed run: the
    # domain lives in host RAM there, so a config whose store cannot be
    # page-locked dies at attach -- after the download, which is exactly
    # what this gate exists on this side of.  Priced only when the host
    # total could be read; unknown host RAM never refuses.
    #
    # The domain's lateral forcing series is part of that claim
    # (``boundary_table_bytes``): it stays in host RAM for the whole run
    # beside the store, and an estimate that left host tables out is how a
    # 0.93 GiB figure admitted a run that committed 125 GB on a 96 GB box.
    #
    # The comparison is the estimate's own (``streamed_host_refusal``), the
    # one ``gpuwm domain`` sizes against and ``gpuwm check`` fails on, so
    # those doors cannot emit or pass a configuration this gate refuses.
    streamed_host_refusal = (phases.streamed_host_refusal()
                             if phases.streamed_forecast else None)
    if streamed_host_refusal is not None:
        refuse = True
        verdict += "; and " + streamed_host_refusal
    return {
        "verdict": verdict,
        "refuse": refuse,
        "warn": peak > budget or not phases.ingest_priced,
        "free_bytes": free,
        "budget_bytes": budget,
        # Present on every return of this function, so a caller never
        # has to distinguish "the key is absent" from "there was no
        # reason"; None means the probe answered.
        "probe_reason": None,
        "phases": phases,
        "device_probe": probe,
        "planner_report_error": getattr(tree_road, "report_error", None),
        # The host-RAM refusal of a CPU preparation, and whether anything
        # about the CARD refused beside it: the remedy differs, because
        # no VRAM lever moves host RAM.  The warning is the admitted case
        # whose estimated peak, but not its floor, is over the RAM.
        "preparation_refusal": preparation_refusal,
        "preparation_warning": preparation_warning,
        "card_refuse": card_refuse,
        "preparation_on_cpu": preparation_on_cpu_note,
    }


def _planner_machine(probe, profile=None):
    """A :class:`tilestream.autoplan.Machine` from this gate's own probe.

    The arithmetic lives in :func:`gpuwm.core.streaming.planner_machine`,
    beside the envelope it feeds, because ``gpuwm check`` needs the same
    Machine built from a DECLARED budget rather than a probe and two
    copies of this would be two answers about one card.

    BOTH HALVES OF THE PROBE, not just its free VRAM.  The tree admission
    prices its non-pool terms off ``machine.device_profile``, so a machine
    built from the probe's byte count alone made this gate weigh a
    different envelope from the run door, whose machine is
    ``Machine.detect`` and always carries the card's profile.  The profile
    is the one this gate already read out of the same probe payload.
    """
    from gpuwm.core.streaming import planner_machine

    return planner_machine(
        vram_bytes=None if probe is None else int(probe["free_bytes"]),
        name="gpuwm go probe", device_profile=profile)


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
    statement about a RUN, and the two answers differ accurately.
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


def _require_forecast_device() -> None:
    """Prove the GPU can run kernels before a launch downloads its inputs."""
    from gpuwm.doctor import _cuda_headers_check
    from gpuwm.explain import layered
    from gpuwm.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu

    checked = _cuda_headers_check()
    if checked.status == "info" and no_local_gpu():
        # The probe did not run: this environment forbids the process the
        # card, and "info" is the not-judged verdict the headers check
        # gives under the switch.  Said by name, because "GPU readiness
        # is info: device not touched" read as a card the probe had
        # misjudged, and a launch from a shell that exported the switch
        # was retried as a flake.  A measured verdict ("missing") keeps
        # its own remedy below whatever the environment says.
        raise GoRefusal(layered(
            f"{NO_LOCAL_GPU_ENV} is set in this environment, so this "
            "process may not open the local card and no forecast can run. "
            f"Next: unset {NO_LOCAL_GPU_ENV} and launch again",
            checked.detail))
    if checked.status != "verified":
        raise GoRefusal(layered(
            f"GPU readiness is {checked.status}: {checked.brief or checked.detail}. "
            f"Next: {checked.action or 'gpuwm doctor --explain'}",
            checked.detail + ("\n" + checked.remedy if checked.remedy else "")))


def failed_line(fields: dict, *, explain: bool) -> str:
    """What ``gpuwm go`` prints for a plan run's ``failed`` event, before a stage's log tail.

    The refusal's own text, then ``Next:`` and the event's ``remedy``
    when the text does not already state that remedy.  A refusal that
    ends its action half with ``remedy: <command>`` gives its event the
    same line (:func:`gpuwm.runplan.stated_remedy`), so appending it
    printed it twice: a chain with no built ``gfs_grib2_bridge`` read
    ``remedy: gpuwm setup`` and then ``Next: gpuwm setup``.
    """

    from gpuwm.runplan import stated_remedy

    text = str(fields.get("message", "launch failed"))
    line = render(text, explain=explain)
    remedy = fields.get("remedy")
    if remedy and " ".join(str(remedy).split()) != " ".join((stated_remedy(text) or "").split()):
        line += f" Next: {remedy}"
    return line


def _registered_launch(args, *, config: Path, payload: dict) -> int:
    """Adapt the human command to the existing executable run-plan contract."""
    import contextlib
    import hashlib
    import io

    from gpuwm import runplan, source_cli

    from gpuwm.core.preflight import config_forcing_source

    prepared_root = getattr(args, "prepared_root", None)
    declared_inputs = "case_data" in payload and prepared_root is None
    if declared_inputs and args.data_dir is not None:
        raise GoRefusal(
            "This config names its input files directly, so --data-dir is unused. "
            "Next: omit --data-dir; change [case_data].forcing to use different files.")
    case_root = Path(args.outdir) if args.outdir is not None else (
        config.parent / f"{config.stem}-go")
    fetch = payload.get("fetch") or {}
    acquires = (not declared_inputs and prepared_root is None
                and {"source", "cycle"} <= fetch.keys())
    if acquires:
        checked_config_fetch_cycle(
            fetch, start_time=payload.get("experiment", {}).get("start_time"))
    from gpuwm.fetch import pinned_host

    flag_transport = getattr(args, "transport", None)
    # Breakage it prevents: a host pin on a route with no fetch stage
    # would be accepted and then asked of nothing, so the run would not
    # use the host its command names.  ``auto`` pins no host, so it asks
    # nothing of a fetch stage and passes; refusing it turned away the
    # default spelled out on every [case_data] and --prepared-root run.
    if pinned_host(flag_transport) is not None and not acquires:
        raise GoRefusal(
            f"--transport {flag_transport} pins the host of a download "
            "route's fetch stage, and this run takes its inputs from "
            "[case_data] or an existing prepared bundle, which have no host "
            "to pin. Next: omit --transport.")
    pinned, pinned_from = (pinned_transport(fetch, flag_transport) if acquires
                           else (None, None))
    # A source with no acquisition route downloads nothing, so there is
    # no managed download cache to compute for it.  Computing one anyway
    # made `[fetch].source_root` -- the key `gpuwm domain --data-dir`
    # writes, and the only root a config can carry by itself -- dead on
    # this door: `resolve_source_root` takes the run option whenever it
    # is set, and this one was always set.  A local-input config then
    # refused against a managed downloads path the reader never named.
    # Only an explicit `--data-dir` overrides the config here; the
    # default leaves the config's own root to speak.
    local_input = acquires and bool(
        runplan.drivability_for(fetch.get("source")).get("requires_source_root"))
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
    elif local_input:
        data_dir = None
    elif acquires:
        data_dir = managed_download_dir(
            case_root, pin_request(config_fetch_request(payload), pinned))
    else:
        data_dir = case_root / "data"
    if (not declared_inputs and prepared_root is None and data_dir is not None
            and case_root.resolve() == data_dir.resolve()):
        raise GoRefusal("--outdir and --data-dir must differ. "
                        "Next: omit --data-dir to use the shared download cache.")
    section = render_section_value(getattr(args, "render_section", None))
    admit_render_products(args.render_products, section=section)
    stamp = run_stamp_module.run_stamp_enabled(args)
    cycle = (payload.get("experiment", {}).get("start_time") if declared_inputs or prepared_root is not None
             else fetch.get("cycle"))
    output = run_stamp_module.resolve(case_root, init=cycle, enabled=stamp,
                                     create=False)
    options = {"geog_root": (None if args.geog_root is None
                             else str(Path(args.geog_root).resolve())),
               "render_products": (args.render_products if args.render_products is not None
                                   else DEFAULT_RENDER_PRODUCTS)}
    if section is not None:
        options["render_section"] = section
    keep = getattr(args, "keep_checkpoints", None)
    if keep is not None:
        options["keep_checkpoints"] = keep
    if flag_transport is not None and acquires:
        # Only a route with a fetch stage carries it: there ``auto`` must
        # reach run-plan to unpin a table's host.  A route without one
        # is only here with ``auto``, which asks for what no flag asks
        # for, and the experiment route has no transport option at all.
        options["transport"] = flag_transport
    posting_flags = _posting_options(args)
    if posting_flags and not acquires:
        # Breakage it prevents: a posting rule on a run with no fetch
        # stage would be accepted and read by nothing.
        raise GoRefusal(
            f"{', '.join(sorted(posting_flags))} set how the fetch stage "
            "waits for its source, and this run takes its inputs from "
            "[case_data] or an existing prepared bundle. Next: omit them.")
    options.update({key: value for key, value in (
        ("as_posted", False if getattr(args, "whole_cycle", False) else None),
        ("late_after_minutes", getattr(args, "late_after_minutes", None)),
    ) if value is not None})
    if not declared_inputs and prepared_root is None and data_dir is not None:
        options["data_dir"] = str(data_dir.resolve())
    elif prepared_root is not None and args.data_dir is not None:
        options["data_dir"] = str(Path(args.data_dir).resolve())
    if getattr(args, "devices", None) is not None:
        options["devices"] = args.devices
    for key in ("restart", "prepared_root", "wps_namelist"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = str(Path(value).resolve())
    if getattr(args, "supplement", None):
        from gpuwm.launch_supplements import bindings
        options["supplement"] = bindings(args.supplement, base=Path.cwd())
    raw = {"schema": runplan.PLAN_SCHEMA, "name": config.stem,
           "route": "experiment" if declared_inputs else "prepared",
           "config": {"path": str(config.resolve())},
           "output_root": str(output.resolve()), "run_options": options}
    identity = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    plan = runplan.build_plan(raw, source=f"gpuwm go {config}",
                             base_dir=config.resolve().parent, sha256=identity)
    # Same physics, moving-statics and streamed-route checks as run-plan.
    # Do this before output allocation, network access or runtime loading.
    resolution, exp, data = runplan.resolve_plan(plan, require_inputs=False)
    bundle = runplan._existing_prepared_bundle(plan)
    if bundle is None:
        acquisition = runplan.declared_forcing_fetch(payload, data)
        if acquisition is not None:
            raw["fetch"] = {"args": acquisition}
            identity = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
            plan = runplan.build_plan(raw, source=f"gpuwm go {config}",
                base_dir=config.resolve().parent, sha256=identity)
    source = (bundle["source"] if bundle is not None else
              config_forcing_source(config, priced_only=False) or "declared inputs")
    stages = (["verify prepared bundle", "restore" if options.get("restart") else "forecast"]
              if bundle is not None else
              ["prepare", "forecast"] if declared_inputs and plan.fetch_arguments is None
              else ["fetch", "prepare", "forecast"])
    if str(options["render_products"]).strip().lower() != "none":
        stages.append("render")
    print(f"go: {source}, {len(exp.domains)} domain(s); " + " -> ".join(stages))
    pinned_line = (transport_note(pinned, pinned_from, fetch.get("transport"))
                   if bundle is None else None)
    if pinned_line is not None:
        print(pinned_line)
    if getattr(getattr(exp, "devices", None), "enabled", False):
        from gpuwm.core.devices import describe_split
        print("go: " + describe_split(exp))
    if args.dry_run:
        if getattr(getattr(exp, "devices", None), "enabled", False):
            gate = memory_gate({"config": config, "source": source,
                                "cadence": fetch.get("cadence")}, experiment=exp)
            print(gate["verdict"])
            if gate["refuse"]:
                raise GoRefusal(memory_refusal_text(gate))
        if bundle is not None:
            for warning in resolution["warnings"]:
                print("warning: " + warning["action"], file=sys.stderr)
        print(f"Output: {output}")
        print("Run: " + printable(["gpuwm", "go", str(config),
              *([] if args.outdir is None else ["--outdir", str(args.outdir)]),
              *([] if args.data_dir is None else ["--data-dir", str(args.data_dir)]),
              *([] if args.geog_root is None else ["--geog-root", str(args.geog_root)]),
              *([] if flag_transport is None else ["--transport", flag_transport]),
              *(token for binding in options.get("supplement", ())
                for token in ("--supplement", binding)),
              *(token for key in ("restart", "prepared_root", "wps_namelist")
                if getattr(args, key, None) is not None
                for token in ("--" + key.replace("_", "-"), str(getattr(args, key)))),
              *([] if args.render_products is None else ["--products", args.render_products]),
              *([] if section is None else [f"--section={section}"]),
              *([] if keep is None else ["--keep-checkpoints", str(keep)]),
              *(["--run-stamp", "off"] if not stamp else []),
              *(["--no-memory-gate"] if args.no_memory_gate else [])]))
        return 0
    capabilities.require_for_command("go")
    _require_forecast_device()
    if bundle is None and not getattr(args, "no_memory_gate", False):
        gate = memory_gate({"config": config, "source": source,
                            "cadence": fetch.get("cadence")}, experiment=exp)
        print(f"go: memory -- {gate['verdict']}")
        if gate["refuse"]:
            raise GoRefusal(memory_refusal_text(gate))
        if gate["warn"]:
            print("warning: memory admission has limited headroom or an "
                  f"unmeasured phase; see gpuwm check {_quote(config)} --explain.",
                  file=sys.stderr)
        if gate.get("preparation_warning"):
            print(f"warning: {gate['preparation_warning']}.", file=sys.stderr)
    from gpuwm.geog_assets import default_geog_root
    geog_root = (data.geog_root if data is not None
                 else Path(args.geog_root) if args.geog_root is not None
                 else default_geog_root())
    if bundle is None:
        geography = geography_refusal(geog_root)
        if geography is not None:
            raise GoRefusal(geography)
    else:
        print("go: existing prepared inputs; restore memory and payload checks "
              "run in the simulation runner")
    if str(args.render_products or "").strip().lower() != "none":
        missing = render_extra_missing()
        if missing is not None:
            from gpuwm.explain import layered
            raise GoRefusal(layered(
                "The Rust renderer is unavailable. Next: gpuwm setup, then "
                "repeat this command. Use --products none to run only the forecast.",
                missing))
    output = run_stamp_module.resolve(case_root, init=cycle, enabled=stamp,
                                     create=True)
    raw["output_root"] = str(output.resolve())
    identity = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    plan = runplan.build_plan(raw, source=f"gpuwm go {config}",
                             base_dir=config.resolve().parent, sha256=identity)
    log_path = output / "launch.log"
    terminal, errors = sys.stdout, sys.stderr
    explain = explain_enabled(args)
    current = {"stage": "preflight", "started": time.monotonic()}
    stage_tail = {"stdout": "", "stderr": ""}
    from gpuwm.prep_progress import PrepProgress, step_record
    from gpuwm.progress import prep_record_event
    from gpuwm.command_output import text_chunks
    prep_progress = PrepProgress()

    class HumanEvents(runplan.EventStream):
        def emit(self, event, **fields):
            record = super().emit(event, **fields)
            if event == "stage_started":
                label = fields.get("phase") or fields["stage"]
                current.update(stage=label, started=time.monotonic())
                stage_tail.update(stdout="", stderr="")
                print(f"go: {label}", file=terminal, flush=True)
            elif event == "first_products_ready":
                print(f"go: first pictures ready in {output}", file=terminal, flush=True)
            elif event == "failed":
                message = failed_line(fields, explain=explain)
                if fields.get("error_class") == "StageExitError" and not explain:
                    # Less the preparer's raw step lines, which were said as
                    # steps while it ran and filled this tail; the log keeps them.
                    lines = (_said_lines(stage_tail["stderr"].rstrip())
                             or _said_lines(stage_tail["stdout"].rstrip()))[-8:]
                    if lines:
                        message += "\n" + "\n".join(lines)
                print(f"go: {message}\nDetails: {log_path}", file=errors, flush=True)
            elif event == "warning" and fields.get("code") == "preparation_progress":
                # A preparation step heard in this process, said as a
                # preparer program's step line is: it is not a warning, and
                # printed as one it read "warning: Start state and boundaries".
                with output_lock:
                    said = prep_progress.event(fields.get("preparation"))
                if said is not None:
                    print(f"go: {said}", file=terminal, flush=True)
            elif event == "warning":
                from gpuwm.explain import split
                said = f"warning: {split(str(fields.get('message', '')))[0]}"
                if _first_showing(said):
                    print(said, file=errors, flush=True)
            elif event == "completed":
                print(f"go: complete. Output: {output}", file=terminal, flush=True)
            return record

    output_lock = threading.RLock()

    class LaunchOutput(io.TextIOBase):
        # Every line goes to the log and each advisory reaches the
        # terminal once, so a stage relay writing here prints them all.
        relays_advisories_once = True

        def __init__(self, destination, log, channel):
            self.destination, self.log, self.channel = destination, log, channel
            self.pending = ""

        def write(self, text):
            steps = []
            with output_lock:
                for chunk in text_chunks(text):
                    self.log.write(chunk)
                    stage_tail[self.channel] = (stage_tail[self.channel] + chunk)[-32768:]
                    if explain:
                        self.destination.write(chunk)
                    self.pending += chunk
                    while "\n" in self.pending:
                        line, self.pending = self.pending.split("\n", 1)
                        record = step_record(line)
                        if record is not None:
                            steps.append(record)
                        elif (not explain
                              and line.lstrip().lower().startswith(("warning:", "note:"))
                              and _first_showing(line)):
                            print(line.strip(), file=errors, flush=True)
                    self.pending = self.pending[-32768:]
            # A preparer program's step (the staged route's `gpuwm prep`
            # adapter) goes on the run's stream, where the page reads it;
            # HumanEvents says it in the terminal as it is written.  It was
            # printed here and never reached the stream.  Outside the lock,
            # which HumanEvents takes to say it.  A record the stream cannot
            # take is dropped: a step's telemetry never fails the preparation
            # whose output this is.
            for record in steps:
                try:
                    events.emit("warning", **prep_record_event(record))
                except Exception:  # noqa: BLE001 - see above
                    pass
            return len(text)

        def flush(self):
            with output_lock:
                if not self.log.closed:
                    self.log.flush()
                if not self.destination.closed:
                    self.destination.flush()

    finished = threading.Event()

    def heartbeat():
        while not finished.wait(HEARTBEAT_SECONDS):
            elapsed = _elapsed_words(time.monotonic() - current["started"])
            label = prep_progress.label if prep_progress.active else current["stage"]
            print(f"go: {label} ({elapsed})", file=terminal, flush=True)

    from gpuwm.command_output import AdapterOutputError, DiagnosticLog
    try:
        with HumanEvents(output / runplan.EVENTS_FILENAME, mirror=None) as events:
            with DiagnosticLog(log_path.open("a", encoding="utf-8", buffering=1)) as log:
                worker = threading.Thread(target=heartbeat, name="go-progress", daemon=True)
                worker.start()
                try:
                    with contextlib.redirect_stdout(LaunchOutput(terminal, log, "stdout")), \
                            contextlib.redirect_stderr(LaunchOutput(errors, log, "stderr")), \
                            source_cli.redirect_adapter_output(sys.stdout, sys.stderr):
                        code = runplan.execute_plan(plan, events=events)
                        return 74 if log.failure is not None else code
                finally:
                    finished.set()
                    worker.join()
    except (AdapterOutputError, BrokenPipeError) as error:
        try:
            print(f"go: diagnostic output failed: {error}", file=errors)
        except (OSError, ValueError):
            pass
        return 74


def chain_io_root(requested: Path, *, downloads: bool = True, depth: int | None = None) -> Path:
    """The spelling a chain should use for ``requested``.

    On Windows, a folder so deep that the files a chain writes below it
    would pass the 260-character path limit is handed over in its
    extended spelling, which every Python and native call below it opens
    at any length.  Any other folder comes back unchanged, so the paths
    a run prints stay the ones the reader typed.  ``downloads`` says
    whether the chain keeps its request cache below ``requested``, which
    is where it writes deepest; ``depth``, when given, is that request's
    own measured depth (:func:`download_depth`).
    """
    from gpuwm.filesystem_paths import (CHAIN_DEPTH_BUDGET, RUN_TREE_DEPTH_BUDGET,
                                        deep_io_path)

    budget = CHAIN_DEPTH_BUDGET if downloads else RUN_TREE_DEPTH_BUDGET
    return deep_io_path(requested, budget if depth is None else max(budget, depth))


#: ``downloads/`` plus a request key's ``-<n>`` retry suffix, the slash
#: before the object, and a ``.part`` staging suffix: what the managed
#: download cache adds around an object's own name.
_CACHE_FRAME = len("downloads/") + len("-99") + 1 + len(".part")


def download_depth(fetch_table: dict) -> int | None:
    """Characters this request's download writes below the output folder.

    A table route knows every object it will move before a byte moves, and
    their names differ by source: GDPS and ICON name an object in more than
    80 characters where GFS uses 37.  The deepest is
    ``downloads/<64-hex key>/<object>.part``, so a folder is spelled in the
    extended form exactly when that request's own files would pass the
    limit.  None when the request is not a table route or cannot be
    planned here, which leaves the measured chain budget in charge.
    """
    from gpuwm import fetch_routes

    try:
        _request, source, cycle, _area, key = _managed_download_request(fetch_table)
        if source not in fetch_routes.route_ids():
            return None
        plan = fetch_routes.resolve_request(
            source, cycle=cycle, hours=int(fetch_table.get("hours", 0)),
            start_hour=int(fetch_table.get("forecast_start_hour", 0) or 0),
            member=fetch_table.get("member"))
    except (GoRefusal, KeyError, TypeError, ValueError, RuntimeError):
        return None
    if not plan.objects:
        return None
    return _CACHE_FRAME + len(key) + max(len(obj.relpath) for obj in plan.objects)


def _extend_outdir(args, config: Path, payload: dict) -> None:
    """Hand a deep output folder to the chain in its extended spelling."""
    requested = (Path(args.outdir) if getattr(args, "outdir", None) is not None
                 else config.parent / f"{config.stem}-go")
    downloads = (isinstance(payload.get("fetch"), dict) and "case_data" not in payload
                 and getattr(args, "prepared_root", None) is None
                 and getattr(args, "data_dir", None) is None)
    depth = download_depth(payload["fetch"]) if downloads else None
    spelled = chain_io_root(requested, downloads=downloads, depth=depth)
    if spelled != Path(requested):
        args.outdir = spelled


def _posting_options(args) -> dict[str, object]:
    """The posting flags ``gpuwm go`` was given (DESIGN A136 3.1)."""

    said = {}
    if getattr(args, "whole_cycle", False):
        said["--whole-cycle"] = True
    if getattr(args, "late_after_minutes", None) is not None:
        said["--late-after-minutes"] = args.late_after_minutes
    if getattr(args, "no_probe", False) and not getattr(args, "readiness", False):
        raise GoRefusal("--no-probe belongs to --readiness.")
    return said


def _flag_cycle(args, payload: dict) -> str | None:
    """``gpuwm go --cycle``, checked: ``latest`` or a concrete cycle, or None.

    Breakage the refusal prevents: a cycle on a run whose inputs are named
    files, an existing prepared bundle or a checkpoint would be accepted
    and read by nothing, and the run would start at those inputs' own
    time while its command named another.
    """

    value = getattr(args, "cycle", None)
    if value is None:
        return None
    value = str(value).strip()
    fetch_table = payload.get("fetch")
    if (not isinstance(fetch_table, dict)
            or not {"source", "cycle"} <= fetch_table.keys()
            or "case_data" in payload
            or any(getattr(args, key, None) is not None
                   for key in ("prepared_root", "restart", "wps_namelist"))):
        raise GoRefusal(
            f"--cycle {value} names the cycle a download route fetches, and "
            "this run takes its inputs from [case_data], an existing "
            "prepared bundle or a checkpoint, whose times a cycle cannot "
            "move. Next: omit --cycle.")
    if value.lower() == "latest":
        return "latest"
    from gpuwm.fetch import parse_cycle

    try:
        return parse_cycle(value, str(fetch_table["source"])).strftime("%Y-%m-%dT%H")
    except ValueError as error:
        raise GoRefusal(str(error)) from error


def _at_flag_cycle(args, config: Path, payload: dict, cycle: str
                   ) -> tuple[Path, dict]:
    """The config re-timed to ``gpuwm go --cycle`` (DESIGN A136 3.1, 3.7).

    ``latest`` is answered first by the fetch's own resolver under this
    run's posting rule (as posted unless ``--whole-cycle`` or the config's
    ``as_posted = false``), so the run and its receipt name a concrete
    cycle.  A cycle other than the config's own is published by
    :func:`gpuwm.companion_setups.retime_to_cycle` under
    ``<outdir>/cycles/<cycle>/`` with the route files rendered again, and
    the launch continues on that file with the original's output folder,
    so every route (the stage composer, the native chains, run-plan's)
    runs the re-timed config the way it runs any other.
    """

    import tomllib

    from gpuwm import runplan
    from gpuwm.companion_setups import retime_to_cycle
    from gpuwm.fetch import parse_cycle

    fetch_table = payload["fetch"]
    source = str(fetch_table["source"])
    if cycle == "latest":
        from gpuwm.cli import _join_negative_coordinates

        pinned, _from = pinned_transport(fetch_table, getattr(args, "transport", None))
        hints = pin_request(config_fetch_request(payload), pinned)
        hints["cycle"] = "latest"
        if getattr(args, "whole_cycle", False):
            hints["as_posted"] = False
        arguments = _join_negative_coordinates(
            runplan._fetch_arguments_from_hints(hints, out=Path("latest")))
        try:
            arguments, _resolutions, _warnings = runplan.resolve_fetch_cycle(arguments)
        except (ValueError, RuntimeError) as error:
            raise GoRefusal(f"--cycle latest: {error}") from error
        cycle = runplan._parse_fetch_arguments(arguments).cycle
        print(f"go: --cycle latest is {cycle} for {source}")
    moment = parse_cycle(cycle, source)
    if moment == parse_cycle(str(fetch_table["cycle"]), source):
        return config, payload
    if getattr(args, "outdir", None) is None:
        # The original's own output folder, so the run lands where a run
        # of this config always lands, in its own cycle-stamped folder.
        args.outdir = config.parent / f"{config.stem}-go"
    out = Path(args.outdir) / "cycles" / cycle / config.name
    try:
        result = retime_to_cycle(config, moment, out)
    except ValueError as error:
        raise GoRefusal(f"--cycle {cycle}: {error}") from error
    print(f"go: --cycle {cycle}: {config} re-timed to start "
          f"{result['start_time'].replace('T', ' ')} UTC, as {out}")
    args.config = out
    return out, tomllib.loads(out.read_text(encoding="utf-8"))


def _go_readiness(payload: dict, options: dict, pinned: str | None, *,
                  no_probe: bool) -> int:
    """``gpuwm go CONFIG --readiness``: the config's window, answered, run nothing."""

    from gpuwm import runplan as plans
    from gpuwm import source_readiness as readiness
    from gpuwm.fetch import readiness_for_fetch

    hints = pin_request(config_fetch_request(payload), pinned)
    for key in ("as_posted", "late_after_minutes"):
        if options.get(key) is not None:
            hints[key] = options[key]
    if hints.get("as_posted") is False:
        hints.pop("late_after_minutes", None)
    arguments = plans._fetch_arguments_from_hints(hints, out=Path("readiness"))
    from gpuwm.cli import _join_negative_coordinates

    parsed = plans._parse_fetch_arguments(_join_negative_coordinates(arguments))
    try:
        document, code = readiness_for_fetch(parsed, no_probe=no_probe)
    except ValueError as error:
        raise GoRefusal(f"--readiness: {error}") from error
    readiness.print_document(document)
    print(f"go: readiness {document['state']}"
          + (f" ({document['refusal']})" if document.get("refusal") else "")
          + (f"; expected ready at {document['expected_ready_at']}"
             if document.get("expected_ready_at") else ""), file=sys.stderr)
    return code


def go_main(args, *, observer=None) -> int:
    """Launch through the source's existing native preparation chain.

    ``--no-memory-gate`` is held for the whole chain, so the forecast
    runner go starts (a subprocess, or hosted here) skips the same envelope
    go's own gate skipped.  It used to stop at go's gate, and the runner
    then refused that envelope again after the fetch and the preparation.
    """
    from gpuwm.core.resident_admission import memory_gate_override

    with memory_gate_override(getattr(args, "no_memory_gate", False)), \
            _each_advisory_once():
        return _go_launch(args, observer=observer)


def _go_launch(args, *, observer=None) -> int:
    """:func:`go_main`'s body, run with each advisory shown once."""
    import tomllib

    from gpuwm.runplan import prepared_chain_for_source

    config = Path(args.config)
    if not config.is_file():
        # A folder is not "missing": saying so sent the reader looking for
        # a path that is right there.
        what = ("is a folder, not a configuration file" if config.is_dir()
                else "does not exist")
        raise GoRefusal(f"{config} {what}. Next: gpuwm domain --help")
    payload = tomllib.loads(config.read_text(encoding="utf-8"))
    cycle = _flag_cycle(args, payload)
    if getattr(args, "readiness", False) or getattr(args, "no_probe", False):
        # Every route answers the same question here, before any gate or
        # folder: the window the config's [fetch] table asks for, at
        # --cycle when it names one (``latest`` is the readiness
        # document's own cycle_basis).
        _posting_options(args)
        if cycle is not None:
            payload = {**payload, "fetch": {**payload["fetch"], "cycle": cycle}}
        fetch_table = payload.get("fetch")
        if (not isinstance(fetch_table, dict)
                or not {"source", "cycle"} <= fetch_table.keys()):
            # Breakage it prevents: a config with no download would be
            # answered for a window nobody fetches.
            raise GoRefusal(
                "--readiness answers for the window a config fetches, and "
                "this config has no [fetch] source and cycle.")
        pinned, _from = pinned_transport(fetch_table,
                                         getattr(args, "transport", None))
        return _go_readiness(payload, {
            "as_posted": False if getattr(args, "whole_cycle", False) else None,
            "late_after_minutes": getattr(args, "late_after_minutes", None)},
            pinned, no_probe=getattr(args, "no_probe", False))
    if cycle is not None:
        config, payload = _at_flag_cycle(args, config, payload, cycle)
    _extend_outdir(args, config, payload)
    if ("case_data" in payload or getattr(args, "prepared_root", None) is not None
            or getattr(args, "restart", None) is not None
            or getattr(args, "wps_namelist", None) is not None):
        return _registered_launch(args, config=config, payload=payload)
    fetch = payload.get("fetch")
    source = fetch.get("source") if isinstance(fetch, dict) else None
    if not source:
        raise GoRefusal("The config has no [fetch] table with a source. "
                        "Next: gpuwm domain --help")
    local_root = getattr(args, "data_dir", None) or fetch.get("source_root")
    chain = prepared_chain_for_source(str(source), source_root=local_root)
    if chain != "prepared:go":
        return _registered_launch(args, config=config, payload=payload)
    if getattr(args, "supplement", None):
        from gpuwm.launch_supplements import validate_route
        validate_route(args.supplement, chain=chain)
    return _go_prepared_main(args, observer=observer)


def _disk_admission(plan: dict, args) -> None:
    """Refuse, before the run folder is claimed or a byte is fetched, a run its disks cannot hold.

    The admission ``gpuwm run-plan`` gives every run it starts
    (:func:`gpuwm.runplan.disk_admission_refusal`), asked here because
    this chain, typed as ``gpuwm go``, enters no run plan.  The run is
    described to it as the plan it is: this config on the prepared route,
    writing into this run's folder, drawing the products this chain draws
    and keeping the checkpoint sets this chain keeps, with the download
    priced from the very arguments the fetch stage is about to be handed,
    and any frame stream its preparation stages measured beside the
    folder that preparation writes.  A stream that may not fit is said
    here, before the download, and the chain goes on.
    """

    import hashlib
    import tomllib

    from gpuwm import runplan
    from gpuwm.experiment import load_experiment
    from gpuwm.resume import DEFAULT_KEEP_CHECKPOINTS

    config = Path(plan["config"]).resolve()
    keep = getattr(args, "keep_checkpoints", None)
    options = {"keep_checkpoints": DEFAULT_KEEP_CHECKPOINTS if keep is None else keep,
               "render_products": plan.get("render_products") or None,
               # The line its sections are cut along: a plan naming an
               # xsec: product and no line is refused when it is built.
               "render_section": plan.get("render_section") or None}
    raw = {"schema": runplan.PLAN_SCHEMA, "name": config.stem, "route": "prepared",
           "config": {"path": str(config)}, "output_root": str(Path(plan["root"]).absolute()),
           "run_options": options}
    described = runplan.build_plan(
        raw, source=f"gpuwm go {config}", base_dir=config.parent,
        sha256=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest())
    refusal = runplan.disk_admission_refusal(
        described, load_experiment(config),
        raw=tomllib.loads(config.read_text(encoding="utf-8")), data=None,
        fetch_arguments=fetch_command(plan)[4:], run_dir=Path(plan["root"]),
        # The download folder is the request's own managed cache unless
        # --data-dir named one: everything in a managed cache is this
        # request's download, a half-finished one included.
        download_keyed=getattr(args, "data_dir", None) is None,
        prep_root=Path(plan["prepared"]), warn=_scratch_caution)
    if refusal is not None:
        raise GoRefusal(refusal)


def _scratch_caution(message: str, detail: str, folder: str | None) -> None:
    """Say, before the download, that the preparation's frame stream may not fit its disk."""

    print(f"go: WARNING -- {message}" + (f"  {detail}" if detail else ""))


def _go_prepared_main(args, *, observer=None) -> int:
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

    A multi-domain config runs here: the forecast stage dispatches to
    ``gpuwm.prepared_domain_tree_forecast`` when the config declares a
    ladder, and to the single-domain runner when it declares one
    domain. Other native preparation routes are dispatched by
    :func:`go_main` before entering this stage composer.
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
    section = render_section_value(getattr(args, "render_section", None))
    admit_render_products(getattr(args, "render_products", None),
                          section=section)
    plan = plan_from_config(args.config,
                            outdir=getattr(args, "outdir", None),
                            data_dir=getattr(args, "data_dir", None),
                            render_products=getattr(
                                args, "render_products", None),
                            render_section=section,
                            run_stamp=run_stamp_module.run_stamp_enabled(args),
                            transport=getattr(args, "transport", None),
                            devices=getattr(args, "devices", None),
                            as_posted=(False if getattr(args, "whole_cycle", False)
                                       else None),
                            late_after_minutes=getattr(
                                args, "late_after_minutes", None))
    bridge = resolve_bridge()
    geog_root = (Path(args.geog_root) if getattr(args, "geog_root", None)
                 else default_geog_root())

    # The dry run names the predicted folder; the real run reads the
    # path again after the claim below.
    manifest = front_door_manifest(plan)
    cycle_stamp = _cycle_stamp(plan["cycle"])

    if getattr(args, "dry_run", False):
        print(f"go: {plan['config']} -- source {plan['source']}, cycle "
              f"{plan['cycle']}, {plan['hours']} h, "
              f"{_physics_words(plan)}{_lead_note(plan)}")
        print(_run_folder_note(plan))
        pinned = transport_note(plan.get("transport"), plan.get("transport_from"),
                                plan.get("transport_table"))
        if pinned is not None:
            print(pinned)
        if plan.get("devices_sentence"):
            print(f"go: {plan['devices_sentence']}")
        if plan.get("tiles"):
            print(f"go: {plan['tiles']['sentence']}")
        if plan.get("devices"):
            gate = memory_gate(plan)
            print(gate["verdict"])
            if gate["refuse"]:
                raise GoRefusal(memory_refusal_text(gate))
        print("")
        beside = posts_beside_preparation(plan)
        for label, command in (
                ("1. authority", authority_command(plan)),
                ("2. fetch" + (" (as posted, beside step 4, which starts on "
                               "the window's first leads)" if beside else ""),
                 fetch_command(plan)),
                ("3. manifest" + (" (only for a window the fetch finds "
                                  "already here whole; as posted, step 4's "
                                  "seal writes it)" if beside else ""),
                 manifest_command(plan, bridge)),
        ):
            print(f"{label}\n     {printable(command)}")
        # The last two carry values that do not exist yet.  Naming the
        # FILE each one is read from beats printing a plausible-looking
        # hash: the point of a dry run is to show what will happen, and
        # what will happen is a read from that artifact.
        whole = printable(prepare_command(
            plan, bridge, manifest=manifest,
            manifest_sha256="<sha256 of " + manifest.name + ", after step 3>",
            cycle_stamp=cycle_stamp, geog_root=geog_root))
        if beside:
            print("4. prepare\n     " + printable(prepare_command(
                plan, bridge, manifest=None, manifest_sha256=None,
                cycle_stamp=cycle_stamp, geog_root=geog_root,
                as_posted=posting_folder(plan))))
            print("     # or, for a window already here whole, after step "
                  "3:\n     " + whole)
        else:
            print("4. prepare\n     " + whole)
        # Branched the way the real run branches in `_run_forecast`: a
        # tree binds ONE preparation receipt, so printing the
        # single-domain three-digest relay for it would show a command
        # this chain will never issue.  The tree arm reads the receipt
        # off disk, which does not exist yet on a dry run, so it is
        # composed from the plan's own paths.
        if plan.get("domains", 1) > 1:
            print("5. forecast\n     " + printable(
                tree_forecast_command(plan, digests={
                    "preparation_receipt":
                        "<sha256 of prepared/proof.json, after step 4>",
                    "experiment_config":
                        "<sha256 of authority/experiment.toml, "
                        "after step 1>"})))
        else:
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
    pinned = transport_note(plan.get("transport"), plan.get("transport_from"),
                            plan.get("transport_table"))
    if pinned is not None:
        print(pinned)
    # The run-folder line, BEFORE the gates -- the same line, from the
    # same function, that --dry-run prints second.  It used to print
    # only after the memory and geography gates, so both of the
    # 2.4.1-upgrader walk's real attempts refused without ever naming
    # the 2.5.0 layout, and no cheap door revealed the new folder
    # naming (UX finding N18).  Announced here it still spends nothing:
    # the folder is only CLAIMED after the gates, below, under this
    # exact name (the plan carries its launch instant).  Only another
    # chain taking that name first moves the claim, and then the line
    # prints again corrected.
    predicted_root = plan["root"]
    print(_run_folder_note(plan))
    # The routing a [tiles] table chose, said before any stage runs.  The
    # forecast stage prints its own decision line too, but that stage's
    # stdout is captured -- a reader watching this terminal would learn
    # their run streamed only from report.json, after the fact.
    if plan.get("devices_sentence"):
        print(f"go: {plan['devices_sentence']}")
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
            if gate.get("preparation_warning"):
                print(f"go: WARNING -- {gate['preparation_warning']}.")

        _require_forecast_device()
        # Same rule as the memory gate, same side of the download.
        geography = geography_refusal(geog_root)
        if geography is not None:
            raise GoRefusal(geography)
        # And the disk, on the same side of the download and of the claim
        # below.  A hosting `gpuwm run-plan` (a caller that brought its
        # own observer) asked this before it started the chain.
        if chain is not None:
            _disk_admission(plan, args)

        hosted_relay = None
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
        # What an earlier run in this case folder could not remove when it
        # ended (a working store another program still held), now that
        # nothing of that run is running.
        _sweep_earlier_render_scratch(plan["case_root"], observer=observer)

        _run_stage("authority", authority_command(plan), explain=explain,
                   observer=observer)
        # AS POSTED (DESIGN A136 2.5): the fetch runs beside the
        # preparation, which starts once the fetch has scheduled the
        # window and waits for each lead's marker, and whose seal writes
        # the input manifest, so there is no manifest stage.  A fetch that
        # found its window already here whole schedules nothing, and the
        # chain binds the whole window as before.
        posting = None
        if posts_beside_preparation(plan):
            from gpuwm.source_posting import POSTING_SCHEDULE_NAME

            # Its beat line reads the fetch's own schedule: the leads in,
            # and the lead it waits for (_posting_note).
            beside = _BesideStage(
                "fetch", fetch_command(plan), plan=plan, explain=explain,
                observer=observer,
                progress=posting_folder(plan) / POSTING_SCHEDULE_NAME)
            posting = _await_posting(beside, plan)
        else:
            _run_stage("fetch", fetch_command(plan), explain=explain,
                       observer=observer)
        if posting is None:
            beside = None
            _run_stage("manifest", manifest_command(plan, bridge),
                       observer=observer,
                       explain=explain)
            manifest = front_door_manifest(plan)
            if not manifest.is_file():
                raise GoRefusal(
                    f"the manifest stage wrote no {manifest}, so the "
                    "preparation stage has nothing to bind against")
            prepare_stage_command = prepare_command(
                plan, bridge, manifest=manifest,
                manifest_sha256=sha256_file(manifest),
                cycle_stamp=cycle_stamp, geog_root=geog_root)
        else:
            print("go: the fetch runs beside the preparation, which starts "
                  "on the window's first leads and waits for each later "
                  "one as it posts", flush=True)
            hosted_events = (getattr(observer, "events", None)
                             if chain is None else None)
            if hosted_events is not None:
                # A hosting run (gpuwm run-plan) keeps its own stream, where
                # go's own relay of the posting folder is not open.
                hosted_relay = chain_events.HostedPostingRelay(
                    hosted_events, data_dir=plan["data"])
                hosted_relay.start(
                    since_unix_ms=int(beside.launched * 1000))
            prepare_stage_command = prepare_command(
                plan, bridge, manifest=None, manifest_sha256=None,
                cycle_stamp=cycle_stamp, geog_root=geog_root,
                as_posted=posting)
        announce_policy_backend(prepare_stage_command)

        # Whether the fetch beside the chain had failed (or said it was
        # failing) when the preparation or the forecast first failed: then
        # they failed because of it and its failure is the chain's, and
        # otherwise the first of them to fail is (_BesideStage.settle).
        first_failure: dict[str, object] = {}
        first_failure_lock = threading.Lock()

        def _failing(label: str, error: BaseException) -> None:
            if beside is None:
                return
            with first_failure_lock:
                if not first_failure:
                    first_failure.update(fetch_first=beside.said_failure(),
                                         stage=label, error=error)
                    if not first_failure["fetch_first"]:
                        beside.suppress_failure_publication()

        def preparation_failure_is_secondary() -> bool:
            with first_failure_lock:
                return (first_failure.get("stage") == "forecast"
                        and not first_failure.get("fetch_first", True))

        def prepare():
            try:
                _run_stage("prepare", prepare_stage_command, explain=explain,
                           # Not `prepared/progress.json`: the stage builds
                           # its product in a staging directory and publishes
                           # it with one rename (at its head, when chained),
                           # so the progress file lives beside it.  The path
                           # is asked of the publisher rather than spelled
                           # here, so the writer and this poller cannot
                           # disagree.
                           progress=prepare_progress_path(plan["prepared"]),
                           observer=observer,
                           secondary_failure=preparation_failure_is_secondary)
            except BaseException as error:
                _failing("prepare", error)
                raise

        # `--keep-checkpoints`, or its default of one set when this chain
        # was typed rather than hosted.  Every other `go` route hands the
        # choice to the run plan that sets it; this chain enters no run
        # plan, so the choice and the default were parsed and dropped and
        # every hourly set stayed on disk.  A hosting run-plan has set its
        # own policy already, and with no flag here that one stands.
        keep = getattr(args, "keep_checkpoints", None)
        if keep is None and chain is not None:
            from gpuwm.resume import DEFAULT_KEEP_CHECKPOINTS
            keep = DEFAULT_KEEP_CHECKPOINTS

        def forecast(head_sha256):
            # A hierarchy proof carries no single prepared-cache identity,
            # and proof_digests says so by refusing.  The tree arm reads its
            # own one digest instead, so it is not asked for here.  A
            # chained head binds by its head digest (head_digests), a
            # tree's as a single domain's.
            try:
                digests = (head_digests(plan["prepared"], head_sha256)
                           if head_sha256 is not None
                           else {} if plan.get("domains", 1) > 1
                           else proof_digests(plan["prepared"]))
                # The first frame the forecast commits is the analysis, and an
                # observer that wants it rendered as it lands is told so before
                # the forecast starts -- with THIS plan, the one the render
                # stage below runs on.  `_notify` because an observer without
                # the hook (there is no such thing in tree, but `go` takes any
                # duck) must not be a reason a chain stops.
                _notify(observer, "arm_first_products", render_plan=plan)
                with _checkpoint_retention(keep):
                    _run_forecast(plan, digests, explain=explain,
                                  observer=observer)
            except BaseException as error:
                _failing("forecast", error)
                raise

        def head_ready(head_sha256):
            print("go: preparation head published; the forecast starts "
                  "while the remaining boundary intervals are prepared",
                  flush=True)
            if hosted_relay is not None:
                # The head read every start need: a lead the fetch asks for
                # again from here is not the run's start wait.
                hosted_relay.head_ready()
            _notify(observer, "prepare_head_ready", head_sha256=head_sha256)

        from gpuwm.ingest.boundary_stream import run_chained

        # One domain or a tree: a GFS-series tree is chained like a single
        # domain (gfs_direct._prepare_chained_gfs_tree), and the tree
        # runner binds its head (--prepared-head-sha256).  A preparation
        # that publishes at its seal says why itself, and run_chained then
        # starts the forecast after that seal.
        try:
            run_chained(prepared_root=plan["prepared"], prepare=prepare,
                        forecast=forecast, on_head=head_ready,
                        observer=observer)
        except BaseException as stopped:
            if hosted_relay is not None:
                # What the fetch published before the chain stopped reaches
                # the hosting run's stream, and nothing after it does.
                hosted_relay.stop()
                hosted_relay = None
            if beside is not None and not isinstance(
                    stopped, (GoInterrupted, KeyboardInterrupt)):
                # Not on Ctrl-C: the interrupt contract kills no child.
                # When the fetch failed first (a lead later than its
                # budget, a host refusal) and the preparation or forecast
                # stopped because of it, the fetch's failure is the chain's,
                # with its exit code and its words.  When the preparation
                # or forecast failed first, the run is over and so is its
                # download: the fetch is stopped, said as stopped, and the
                # stage that failed is the chain's failure.
                with first_failure_lock:
                    first = dict(first_failure)
                    fetch_first = bool(first.get(
                        "fetch_first", beside.said_failure()))
                failed = beside.settle(fetch_first=fetch_first)
                if failed is not None:
                    # The observer may have heard the stage that failed
                    # because of the fetch before the fetch's own failure
                    # (a fetch writes its source_behind record before it
                    # exits), so the chain names its failure once more.
                    _notify(observer, "chain_failed", label=beside.label,
                            exit_code=getattr(failed, "code", 1),
                            diagnostic=getattr(failed, "diagnostic", ""))
                    raise failed from None
                if not fetch_first and first:
                    # A later preparation or fetch stage may have told
                    # the observer it failed during the seal hold. The
                    # complete callback's first failure remains primary.
                    error = first["error"]
                    code = getattr(error, "code", getattr(
                        error, "exit_code", 2 if isinstance(error, GoRefusal)
                        else 1))
                    _notify(observer, "chain_failed", label=first["stage"],
                            exit_code=code,
                            diagnostic=getattr(error, "diagnostic", "")
                            or str(error))
                    from gpuwm.ingest.boundary_stream import read_replaced_json
                    from gpuwm.source_posting import POSTING_FAILED_NAME

                    try:
                        later = read_replaced_json(
                            posting_folder(plan) / POSTING_FAILED_NAME)
                    except (OSError, ValueError):
                        later = None
                    if isinstance(later, dict) and later.get("code") == "source_behind":
                        _notify(observer, "warn", code="secondary_source_behind",
                                message="After the run failed: " + (
                                    later.get("message") or
                                    "a later source lead timed out"),
                                source_behind=later)
                    raise first["error"] from None
            raise
        if beside is not None:
            # Every lead's marker was read by the seal, so the fetch has
            # published its whole window; it ends here, and a failure it
            # met after that is still the chain's.
            beside.result()
        if hosted_relay is not None:
            hosted_relay.stop()
            hosted_relay = None
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


def _checkpoint_sets(text: str) -> int:
    """``--keep-checkpoints``: a whole number of sets, 0 keeping every one."""
    import argparse
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a whole number of checkpoint sets; 0 keeps every set")
    return value


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "go",
        help="prepare, run and render a forecast, fetching inputs when needed")
    parser.add_argument("config", type=Path, metavar="CONFIG",
                        help="an experiment TOML from gpuwm domain; its source and "
                             "domain tree choose the preparation route")
    parser.add_argument("--outdir", type=Path, default=None, metavar="DIR",
                        help="output root for one timestamped run folder per launch, "
                             "with forecast files, pictures and diagnostics "
                             "(default <config-stem>-go beside the config); "
                             "an existing run-... folder is used directly")
    run_stamp_module.add_argument(
        parser, artifacts="forecast files, pictures and diagnostics",
        option="--outdir")
    parser.add_argument("--restart", type=Path, default=None, metavar="CHECKPOINT",
                        help="continue an existing checkpoint; prepared-cache runs "
                             "also need --prepared-root, and use fresh output")
    parser.add_argument("--prepared-root", type=Path, default=None, metavar="DIR",
                        help="run this existing prepared bundle without fetch or "
                             "preparation; add --restart to continue its checkpoint")
    parser.add_argument("--wps-namelist", type=Path, default=None, metavar="PATH",
                        help="with --prepared-root: the exact WPS authority required "
                             "by a single-domain portable bundle")
    parser.add_argument("--data-dir", type=Path, default=None, metavar="DIR",
                        dest="data_dir",
                        help="download routes only: use this existing download "
                             "instead of the automatically managed request cache")
    parser.add_argument("--geog-root", type=Path, default=None, metavar="DIR",
                        help="override the geography tree (default: [case_data].geog_root "
                             "for declared inputs, otherwise the staged WPS_GEOG tree)")
    from gpuwm.fetch import FETCH_TRANSPORTS
    parser.add_argument("--transport", default=None, choices=FETCH_TRANSPORTS,
                        help="download routes only: pin the fetch stage to one host "
                             "of the source's endpoint ladder, the value `gpuwm "
                             "fetch --transport` takes; wins over the config's "
                             "[fetch] transport, and the plan says which it used")
    parser.add_argument("--cycle", default=None, metavar="YYYY-MM-DDTHH|latest",
                        help="download routes only: run the config at this "
                             "cycle; wins over the config's [fetch] cycle the "
                             "way --transport does. The config is re-timed "
                             "(start time, delayed nests, namelists) into "
                             "<outdir>/cycles/<cycle>/; latest is resolved "
                             "once, under the run's posting rule (as posted: "
                             "the newest cycle whose start needs are posted)")
    parser.add_argument("--whole-cycle", action="store_true", dest="whole_cycle",
                        help="download routes only: the fetch stage waits for "
                             "the whole cycle (the old rule) instead of taking "
                             "each lead as it posts; wins over the config's "
                             "[fetch] as_posted")
    parser.add_argument("--late-after-minutes", type=float, default=None,
                        dest="late_after_minutes", metavar="MIN",
                        help="download routes only: how far past its scheduled "
                             "time a lead may be before the run stops with exit "
                             "75; wins over the config's [fetch] "
                             "late_after_minutes and the source row's budget")
    parser.add_argument("--readiness", action="store_true",
                        help="print gpuwm.readiness.v1 for the config's fetch "
                             "window on stdout and run nothing: exit 0 ready, "
                             "75 not yet, 2 refused")
    parser.add_argument("--no-probe", action="store_true", dest="no_probe",
                        help="with --readiness: compute the schedule from the "
                             "source table only and ask no host")
    parser.add_argument("--supplement", action="append", metavar="ROLE=PATH",
                        help="explicit preparation donor; repeat for multiple files. "
                             "HRRR accepts PMSL=GRIB inside --data-dir and binds its "
                             "bytes in the preparation source manifest")
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
    # `gpuwm render --section`, carried to every render this chain runs:
    # the frames drawn as they land, the early first frame and the
    # end-of-run batch.  Without it an `xsec:` term in --products passed
    # review and was dropped at render with advice to add a flag this
    # command did not have.
    parser.add_argument("--section", default=None, dest="render_section",
                        metavar="lat,lon,lat,lon|FILE.json",
                        help="the line the vertical-section products "
                             "(xsec:<fill>[/<overlay>...] in --products) are "
                             "cut along, the same value `gpuwm render "
                             "--section` takes; a JSON file gives {start, "
                             "end} or a {points, extend_km} polyline.  "
                             "Spell a line that starts with a minus sign "
                             "as --section=-33.9,151.2,-34.1,151.3.  An "
                             "xsec: product with no line is refused before "
                             "anything is fetched")
    parser.add_argument("--keep-checkpoints", type=_checkpoint_sets, default=None,
                        dest="keep_checkpoints", metavar="N",
                        help="how many complete checkpoint sets the run keeps in "
                             "its folder (default 1, enough to resume); 0 keeps "
                             "every hourly set, which a later branch or downscale "
                             "from an earlier checkpoint needs")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="validate the route and show how to launch it; fetch "
                             "and run nothing")
    parser.add_argument("--devices", type=int, default=None, metavar="N",
                        help="resident slab count; replaces [devices] count")
    parser.add_argument("--no-memory-gate", action="store_true",
                        dest="no_memory_gate",
                        help="skip the before-launch memory check that "
                             "refuses a configuration whose binding phase "
                             "cannot fit this card's free VRAM, and the "
                             "forecast runner's own check of the same "
                             "envelope; a model state too big to build at "
                             "all is still refused")
    parser.set_defaults(func=go_main)
    return parser


__all__ = [
    "DEFAULT_RENDER_PRODUCTS",
    "GoRefusal", "GoStageFailed", "HEARTBEAT_SECONDS", "MANUAL_CHAIN",
    "ORCHESTRATED_SOURCES", "RUNNER_MODULE", "RUNNER_RELATIVE",
    "WRFOUT_GLOB", "admit_render_products", "unknown_render_products",
    "TREE_RUNNER_MODULE", "tree_forecast_command",
    "END_STAGE_GRACE_SECONDS", "end_stage_processes",
    "authority_command", "config_fetch_request", "failed_line", "fetch_command",
    "fetch_request", "forecast_command", "go_main",
    "manifest_command", "memory_gate", "plan_from_config",
    "prepare_command", "printable",
    "proof_digests", "register_cli", "render_command",
    "render_extra_missing", "render_inputs_file", "render_section_value",
    "resolve_bridge",
    "run_render_pass", "run_stage", "sweep_kept_render_inputs",
    "wrfout_frames",
]
