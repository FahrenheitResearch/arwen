"""A versioned plan in, a structured event stream out.

Every other front door in this package talks to a person: it prints a
resolved-config report, a progress line, a refusal sentence.  A program
driving gpuwm as a subprocess -- a GUI, a scheduler, a fleet controller
-- has to read those same facts, and until now its only route to them
was to parse the prose.  Prose is not an interface.  It gets rewritten
for clarity, and the rewrite silently breaks the consumer.

``gpuwm run-plan PLAN.json`` is the interface.  A caller writes ONE JSON
document naming which existing route to execute and which existing
config to execute it with, and gets back an append-only JSONL stream in
which every fact the human output carries is a typed field on a typed
event.  Nothing about the model changes: the plan is an ENVELOPE over
the config system, it is resolved by the same
:func:`gpuwm.case_data.load_experiment_case` seam ``gpuwm run`` uses,
and it is executed by the same :func:`gpuwm.runtime.run_experiment`.
There is no second config format and no forked validation.

The three documents
-------------------

``gpuwm.run-plan.v1`` -- the plan.  ``schema``, ``name``, ``route``,
``config`` (``{"path": ...}`` or ``{"inline": "..."}``), and the
optional ``fetch`` / ``output_root`` / ``run_options``.  Unknown
top-level keys and unknown schema ids are refused, not ignored: a
dropped key runs a default under the name of your value.

``gpuwm.run-plan.event.v1`` -- one JSON object per line of
``<run_dir>/events.jsonl``, mirrored verbatim to stdout.  Every line
carries ``schema_version``, a monotonic ``sequence``, ``emitted_unix_ms``
and an ``event`` tag; the event's own fields are flattened alongside.
The tags are :data:`EVENT_TAGS`.

``gpuwm.run-manifest.v1`` -- ``<run_dir>/run-manifest.json``, written
before any work starts.  It carries this process's pid and the absolute
path of every stream a consumer may want, INCLUDING the two the rest of
the package already owns.

Reattach: read the heartbeat, do not own the pipe
-------------------------------------------------

This module publishes no progress state of its own.
:mod:`gpuwm.supervisor` already writes ``run-progress.json``
(``gpuwm.run-progress/v1``) atomically on every step, and it stays the
only writer of it: :class:`RunObserver` COMPOSES with
:class:`gpuwm.supervisor.RuntimeHeartbeat` rather than replacing it, so
a run-plan run leaves exactly the same heartbeat a ``gpuwm run`` leaves.

So a consumer that attaches to a run already in flight does three
things, in this order:

1. Read ``run-manifest.json`` for the paths and the pid.
2. Read ``run-progress.json`` for CURRENT state.  That file is the
   authoritative anchor -- it is atomically republished, it is what the
   supervisor's own recovery reads, and it is one small read rather
   than a replay.
3. Replay ``events.jsonl`` from byte zero for HISTORY (it is the
   complete record, never rotated or truncated), then tail it for live
   detail.

A consumer that treats the event stream as the anchor will be wrong
exactly once: after a crash between the last event flush and the
process exit.  The heartbeat is the thing that is durable by design.

Query modes
-----------

``--resolve``, ``--estimate`` and ``--probe`` answer a front end's three
pre-flight questions without running anything, each as one JSON document
on stdout.  They live here rather than in the front end because the
answers are derived from this package's own machinery -- the config
loaders, :mod:`gpuwm.core.preflight`'s VRAM itemization,
:mod:`gpuwm.doctor`'s estate checks -- and a front end that
reimplemented them would be reporting its own arithmetic under gpuwm's
name.  Where this package has no measured number for something
(wall-time for an arbitrary configuration), the field is ``null`` with
its ``basis`` stated.  Nothing is estimated by guess.

Nothing silent
--------------

``resolved_plan`` carries ``automatic_resolutions``: one entry for every
value this pipeline chose on its own -- an omitted plan key taking its
default, a schema default filling an unspelled config key, a per-domain
timestep derived down the nest ratio chain.  A consumer can render that
list and a reader can see, before the run, every number nobody typed.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gpuwm.explain import layered


#: The plan document a caller writes.
PLAN_SCHEMA = "gpuwm.run-plan.v1"

#: The ``schema_version`` on every line of the event stream.  Its shape
#: -- envelope keys first, event-specific fields flattened after -- is
#: the one already proven by the Rust side of this stack, so a reader
#: written for one stream needs no second framing for the other.
EVENT_SCHEMA = "gpuwm.run-plan.event.v1"

#: The attach manifest.
MANIFEST_SCHEMA = "gpuwm.run-manifest.v1"

#: The documents this module answers ``--resolve``/``--estimate``/
#: ``--probe`` with.  Versioned for the same reason the others are: a
#: front end pins what it parses.
RESOLVE_SCHEMA = "gpuwm.run-plan.resolved.v1"
ESTIMATE_SCHEMA = "gpuwm.run-plan.estimate.v1"
PROBE_SCHEMA = "gpuwm.run-plan.probe.v1"

EVENTS_FILENAME = "events.jsonl"
MANIFEST_FILENAME = "run-manifest.json"

#: The five stages a run passes through, in order.  ``fetch`` is skipped
#: when the plan declares no ``[fetch]``; every other stage always runs
#: and always emits its pair of events, so a consumer's stage timeline
#: has no holes to interpret.
STAGES = ("fetch", "prepare", "initialize", "forecast", "finalize")

#: Every event tag this module will ever emit.  A consumer switching on
#: ``event`` can be exhaustive against this tuple.
EVENT_TAGS = (
    "plan_accepted", "resolved_plan", "stage_started", "stage_finished",
    "model_progress", "output_committed", "warning", "completed", "failed",
)

#: Envelope keys an event's own fields may not shadow.
_ENVELOPE_KEYS = frozenset({
    "schema_version", "sequence", "emitted_unix_ms", "event"})

_TOP_LEVEL_KEYS = frozenset({
    "schema", "name", "route", "config", "fetch", "output_root",
    "run_options"})
_REQUIRED_KEYS = ("schema", "name", "route", "config")
_CONFIG_KEYS = frozenset({"path", "inline", "intent"})
_FETCH_KEYS = frozenset({"args"})

#: ``config.intent`` keys, and the ``gpuwm domain`` flag each one is.
#:
#: A mapping and not a schema.  Intent is not a third config format --
#: it is the wizard's own question list, spelled as JSON so a front end
#: can build it from typed fields instead of assembling a command line.
#: Every value is validated by the wizard's REAL parser and every
#: refusal is the wizard's own, so this table is the only thing that
#: could go stale, and a key whose flag disappears fails loudly at the
#: parser rather than being quietly dropped.
#:
#: ``--out`` is deliberately absent: run-plan owns where the generated
#: config lands, and a plan that could redirect it would be able to
#: write outside its own run directory.
_INTENT_FLAGS = {
    "point": "--point",
    "polygon": "--polygon",
    "buffer_km": "--buffer-km",
    "projection": "--projection",
    "name": "--name",
    "card": "--card",
    "vram_gib": "--vram-gib",
    "ladder": "--ladder",
    "root_dx_km": "--root-dx",
    "chain": "--chain",
    "physics_profile": "--physics-profile",
    "hours": "--hours",
    "source": "--source",
    "cycle": "--cycle",
    "forecast_start_hour": "--forecast-start-hour",
    "data_dir": "--data-dir",
    "forcing": "--forcing",
    "vtable": "--vtable",
    "geog_root": "--geog-root",
    "history_interval_s": "--history-interval",
    "nest_history_interval_s": "--nest-history-interval",
}

#: How each intent key actually reaches the chain that executes it.
#:
#: This table exists because three keys did not reach it at all.  The
#: wizard accepts ``--geog-root``, ``--data-dir``, ``--forcing`` and
#: ``--vtable`` and writes the last three into ``[case_data]`` -- a
#: table it only emits for ERA5.  On the prepared route (gfs) there is
#: no ``[case_data]``, so those values were validated, accepted, and
#: then silently dropped: a plan naming a non-default geography tree ran
#: against the default one.
#:
#: The values:
#:
#: ``"config"``      the wizard bakes it into the generated TOML, and
#:                   every consumer reads it from there.
#: ``"go:--flag"``   it does NOT survive into the config on the prepared
#:                   route and must be forwarded to ``gpuwm go``.
#: ``"case_data"``   it lands in ``[case_data]``, so it is meaningful
#:                   only on a route that has one, and is refused
#:                   loudly on a route that does not.
#:
#: Every key in :data:`_INTENT_FLAGS` must appear here; a test fails if
#: one does not, so a key cannot be added without answering "and how
#: does that reach the thing that runs?".
_INTENT_DELIVERY = {
    "point": "config",
    "polygon": "config",
    "buffer_km": "config",
    "projection": "config",
    "name": "config",
    "card": "config",
    "vram_gib": "config",
    "ladder": "config",
    "root_dx_km": "config",
    "chain": "config",
    "physics_profile": "config",
    "hours": "config",
    "source": "config",
    "cycle": "config",
    "forecast_start_hour": "config",
    "history_interval_s": "config",
    "nest_history_interval_s": "config",
    # `go` defaults its data directory to <outdir>/data and never reads
    # the [fetch].out hint the wizard wrote, so this has to travel as a
    # flag or it does not travel.
    "data_dir": "go:--data-dir",
    # The static geography tree: [case_data].geog_root on the ERA5
    # route, and a `gpuwm go` flag on the prepared one.
    "geog_root": "go:--geog-root",
    # ERA5's declared inputs.  The prepared chain fetches its own GRIB
    # and takes its Vtable from the bridge, so there is nothing for
    # these to mean there and no flag to carry them.
    "forcing": "case_data",
    "vtable": "case_data",
}

#: The filename the generated config takes inside the run directory.
GENERATED_CONFIG_NAME = "intent-config.toml"

#: The one source whose emitted config the ``experiment`` route can
#: actually run.  Not a policy invented here -- the wizard writes a
#: ``[case_data]`` table for ERA5 and for nothing else, because (its
#: words) "the config-driven route decodes native GRIB1 = ERA5 today"
#: (domain_wizard.py:3445).  A GFS or HRRR emission is a real config,
#: but its consumer is the native/prepared front door, and those are
#: not routes yet.
_CASE_DATA_SOURCES = frozenset({"era5"})

#: ``gpuwm run``'s own documented default output directory.  A plan that
#: omits ``output_root`` lands where the command it wraps would have.
DEFAULT_OUTPUT_ROOT = Path("out") / "run"

#: Pipeline preparation phases, mapped to the stage each one belongs to.
#: The literals are :func:`gpuwm.runtime._preparation_progress`'s own --
#: this table reads them, it does not author them.  A phase absent here
#: does not silently land in whichever stage happens to be open: it
#: emits a ``warning`` naming itself, so a pipeline that grows a phase
#: is reported rather than mis-filed.
_PHASE_STAGES = {
    "quarantine-wrfout": "prepare",
    "resolve-schedule": "prepare",
    "prepare-case": "prepare",
    "build-domain-tree": "prepare",
    "initialize-health-validator": "initialize",
    "cold-start-wrfout": "initialize",
    "validate-checkpoint": "initialize",
    "restore-checkpoint": "initialize",
    "restore-tree-checkpoint": "initialize",
    "initialize-domain-writers": "initialize",
    "initial-health-gate": "initialize",
}


class PlanError(ValueError):
    """A plan document this front door refuses to execute.

    ``ValueError`` because that is what every refusal in this package
    travels as, and what :func:`gpuwm.cli.main` prints as one sentence
    at exit 2 rather than as a traceback.
    """


# ---------------------------------------------------------------------------
# The plan document
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunPlan:
    """One validated plan, with every path made absolute."""

    name: str
    route: str
    config_path: Path | None
    config_inline: str | None
    config_intent: Mapping[str, Any] | None
    config_base_dir: Path
    fetch_arguments: tuple[str, ...] | None
    output_root: Path
    run_options: Mapping[str, Any]
    sha256: str
    source: str
    automatic_resolutions: tuple[Mapping[str, Any], ...]

    @property
    def run_dir(self) -> Path:
        """The one directory this run writes into.

        ``output_root`` IS the run directory, not a parent to derive one
        under: deriving would mean inventing a directory name from the
        plan's ``name``, and a caller who cannot predict where its
        outputs land cannot collect them.
        """

        return self.output_root

    @property
    def config_kind(self) -> str:
        """``"path"``, ``"inline"`` or ``"intent"``."""

        if self.config_path is not None:
            return "path"
        return "inline" if self.config_inline is not None else "intent"

    def config_bytes(self) -> bytes:
        """The config TOML this plan names, as bytes.

        One accessor for the two spellings that ARE a config already.
        ``intent`` is not one of them: it has to be generated first,
        which needs a directory to generate into, so it is resolved by
        :func:`resolve_plan` rather than read here.
        """

        if self.config_inline is not None:
            return self.config_inline.encode("utf-8")
        if self.config_path is None:
            raise PlanError(
                "an intent plan has no config bytes until the wizard has "
                "generated them; resolve_plan does that")
        return Path(self.config_path).read_bytes()


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{label} must be a non-empty string")
    result = value.strip()
    if any(character in result for character in "\r\n"):
        raise PlanError(f"{label} must fit on one line")
    return result


def _reject_unknown(mapping: Mapping[str, Any], known, label: str) -> None:
    unknown = sorted(set(mapping) - set(known))
    if not unknown:
        return
    raise PlanError(layered(
        f"{label} does not have the key(s) {unknown}; no key is "
        "ignored, because a dropped key runs a default under the name "
        "of your value.",
        f"Known {label} keys: {sorted(known)}."))


def build_plan(raw: Mapping[str, Any], *, source: str,
               base_dir: str | Path, sha256: str) -> RunPlan:
    """Validate a parsed plan document and build the :class:`RunPlan`.

    Separate from :func:`load_plan` for the reason
    :func:`gpuwm.experiment.build_experiment` is separate from
    :func:`gpuwm.experiment.load_experiment`: the validation is the
    part a test wants to reach without a file on disk.
    """

    if not isinstance(raw, dict):
        raise PlanError("run plan must be a JSON object")
    _reject_unknown(raw, _TOP_LEVEL_KEYS, "run plan")
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise PlanError(
            f"run plan {source} is missing required key(s) {missing}")
    if raw["schema"] != PLAN_SCHEMA:
        raise PlanError(layered(
            f"{source} is not a {PLAN_SCHEMA} document: schema is "
            f"{raw['schema']!r}.",
            "The schema id carries this document's version.  A reader "
            "that accepted an unrecognized id would be guessing which "
            "fields it holds, which is how a plan runs as something "
            "other than what it says."))

    base = Path(base_dir)
    resolutions: list[dict[str, Any]] = []

    name = _nonempty_string(raw["name"], "run plan 'name'")
    route = _nonempty_string(raw["route"], "run plan 'route'")
    if route not in ROUTES:
        raise PlanError(layered(
            f"run plan {source} names route {route!r}, which this build "
            f"does not have.",
            "Known routes: " + ", ".join(
                f"{key} ({value.summary})"
                for key, value in sorted(ROUTES.items())) + "."))

    config = raw["config"]
    if not isinstance(config, dict):
        raise PlanError("run plan 'config' must be an object with "
                        "exactly one of 'path', 'inline' or 'intent'")
    _reject_unknown(config, _CONFIG_KEYS, "run plan 'config'")
    spelled = sorted(set(config) & _CONFIG_KEYS)
    if len(spelled) != 1:
        raise PlanError(layered(
            "run plan 'config' must carry exactly ONE of 'path' (a TOML "
            "on disk), 'inline' (the TOML text itself) or 'intent' (the "
            f"shape of the run, for the wizard to write), got {spelled}.",
            "They are three ways of saying the same thing, so two of "
            "them is a question about which one wins, and this front "
            "door does not answer questions like that by picking."))
    config_path = None
    config_inline = None
    config_intent = None
    if spelled == ["path"]:
        config_path = _absolute(config["path"], base,
                                "run plan 'config.path'")
    elif spelled == ["inline"]:
        if not isinstance(config["inline"], str) or not config["inline"]:
            raise PlanError("run plan 'config.inline' must be non-empty "
                            "TOML text")
        config_inline = config["inline"]
    else:
        config_intent = _build_intent(config["intent"], route=route)
        resolutions.append({
            "scope": "config", "key": "generated_by",
            "value": "gpuwm domain", "basis": "intent_route",
            "note": "the config is written by the wizard from this "
                    "plan's intent, not authored by the caller; its "
                    "full text is on the resolved_plan event"})

    fetch_arguments = None
    fetch = raw.get("fetch")
    if fetch is not None:
        if not isinstance(fetch, dict):
            raise PlanError("run plan 'fetch' must be an object")
        _reject_unknown(fetch, _FETCH_KEYS, "run plan 'fetch'")
        if "args" not in fetch:
            raise PlanError(
                "run plan 'fetch' must carry 'args': the argv list "
                "`gpuwm fetch` itself takes.  The flags are not "
                "restated here, because a second spelling of them "
                "would be a second thing to keep in step.")
        if not isinstance(fetch["args"], list) or not fetch["args"]:
            raise PlanError("run plan 'fetch.args' must be a non-empty "
                            "list of argv strings")
        fetch_arguments = tuple(
            _nonempty_string(argument, f"run plan 'fetch.args[{index}]'")
            for index, argument in enumerate(fetch["args"]))
        _validate_fetch_arguments(fetch_arguments)

    if "output_root" in raw:
        output_root = _absolute(raw["output_root"], base,
                                "run plan 'output_root'")
    else:
        output_root = (base / DEFAULT_OUTPUT_ROOT).resolve()
        resolutions.append({
            "scope": "plan", "key": "output_root",
            "value": str(output_root), "basis": "front_door_default",
            "note": "gpuwm run's own default --outdir, resolved against "
                    "the plan's directory"})

    options = raw.get("run_options", {})
    if not isinstance(options, dict):
        raise PlanError("run plan 'run_options' must be an object")
    known_options = ROUTES[route].run_options
    _reject_unknown(options, known_options, "run plan 'run_options'")
    resolved_options: dict[str, Any] = {}
    for key in sorted(known_options):
        if key in options:
            resolved_options[key] = _run_option(key, options[key], base)
            continue
        resolved_options[key] = _RUN_OPTION_DEFAULTS[key]
        resolutions.append({
            "scope": "run_options", "key": key,
            "value": _RUN_OPTION_DEFAULTS[key], "basis": "schema_default"})

    return RunPlan(
        name=name, route=route, config_path=config_path,
        config_inline=config_inline, config_intent=config_intent,
        config_base_dir=base,
        fetch_arguments=fetch_arguments, output_root=output_root,
        run_options=resolved_options, sha256=sha256, source=source,
        automatic_resolutions=tuple(resolutions))


def _build_intent(intent: object, *, route: str) -> dict[str, Any]:
    """Validate a ``config.intent`` block into wizard-flag arguments.

    Shape only.  Every VALUE is left to the wizard's own parser -- a
    latitude, a ladder name, a physics profile id and a cycle are all
    things ``gpuwm domain`` already refuses precisely, and a second
    opinion here would be a second thing to keep in step.
    """

    if not isinstance(intent, dict):
        raise PlanError("run plan 'config.intent' must be an object")
    _reject_unknown(intent, _INTENT_FLAGS, "run plan 'config.intent'")
    for required in ("cycle",):
        if required not in intent:
            raise PlanError(
                f"run plan 'config.intent' must carry {required!r}; "
                "`gpuwm domain` requires it and cannot guess one")
    if not ({"point", "polygon"} & set(intent)):
        raise PlanError(
            "run plan 'config.intent' must carry 'point' (LAT,LON) or "
            "'polygon' (a GeoJSON path): the wizard sizes a domain "
            "around a place, and there is no default place")

    if not ROUTES[route].needs_case_data:
        stranded = sorted(
            key for key in intent
            if _INTENT_DELIVERY.get(key) == "case_data")
        if stranded:
            raise PlanError(layered(
                f"run plan 'config.intent' sets {stranded}, which the "
                f"{route!r} route has nowhere to put.",
                "`gpuwm domain` writes those into [case_data], and it "
                "writes no [case_data] for this route's sources -- so "
                "they would be accepted here and then silently dropped, "
                "which is how a run uses a default nobody chose.  The "
                "prepared chain fetches its own forcing and takes its "
                "Vtable from the bridge; neither is yours to set."))

    source = intent.get("source", "era5")
    if ROUTES[route].needs_case_data and source not in _CASE_DATA_SOURCES:
        raise PlanError(layered(
            f"run plan 'config.intent' names source {source!r}, which "
            f"the {route!r} route cannot run from an intent.",
            "`gpuwm domain` writes a [case_data] table -- the declared "
            "inputs a config-driven run needs -- only for era5, because "
            "the config-driven route decodes native GRIB1 = ERA5 today "
            "(gpuwm/domain_wizard.py:3445).  A gfs or hrrr emission is "
            "a real config, but its consumer is the native/prepared "
            "front door, and that is not a run-plan route yet.  Use "
            "source = \"era5\", or supply a complete config with "
            "config.path / config.inline."))
    return dict(intent)


def intent_arguments(intent: Mapping[str, Any], *, out: Path
                     ) -> list[str]:
    """The ``gpuwm domain`` argv one intent block spells.

    Exposed because a front end that wants to show the reader the
    command behind their form has to be able to ask for it, and
    reconstructing it from the flag table would be a second copy.
    """

    arguments: list[str] = []
    for key in sorted(intent):
        flag = _INTENT_FLAGS[key]
        value = intent[key]
        if isinstance(value, bool):
            raise PlanError(
                f"run plan 'config.intent.{key}' must be a value, not a "
                "flag; every wizard option this front door exposes "
                "takes one")
        if isinstance(value, (list, tuple)):
            arguments.append(flag)
            arguments.extend(str(item) for item in value)
            continue
        arguments.extend((flag, str(value)))
    arguments.extend(("--out", str(out)))
    return arguments


#: Every ``run_options`` key this module understands, with the value a
#: plan that omits it gets.  A route declares which subset it supports;
#: an option the route does not support is refused, never accepted and
#: dropped.
_RUN_OPTION_DEFAULTS: dict[str, Any] = {
    "device": None,
    "dry_run": False,
    "restart": None,
    "health_debug": False,
    "data_dir": None,
    "geog_root": None,
}


def _run_option(key: str, value: object, base: Path) -> Any:
    label = f"run plan 'run_options.{key}'"
    if key in ("dry_run", "health_debug"):
        if not isinstance(value, bool):
            raise PlanError(f"{label} must be true or false")
        return value
    if key == "device":
        if value is None:
            return None
        if isinstance(value, bool):
            raise PlanError(f"{label} must be a GPU index or full GPU UUID")
        if isinstance(value, int):
            if value < 0:
                raise PlanError(f"{label} GPU index must be nonnegative")
            return str(value)
        selector = _nonempty_string(value, label)
        if selector.isdigit() or selector.startswith("GPU-"):
            return selector
        raise PlanError(
            f"{label} must be a nonnegative GPU index or full GPU UUID, "
            f"got {selector!r}")
    if key in ("restart", "data_dir", "geog_root"):
        return None if value is None else str(_absolute(value, base, label))
    raise PlanError(f"{label} is not a run option this build understands")


def _absolute(value: object, base: Path, label: str) -> Path:
    text = _nonempty_string(value, label)
    path = Path(text)
    return path if path.is_absolute() else (base / path).resolve()


def _validate_fetch_arguments(arguments: Sequence[str]) -> None:
    """Refuse a ``fetch.args`` list ``gpuwm fetch`` would refuse anyway.

    The check is the REAL parser, built from
    :func:`gpuwm.cli.build_parser` -- there is no second copy of the
    fetch flag table here, so a flag added there is accepted here on the
    same commit, and a typo is refused before the run claims a
    directory rather than an hour later.
    """

    from gpuwm.cli import build_parser

    parser = build_parser()
    try:
        parser.parse_args(["fetch", *arguments])
    except SystemExit as stop:
        raise PlanError(layered(
            "run plan 'fetch.args' is not a valid `gpuwm fetch` "
            "argument list; argparse refused it above.",
            "The list is handed to gpuwm's own fetch parser verbatim, "
            "so anything `gpuwm fetch` accepts is accepted here and "
            "nothing else is.")) from stop


def domain_size_floor() -> dict[str, Any]:
    """The smallest domain this engine will size, from the engine.

    Studio asked for the number.  It is DERIVED here rather than
    copied: :func:`gpuwm.domain_wizard._dims_for_scale` at the fit
    loop's own ``_MIN_SCALE`` is what the wizard actually bottoms out
    at, so this answer moves when the wizard moves.  A constant
    transcribed into a front end is a number that is right until
    somebody tunes the bracket, and then wrong silently and forever.

    Reaching for two private names is the price of that, and it is the
    right way round: a stale duplicate is worse than a tight coupling
    that breaks loudly.
    """

    from gpuwm import domain_wizard

    nx, ny = domain_wizard._dims_for_scale(  # noqa: SLF001 - see docstring
        domain_wizard._MIN_SCALE, ())[0]  # noqa: SLF001
    return {
        "root_mass_points": {"nx": int(nx), "ny": int(ny)},
        "nest_span_mass_points": 12,
        "clearance_rows": domain_wizard._CLEARANCE_ROWS,  # noqa: SLF001
        "basis": (
            "the wizard's fit loop bisects grid scale between _MIN_SCALE "
            "and _MAX_SCALE; _MIN_SCALE is the smallest layout that "
            "still hosts the deepest ladder with full Davies/blend "
            "clearance.  A nest span below 12 mass points is refused "
            "outright.  Domain size is FITTED from the ladder and the "
            "VRAM budget -- there is no nx/ny input to set."),
    }


def load_plan(path: str | Path) -> RunPlan:
    """Read, hash and validate one plan document."""

    plan_path = Path(path)
    try:
        payload = plan_path.read_bytes()
    except OSError as error:
        raise PlanError(f"run plan {plan_path} could not be read: "
                        f"{error}") from error
    digest = hashlib.sha256(payload).hexdigest()
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanError(f"run plan {plan_path} is invalid JSON: "
                        f"{error}") from error
    return build_plan(raw, source=str(plan_path),
                      base_dir=plan_path.resolve().parent, sha256=digest)


# ---------------------------------------------------------------------------
# The event stream
# ---------------------------------------------------------------------------


def _jsonable(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class EventStream:
    """Append-only JSONL, to a file and to a mirror, under one lock.

    The lock is not decoration.  ``output_committed`` is raised on the
    per-domain wrfout writer's own daemon thread
    (:class:`gpuwm.io.wrfout.AsyncDomainWrfoutWriter`), so two threads
    genuinely do reach :meth:`emit` at once, and a JSONL line that
    interleaves with another is not recoverable by any reader.

    Every line is flushed as it is written.  A consumer tailing the file
    or reading the mirrored pipe sees each event when it happens, not
    when a buffer happens to fill.
    """

    #: Default for ``mirror``.  A sentinel rather than ``None`` because
    #: ``None`` has to be able to mean "no mirror at all" -- a caller
    #: that wanted the file only and got stdout anyway would have no
    #: spelling left for what it asked for.
    MIRROR_STDOUT = object()

    def __init__(self, path: str | Path, *, mirror=MIRROR_STDOUT):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mirror = sys.stdout if mirror is EventStream.MIRROR_STDOUT \
            else mirror
        self._lock = threading.Lock()
        self._sequence = 0
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")

    @property
    def sequence(self) -> int:
        """The sequence number of the last emitted event (0 before any)."""

        return self._sequence

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event; return the record exactly as written."""

        if event not in EVENT_TAGS:
            raise PlanError(
                f"{event!r} is not a run-plan event tag; known tags: "
                f"{list(EVENT_TAGS)}")
        shadowed = sorted(_ENVELOPE_KEYS & set(fields))
        if shadowed:
            raise PlanError(
                f"event {event!r} may not carry the envelope key(s) "
                f"{shadowed}")
        with self._lock:
            self._sequence += 1
            record: dict[str, Any] = {
                "schema_version": EVENT_SCHEMA,
                "sequence": self._sequence,
                "emitted_unix_ms": int(time.time() * 1000),
                "event": event,
            }
            record.update(fields)
            line = json.dumps(record, default=_jsonable)
            self._stream.write(line + "\n")
            self._stream.flush()
            if self._mirror is not None:
                self._mirror.write(line + "\n")
                self._mirror.flush()
        return record

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.flush()
                self._stream.close()

    def __enter__(self) -> "EventStream":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_events(path: str | Path, *,
                allow_partial_tail: bool = False) -> list[dict[str, Any]]:
    """Replay one event stream from the top.

    This is the whole of reattach's history half, and it is a plain
    read: the file is append-only and never rotated, so byte zero to
    EOF IS the run.

    A torn final line means the writer died between opening the write
    and flushing it.  That is refused by default rather than trimmed,
    because a reader that silently drops a partial line cannot tell the
    difference between "the run is still going" and "the run died
    here".  ``allow_partial_tail`` is the caller saying it has already
    established which.
    """

    events: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if number == len(lines) and allow_partial_tail:
                break
            raise PlanError(
                f"{path} line {number} is not valid JSON: {error}"
            ) from error
        if record.get("schema_version") != EVENT_SCHEMA:
            raise PlanError(
                f"{path} line {number} is not a {EVENT_SCHEMA} record: "
                f"schema_version is {record.get('schema_version')!r}")
        expected = len(events) + 1
        if record.get("sequence") != expected:
            raise PlanError(
                f"{path} line {number} has sequence "
                f"{record.get('sequence')!r}, expected {expected}; the "
                "stream is append-only and its sequence is dense, so a "
                "gap is a lost or reordered line, never a skipped one")
        events.append(record)
    return events


# ---------------------------------------------------------------------------
# The observer: gpuwm's own progress protocol, rendered as events
# ---------------------------------------------------------------------------


class RunObserver:
    """The ``progress_callback`` the runtime already accepts.

    Composition, not replacement.  Every call is forwarded verbatim to
    the supervisor's :class:`~gpuwm.supervisor.RuntimeHeartbeat`, which
    stays the ONLY writer of ``run-progress.json``: a run-plan run
    leaves the same heartbeat a ``gpuwm run`` leaves, byte-shaped the
    same way, readable by the same recovery code.  This class adds the
    event stream on top of it and publishes no run state of its own.

    It implements the runtime's whole duck-typed surface --
    ``__call__``, ``preparing``, ``starting``, ``complete``, ``failed``
    -- plus ``output_committed``, the one hook this work added
    (:func:`gpuwm.runtime._output_committed`).
    """

    def __init__(self, events: EventStream, *, heartbeat=None,
                 root_domain: int = 1):
        self._events = events
        self._heartbeat = heartbeat
        self._root_domain = int(root_domain)
        self._stage: str | None = None
        self._stage_phases: list[str] = []
        self._stage_started_wall = 0.0
        self._forecast_started_wall: float | None = None
        self._forecast_started_model: float | None = None
        self._committed = 0
        self._progress_events = 0

    # -- stage bookkeeping --------------------------------------------

    @property
    def stage(self) -> str | None:
        """The stage currently open, or ``None`` before/after the run."""

        return self._stage

    @property
    def outputs_committed(self) -> int:
        return self._committed

    def enter_stage(self, stage: str, *, phase: str | None = None) -> None:
        """Close whatever stage is open and open ``stage``."""

        if stage not in STAGES:
            raise PlanError(f"{stage!r} is not a run stage; known stages: "
                            f"{list(STAGES)}")
        if stage == self._stage:
            return
        self.finish_stage()
        self._stage = stage
        self._stage_phases = [] if phase is None else [phase]
        self._stage_started_wall = time.perf_counter()
        payload: dict[str, Any] = {"stage": stage}
        if phase is not None:
            payload["phase"] = phase
        self._events.emit("stage_started", **payload)

    def finish_stage(self, **fields: Any) -> None:
        """Close the open stage, if any."""

        if self._stage is None:
            return
        stage = self._stage
        self._stage = None
        self._events.emit(
            "stage_finished", stage=stage,
            wall_seconds=round(
                time.perf_counter() - self._stage_started_wall, 6),
            phases=list(self._stage_phases), **fields)

    def warn(self, code: str, message: str, **fields: Any) -> None:
        self._events.emit("warning", code=code, message=message, **fields)

    # -- the runtime's progress protocol ------------------------------

    def preparing(self, phase: str) -> None:
        if self._heartbeat is not None:
            self._heartbeat.preparing(phase)
        stage = _PHASE_STAGES.get(phase)
        if stage is None:
            self.warn(
                "unmapped_pipeline_phase",
                f"the pipeline reported preparation phase {phase!r}, "
                "which this front door has no stage for; it is being "
                "attributed to the open stage rather than dropped",
                phase=phase, stage=self._stage)
            self._stage_phases.append(phase)
            return
        if stage == self._stage:
            self._stage_phases.append(phase)
            return
        self.enter_stage(stage, phase=phase)

    def starting(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.starting()

    def __call__(self, *, model_elapsed_seconds: float, outer_step: int,
                 last_durable_wrfout=None, last_checkpoint=None,
                 phase: str = "synchronized-step",
                 step_wall_seconds: float = 0.0, **extra: Any) -> None:
        if self._heartbeat is not None:
            self._heartbeat(
                model_elapsed_seconds=model_elapsed_seconds,
                outer_step=outer_step,
                last_durable_wrfout=last_durable_wrfout,
                last_checkpoint=last_checkpoint, phase=phase,
                step_wall_seconds=step_wall_seconds, **extra)
        model_seconds = float(model_elapsed_seconds)
        if self._stage != "forecast":
            self.enter_stage("forecast", phase=phase)
            self._forecast_started_wall = time.perf_counter()
            self._forecast_started_model = model_seconds
        wall_seconds = (
            0.0 if self._forecast_started_wall is None
            else time.perf_counter() - self._forecast_started_wall)
        advanced = model_seconds - (self._forecast_started_model or 0.0)
        payload: dict[str, Any] = {
            "domain": self._root_domain,
            "outer_step": int(outer_step),
            "model_seconds": model_seconds,
            "wall_seconds": round(wall_seconds, 6),
            "phase": phase,
        }
        # speed_x is a rate, and a rate over no elapsed wall is not a
        # large number, it is an undefined one.  Reported as null rather
        # than as an infinity a consumer would have to special-case.
        payload["speed_x"] = (round(advanced / wall_seconds, 4)
                              if wall_seconds > 0.0 and advanced > 0.0
                              else None)
        step_ms = float(step_wall_seconds) * 1000.0
        payload["step_ms"] = round(step_ms, 3) if step_ms > 0.0 else None
        if last_checkpoint is not None:
            payload["last_checkpoint"] = str(last_checkpoint)
        self._progress_events += 1
        self._events.emit("model_progress", **payload)

    def stage_progress(self, *, phase: str, elapsed_seconds: float,
                       model_seconds: float, status=None) -> None:
        """Model progress observed from a stage's published progress file.

        The coarse arm of ``model_progress``, for a route whose stage
        runs as a subprocess: the numbers are the stage's own, read from
        the artifact it republishes, and the cadence is that stage's
        rather than every step.  It carries ``step_ms: null`` and says
        where it came from in ``source``, so a consumer can tell a
        polled sample from a per-step one instead of assuming.
        """

        speed = (round(model_seconds / elapsed_seconds, 4)
                 if elapsed_seconds > 0.0 and model_seconds > 0.0 else None)
        self._events.emit(
            "model_progress", domain=self._root_domain,
            model_seconds=float(model_seconds),
            wall_seconds=round(float(elapsed_seconds), 6),
            speed_x=speed, step_ms=None, phase=phase,
            status=status, source="stage_progress_file")

    def output_committed(self, *, domain: int, valid_time, path) -> None:
        """One wrfout is durable on disk.  Raised from the writer.

        On the domain-tree route this arrives on the per-domain writer
        thread, after ``WrfoutWriter.close`` has fsynced, self-validated
        and renamed the temporary onto its final name -- so the event is
        emitted for a file that exists and passes its own inventory
        check, never for one that is merely queued.
        """

        self._committed += 1
        self._events.emit(
            "output_committed", domain=int(domain),
            valid_time=(valid_time.isoformat()
                        if isinstance(valid_time, datetime)
                        else str(valid_time)),
            path=str(path))

    def complete(self, model_elapsed_seconds: float) -> None:
        if self._heartbeat is not None:
            self._heartbeat.complete(model_elapsed_seconds)

    def failed(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.failed()


# ---------------------------------------------------------------------------
# Resolution: the snapshot and every value nobody typed
# ---------------------------------------------------------------------------


def _config_snapshot(exp, data) -> dict[str, Any]:
    """The fully resolved configuration, as JSON.

    ``dataclasses.asdict`` over the loaded config pair: the snapshot is
    the objects the model will actually run, not a re-reading of the
    TOML, so a value the loader derived appears here at its derived
    value.
    """

    snapshot = {
        "experiment": dataclasses.asdict(exp),
        # None on the prepared route: those configs declare no
        # [case_data], because their inputs are bound by the prepared
        # cache rather than named in the TOML.  Reported as null rather
        # than omitted, so the key's absence is never ambiguous.
        "case_data": None if data is None else dataclasses.asdict(data),
    }
    return json.loads(json.dumps(snapshot, default=_jsonable))


def _schema_default_resolutions(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every ``[experiment]`` key the document did not spell.

    Mechanical, from the dataclass itself: a field with a declared
    default that the TOML did not set was chosen by the schema, not by
    the author, and that is exactly what ``automatic_resolutions``
    exists to say out loud.
    """

    from gpuwm.experiment import ExperimentConfig

    table = raw.get("experiment")
    spelled = set(table) if isinstance(table, dict) else set()
    resolutions = []
    for field in dataclasses.fields(ExperimentConfig):
        if field.default is dataclasses.MISSING:
            continue
        if field.name in spelled:
            continue
        resolutions.append({
            "scope": "experiment", "key": field.name,
            "value": _jsonable_scalar(field.default),
            "basis": "schema_default"})
    return resolutions


def _jsonable_scalar(value: object) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _jsonable(value)


def _timestep_resolutions(exp) -> list[dict[str, Any]]:
    """The per-domain timestep, and how each one came to be.

    The root domain's step is the ``time_step`` (plus its optional
    rational correction) the author wrote.  Every child's is DERIVED by
    dividing the parent's down ``parent_time_step_ratio`` -- a number
    nobody typed, which changes the model's answer, and which until now
    appeared only inside a printed report.
    """

    resolutions = []
    for domain in exp.domains:
        derived = domain.parent_id != 0
        resolutions.append({
            "scope": "domain", "grid_id": domain.grid_id, "key": "dt",
            "value": float(domain.run.dt),
            "exact": str(exp.dt_exact(domain.grid_id)),
            "basis": ("derived_from_parent_time_step_ratio" if derived
                      else "declared_time_step"),
            "note": (f"parent d{domain.parent_id:02d} step divided by "
                     f"parent_time_step_ratio={domain.parent_time_step_ratio}"
                     if derived else
                     f"time_step={domain.time_step} "
                     f"+{domain.time_step_fract_num}/"
                     f"{domain.time_step_fract_den}")})
    return resolutions


def generate_intent_config(plan: RunPlan, *, destination: Path
                           ) -> tuple[Path, list[dict[str, Any]]]:
    """Write this plan's intent out as a config, using ``gpuwm domain``.

    Delegation, not composition.  The wizard exposes clean pieces --
    ``fit_ladder``, ``render_config`` -- but between them sits
    ``domain_main``'s orchestration: cycle resolution, projection
    selection by latitude, the fetch area hint and its per-source
    coverage gate, forcing cadence, the ``[case_data]`` block, profile
    resolution, and a full round trip through the real loader before
    anything is written.  Calling the pieces would mean re-deriving all
    of that here, which is the forked config logic this front door
    exists to avoid.  So the wizard's own entry point runs, through the
    wizard's own parser, and this function only supplies ``--out`` and
    reads what landed.

    The wizard is a talker -- it prints its sizing table, its resolved
    cycle, its gray-zone advisories, its next steps.  Its caller has
    already redirected stdout to stderr, so all of that reaches the
    reader and none of it reaches the machine channel.
    """

    from gpuwm.cli import build_parser

    destination.mkdir(parents=True, exist_ok=True)
    out = destination / GENERATED_CONFIG_NAME
    arguments = intent_arguments(plan.config_intent, out=out)
    try:
        args = build_parser().parse_args(["domain", *arguments])
    except SystemExit as stop:
        raise PlanError(layered(
            "run plan 'config.intent' is not a valid `gpuwm domain` "
            "request; argparse refused it above.",
            "The intent block is handed to the wizard's own parser "
            "verbatim, so anything `gpuwm domain` accepts is accepted "
            "here and nothing else is.")) from stop
    args.interactive = False
    from gpuwm.domain_wizard import DomainFitError, domain_main

    try:
        code = domain_main(args)
    except DomainFitError as error:
        # The one refusal a front end most needs the numbers from: the
        # requested shape does not fit the requested card.  The wizard's
        # own sentence is carried verbatim -- it already names the
        # budget, the layout it bottomed out at and what to change --
        # and the structural floor is attached beside it so a form can
        # bound its own inputs instead of guessing.
        raise PlanError(layered(
            f"run plan 'config.intent' does not fit: {error}",
            "The smallest domain this engine will size is "
            f"{json.dumps(domain_size_floor(), indent=2)}")) from error
    except ValueError as error:
        # Every other refusal the wizard raises, including the loader's
        # own when it round-trips the emitted bytes before writing them
        # -- an output cadence that is not a whole number of a domain's
        # steps arrives here.  Re-raised as a PlanError so it reaches
        # the caller as a refused plan rather than a failed run, with
        # the engine's sentence intact.
        raise PlanError(
            f"run plan 'config.intent' was refused: {error}") from error
    if code:
        raise PlanError(
            f"`gpuwm domain` exited {code} for this plan's intent; the "
            "refusal it printed is above")
    if not out.is_file():
        raise PlanError(
            f"`gpuwm domain` reported success but wrote no {out.name}")

    resolutions = [{
        "scope": "config", "key": "generated_config",
        "value": str(out), "basis": "intent_route",
        "note": "written by `gpuwm domain " + " ".join(arguments) + "`"}]
    # The wizard resolves `latest` into a concrete cycle and writes THAT
    # into the emitted [fetch] table (never the query -- domain_wizard
    # :3210).  Reading it back is how the resolved cycle is reported
    # without resolving it a second time and risking a different answer.
    emitted = tomllib.load(io.BytesIO(out.read_bytes()))
    hints = emitted.get("fetch") or {}
    if "cycle" in hints:
        requested = str(plan.config_intent.get("cycle", ""))
        resolutions.append({
            "scope": "fetch", "key": "cycle",
            "value": hints["cycle"],
            "basis": ("resolved_latest"
                      if requested.strip().lower() == "latest"
                      else "declared"),
            "note": (
                "`latest` probed the mirrors for the newest cycle "
                "complete through the end of the requested window; the "
                "concrete cycle is recorded so this run is reproducible"
                if requested.strip().lower() == "latest"
                else "as declared in the intent")})
    for domain in emitted.get("domain") or ():
        if "nx" in domain and "ny" in domain:
            resolutions.append({
                "scope": "domain", "grid_id": domain.get("grid_id"),
                "key": "nx_ny",
                "value": [domain["nx"], domain["ny"]],
                "basis": "fitted_to_vram_budget",
                "note": "domain size is fitted by the wizard's estimator "
                        "loop from the ladder and the card budget; it is "
                        "not an input"})
    return out, resolutions


def declared_inputs(data) -> list[dict[str, str]]:
    """Every file/directory a ``[case_data]`` block declares, and whether
    it is there yet.

    Used twice: to report readiness in a planning answer, and as the
    gate after the fetch stage.  One function, so "what this run needs"
    cannot mean two different sets.
    """

    entries: list[tuple[str, Path, str]] = [
        *(("forcing", path, "file") for path in data.forcing),
        ("vtable", data.vtable, "file"),
        ("wps_namelist", data.wps_namelist, "file"),
        ("geog_root", data.geog_root, "directory"),
    ]
    overlay = getattr(data, "water_temperature_overlay", None)
    if overlay is not None:
        entries.append(("water_temperature_overlay", overlay, "file"))
    orography = getattr(data, "source_orography", None)
    if orography is not None and getattr(orography, "path", None):
        entries.append(("source_orography", orography.path, "file"))
    return [{
        "role": role, "path": str(path), "kind": kind,
        "present": path.is_dir() if kind == "directory" else path.is_file(),
    } for role, path, kind in entries]


def resolve_plan(plan: RunPlan, *, generate_into: Path | None = None,
                 require_inputs: bool = True) -> dict[str, Any]:
    """Load a plan's config through the real seam and describe it.

    Everything ``--resolve`` prints, and everything the ``resolved_plan``
    event carries, is built here -- one function, so the document a
    caller inspects before a run and the event it receives during one
    cannot drift apart.
    """

    from gpuwm.case_data import load_experiment_case_bytes

    resolutions = list(plan.automatic_resolutions)
    generated_text = None
    scratch: tempfile.TemporaryDirectory | None = None
    try:
        if plan.config_intent is not None:
            # A query mode generates into a throwaway directory and
            # hands the text back in its document; a run generates into
            # its own run directory, where the config and the WPS
            # namelist the [case_data] block references have to STAY --
            # they are inputs to the run and provenance afterwards.
            if generate_into is None:
                scratch = tempfile.TemporaryDirectory(prefix="gpuwm-intent-")
                destination = Path(scratch.name)
            else:
                destination = generate_into
            warnings_generated: list[dict[str, str]] = []
            with collect_warnings(warnings_generated):
                generated, generated_resolutions = generate_intent_config(
                    plan, destination=destination)
            resolutions.extend(generated_resolutions)
            payload = generated.read_bytes()
            generated_text = payload.decode("utf-8")
            config_source = str(generated)
            base_dir = generated.parent
        else:
            warnings_generated = []
            payload = plan.config_bytes()
            config_source = (
                str(plan.config_path) if plan.config_path is not None
                else f"{plan.source}#config.inline")
            base_dir = (plan.config_path.parent
                        if plan.config_path is not None
                        else plan.config_base_dir)

        warnings: list[dict[str, str]] = list(warnings_generated)
        with collect_warnings(warnings):
            if ROUTES[plan.route].needs_case_data:
                exp, data = load_experiment_case_bytes(
                    payload, source=config_source, base_dir=base_dir,
                    require_inputs=require_inputs)
            else:
                # The prepared route's configs declare no [case_data]:
                # their inputs are bound by the prepared cache, not
                # named in the TOML.  experiment_from_text is the
                # wizard's own round-trip loader for exactly that shape
                # -- same build_experiment seam, [fetch] validated,
                # [case_data] split off if present.
                from gpuwm.domain_wizard import experiment_from_text

                exp = experiment_from_text(
                    payload.decode("utf-8"), source=config_source)
                data = None
        raw = tomllib.load(io.BytesIO(payload))
    finally:
        if scratch is not None:
            scratch.cleanup()

    source = config_source
    inputs = [] if data is None else declared_inputs(data)
    resolutions.extend(_schema_default_resolutions(raw))
    resolutions.extend(_timestep_resolutions(exp))
    resolutions.append({
        "scope": "execution", "key": "execution_mode",
        "value": "in_process", "basis": "front_door_contract",
        "note": "run-plan integrates in THIS process rather than "
                "re-executing under gpuwm's own supervisor, so the pid "
                "in run-manifest.json is the pid doing the model work "
                "and the caller owns restart policy"})

    return {
        "schema": RESOLVE_SCHEMA,
        "plan": {
            "name": plan.name, "route": plan.route,
            "source": plan.source, "sha256": plan.sha256,
            "config_kind": plan.config_kind,
            "config_source": source,
            "config_sha256": hashlib.sha256(payload).hexdigest(),
            "run_dir": str(plan.run_dir),
            "fetch_args": (list(plan.fetch_arguments)
                           if plan.fetch_arguments else None),
            "run_options": dict(plan.run_options),
        },
        # The generated TOML, verbatim.  An intent plan's caller never
        # typed this text, so it is the one thing they cannot look up:
        # a front end shows it, and a reader can check what the wizard
        # decided on their behalf before anything runs.
        "generated_config": generated_text,
        "configuration": _config_snapshot(exp, data),
        "declared_inputs": inputs,
        "inputs_present": all(entry["present"] for entry in inputs),
        "domain_size_floor": domain_size_floor(),
        "automatic_resolutions": resolutions,
        "warnings": warnings,
    }, exp, data


class collect_warnings:
    """Capture :func:`gpuwm.explain.warn` output as structured records.

    The library's warnings are one-line stderr sentences by design, and
    they stay that way.  This attaches a sink to the same call so a
    machine consumer receives them as fields instead of having to
    recognize them in a text stream.
    """

    def __init__(self, sink: list[dict[str, str]]):
        self._sink = sink
        self._observer = None

    def __enter__(self) -> "collect_warnings":
        from gpuwm import explain

        def observer(record: Mapping[str, str]) -> None:
            self._sink.append(dict(record))

        self._observer = observer
        explain.add_warning_observer(observer)
        return self

    def __exit__(self, *exc_info) -> None:
        from gpuwm import explain

        if self._observer is not None:
            explain.remove_warning_observer(self._observer)
            self._observer = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Route:
    """One existing front door, named so a plan can ask for it.

    Route names are generic on purpose.  A route is a way of running the
    model, never a particular experiment: nothing here may name a case.
    """

    name: str
    summary: str
    run_options: frozenset[str]
    execute: Callable[..., Any]
    #: Whether this route's configs carry a ``[case_data]`` table.  The
    #: config-driven route requires one (it IS the declared-input
    #: contract); the prepared route's configs have none, because the
    #: prepared cache binds the inputs instead.
    needs_case_data: bool = True


def _execute_experiment_route(plan: RunPlan, *, exp, data, config_path,
                              observer: RunObserver) -> Mapping[str, Any]:
    """The config-driven experiment route: what ``gpuwm run CONFIG`` runs.

    One call, into the same :func:`gpuwm.runtime.run_experiment` the
    ``run`` subcommand reaches, with this module's observer in the
    ``progress_callback`` slot that function already had.
    """

    from gpuwm import runtime

    restart = plan.run_options.get("restart")
    summary = runtime.run_experiment(
        exp, data, plan.run_dir,
        restart=None if restart is None else Path(restart),
        progress_callback=observer,
        health_debug=bool(plan.run_options.get("health_debug")))
    return {
        "wrfout_count": len(summary.wrfout_paths),
        "completed_seconds": float(summary.completed_seconds),
        "nan_free": bool(summary.nan_free),
        "restarted": restart is not None,
    }


#: ``gpuwm go``'s six stage labels, mapped to the run-plan stage each
#: belongs to.  The chain's order is authority, fetch, manifest,
#: prepare, forecast, render -- so ``prepare`` opens, closes for the
#: download, and opens again.  Stages repeat on this route and the
#: pairs stay strictly ordered; a consumer keyed on "which stage is
#: open" reads it correctly either way.  The go label itself rides on
#: the event as the stage's phase, so nothing is flattened away.
_GO_STAGES = {
    "authority": "prepare",
    "fetch": "fetch",
    "manifest": "prepare",
    "prepare": "prepare",
    "forecast": "forecast",
    "render": "finalize",
}


class _GoObserver:
    """``gpuwm go``'s stage hooks, rendered onto the run-plan stages.

    ``go`` reports at stage granularity and hands over the running
    stage's own published progress file; the forecast stage, hosted in
    process, additionally drives this object as the RUNNER's progress
    callback -- so the same object is both the chain observer and the
    model observer, and the events interleave in real order.
    """

    def __init__(self, observer: RunObserver):
        self._observer = observer

    # -- gpuwm go's chain hooks ---------------------------------------

    def stage_begin(self, *, label: str, command) -> None:
        self._observer.enter_stage(_GO_STAGES[label], phase=label)

    def stage_heartbeat(self, *, label: str, elapsed_seconds: float,
                        progress) -> None:
        # A subprocess stage can only be observed through what it
        # publishes.  The forecast stage runs in process and reports
        # per step through __call__ below, so this is the honest
        # coarse signal for the stages that do not.
        if not isinstance(progress, dict):
            return
        model_seconds = progress.get("model_elapsed_seconds")
        if not isinstance(model_seconds, (int, float)):
            return
        self._observer.stage_progress(
            phase=label, elapsed_seconds=elapsed_seconds,
            model_seconds=float(model_seconds),
            status=progress.get("status"))

    def stage_end(self, *, label: str, exit_code: int, ok: bool,
                  elapsed_seconds: float, progress) -> None:
        if not ok:
            self._observer.warn(
                "chain_stage_failed",
                f"`gpuwm go` stage {label!r} exited {exit_code}; no later "
                "stage ran, because each consumes the previous one's "
                "output", stage=label, exit_code=exit_code)

    # -- the runner's progress protocol, for the hosted forecast ------

    def __call__(self, **event) -> None:
        self._observer(**event)

    def preparing(self, phase: str) -> None:
        self._observer.preparing(phase)

    def output_committed(self, **fields) -> None:
        self._observer.output_committed(**fields)

    def starting(self) -> None:
        self._observer.starting()

    def complete(self, model_elapsed_seconds: float) -> None:
        self._observer.complete(model_elapsed_seconds)

    def failed(self) -> None:
        self._observer.failed()


def _execute_prepared_route(plan: RunPlan, *, exp, data, config_path,
                            observer: RunObserver) -> Mapping[str, Any]:
    """The native/prepared route: ``gpuwm go``'s chain, in this process.

    ``gpuwm go`` IS the documented sequence for this route -- authority,
    fetch, manifest, prepare, forecast, render, with the integrity
    digests relayed between them out of each stage's artifacts.  This
    function does not re-implement one step of it.  It builds the same
    argparse namespace the ``go`` subcommand builds, from this plan, and
    hands ``go_main`` an observer.

    The plan's own ``[fetch]`` block is not used here: ``go`` runs the
    fetch itself, from the ``[fetch]`` hints the wizard wrote into the
    config, which is where the resolved cycle and the sized area
    already live.
    """

    from gpuwm.cli import build_parser
    from gpuwm.go_cli import go_main

    tokens = ["go", str(config_path), "--outdir",
              str(plan.run_dir / "chain")]
    # Every intent key whose delivery is a `gpuwm go` flag, forwarded.
    # Driven off _INTENT_DELIVERY rather than written out here, so a key
    # that gains a flag is carried by declaring it in one table instead
    # of by remembering to edit this function too.
    intent = plan.config_intent or {}
    for key, delivery in sorted(_INTENT_DELIVERY.items()):
        if not delivery.startswith("go:"):
            continue
        # An explicit run option wins over the intent's copy: it is the
        # later and more specific statement, and it is the only way a
        # config.path plan can say these at all.
        value = plan.run_options.get(key) or intent.get(key)
        if value:
            tokens += [delivery.split(":", 1)[1], str(value)]
    args = build_parser().parse_args(tokens)
    code = go_main(args, observer=_GoObserver(observer))
    if code:
        raise RuntimeError(
            f"`{' '.join(tokens)}` exited {code}; the stage that stopped "
            "the chain is named in the failed event's warning above")
    # The chain's own completion signals, from the artifacts it leaves
    # -- `go`'s standing rule, and the only honest source here: this
    # function did not integrate anything, the hosted runner did, and it
    # publishes what it finished.
    #
    # These used to be None, and None reached `heartbeat.complete`,
    # whose float() raised INSIDE the try that emits `failed`.  So every
    # successful prepared run ended by announcing failure, after `go`
    # had already printed its validity PASS.  A consumer that trusts the
    # contract marked every good run failed.
    chain = plan.run_dir / "chain"
    forecast = chain / "run"
    progress = _read_json_object(forecast / "progress.json")
    report = _read_json_object(forecast / "report.json")
    wrfouts = sorted(forecast.glob("**/wrfout_*"))

    completed_seconds = progress.get("model_elapsed_seconds")
    if not isinstance(completed_seconds, (int, float)):
        completed_seconds = 0.0
    frame_count = progress.get("frame_count")
    status = report.get("status") or progress.get("status")
    return {
        "wrfout_count": (len(wrfouts) if wrfouts
                         else int(frame_count or 0)),
        "completed_seconds": float(completed_seconds),
        # The runner gates its own NaN health and refuses rather than
        # reporting; a PASS is the statement.  Reported as the status it
        # actually is, never as a bool this function invented.
        "nan_free": None if status is None else status == "PASS",
        "status": status,
        "chain_root": str(chain),
        "report": (str(forecast / "report.json")
                   if (forecast / "report.json").is_file() else None),
        "restarted": False,
    }


ROUTES: dict[str, Route] = {
    "prepared": Route(
        name="prepared",
        summary="the native prepared-cache route: authority, fetch, "
                "manifest, preparation, forecast and render, in the "
                "documented order (what `gpuwm go CONFIG` executes) -- "
                "for sources the config-driven route cannot decode",
        run_options=frozenset({*_RUN_OPTION_DEFAULTS, "data_dir",
                               "geog_root"}),
        needs_case_data=False,
        execute=_execute_prepared_route),
    "experiment": Route(
        name="experiment",
        summary="the config-driven experiment route: one experiment TOML "
                "with its [case_data] inputs, prepared and integrated in "
                "this process (what `gpuwm run CONFIG` executes)",
        run_options=frozenset(_RUN_OPTION_DEFAULTS)
        - {"data_dir", "geog_root"},
        execute=_execute_experiment_route),
}


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def write_manifest(plan: RunPlan, *, run_dir: Path, events_path: Path,
                   run_id: str, started_at_utc: str) -> Path:
    """Publish the attach manifest before any work starts.

    It names every stream a consumer may want, including the two this
    module does not own: the supervisor's heartbeat and its failure
    capsule.  A front end should never have to know those filenames.
    """

    from gpuwm.supervisor import (FAILURE_CAPSULE_NAME,
                                  FAILURE_CAPSULE_SCHEMA, HEARTBEAT_NAME,
                                  HEARTBEAT_SCHEMA, atomic_write_json)

    document = {
        "schema": MANIFEST_SCHEMA,
        "name": plan.name,
        "route": plan.route,
        "run_id": run_id,
        "pid": os.getpid(),
        "started_at_utc": started_at_utc,
        "plan_source": plan.source,
        "plan_sha256": plan.sha256,
        "run_dir": str(run_dir),
        "outputs_dir": str(run_dir),
        "events_path": str(run_dir / EVENTS_FILENAME),
        "events_schema": EVENT_SCHEMA,
        "progress_path": str(run_dir / HEARTBEAT_NAME),
        "progress_schema": HEARTBEAT_SCHEMA,
        "failure_capsule_path": str(run_dir / FAILURE_CAPSULE_NAME),
        "failure_capsule_schema": FAILURE_CAPSULE_SCHEMA,
        "reattach": (
            "read progress_path for CURRENT state, replay events_path "
            "from byte zero for HISTORY, then tail it for live detail; "
            "the heartbeat is the durable anchor, the event stream is "
            "the fine-grained feed"),
    }
    path = run_dir / MANIFEST_FILENAME
    atomic_write_json(path, document)
    return path


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


#: Remedies for the failure classes whose cause is known from the class
#: alone.  Absent is ``None``, never a guess: a wrong remedy costs a
#: reader more than no remedy.
_REMEDIES = {
    "ModuleNotFoundError": (
        "this route needs the GPU runtime (CuPy), which the base "
        "install does not include: pip install 'gpuwm[gpu]'"),
    "PlanError": "fix the plan document and re-run; nothing was started",
    "FileNotFoundError": "a declared input is not at the path the config "
                         "names; `gpuwm check CONFIG` names all of them",
}


def execute_plan(plan: RunPlan, *, events: EventStream) -> int:
    """Run one plan to completion, or to its ``failed`` event.

    Returns the process exit code: 0 when a ``completed`` event was the
    last line, nonzero when a ``failed`` one was.
    """

    from gpuwm.supervisor import (HEARTBEAT_NAME, RuntimeHeartbeat,
                                  utc_now)

    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at_utc = utc_now()
    run_id = hashlib.sha256(
        f"{plan.sha256}:{started_at_utc}:{os.getpid()}".encode("utf-8")
    ).hexdigest()[:16]
    manifest_path = write_manifest(
        plan, run_dir=run_dir, events_path=events.path, run_id=run_id,
        started_at_utc=started_at_utc)

    events.emit(
        "plan_accepted", name=plan.name, route=plan.route,
        plan_source=plan.source, plan_sha256=plan.sha256,
        run_dir=str(run_dir), manifest_path=str(manifest_path),
        events_path=str(events.path), pid=os.getpid(), run_id=run_id)

    stage = "prepare"
    observer: RunObserver | None = None
    try:
        device = plan.run_options.get("device")
        if device is not None:
            # Before anything can import cupy: the mask is read at
            # context creation, so setting it afterwards would select
            # nothing while appearing to.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        # A `latest` cycle is resolved BEFORE resolution reports, so
        # the resolved_plan event carries the concrete cycle rather than
        # the question the caller asked.
        fetch_arguments = plan.fetch_arguments
        cycle_resolutions: list[dict[str, Any]] = []
        if fetch_arguments is not None:
            stage = "fetch"
            fetch_arguments, cycle_resolutions, cycle_warnings = \
                resolve_fetch_cycle(fetch_arguments)
            for warning in cycle_warnings:
                events.emit("warning", **warning)
        stage = "prepare"

        # An intent plan generates into the run directory: the config
        # and the WPS namelist its [case_data] names are inputs to this
        # run, not scratch.
        # Resolved WITHOUT requiring the inputs to be on disk.  They may
        # not be yet: the fetch stage below is what puts them there, and
        # a plan that fetches its own forcing would otherwise be refused
        # for the absence of the thing it was about to download.  The
        # gate still happens -- after the fetch, before the model.
        resolution, exp, data = resolve_plan(
            plan, generate_into=run_dir, require_inputs=False)
        for warning in resolution["warnings"]:
            events.emit("warning", code="library_warning",
                        message=warning["action"], detail=warning["why"])
        events.emit(
            "resolved_plan",
            configuration=resolution["configuration"],
            automatic_resolutions=(
                list(resolution["automatic_resolutions"])
                + cycle_resolutions),
            generated_config=resolution["generated_config"],
            config_kind=plan.config_kind,
            config_sha256=resolution["plan"]["config_sha256"],
            config_source=resolution["plan"]["config_source"],
            domain_size_floor=resolution["domain_size_floor"],
            declared_inputs=resolution["declared_inputs"],
            inputs_present=resolution["inputs_present"],
            run_options=dict(plan.run_options))

        if plan.run_options.get("dry_run"):
            events.emit(
                "completed", dry_run=True, run_dir=str(run_dir),
                receipt_path=str(manifest_path),
                summary={"executed": False,
                         "reason": "run_options.dry_run resolved the plan "
                                   "and stopped before any device work"})
            return 0

        heartbeat = RuntimeHeartbeat(
            run_dir / HEARTBEAT_NAME, run_id=run_id,
            config_sha256=resolution["plan"]["config_sha256"],
            started_at_utc=started_at_utc)
        observer = RunObserver(
            events, heartbeat=heartbeat, root_domain=exp.root.grid_id)
        heartbeat.starting()

        if fetch_arguments is not None:
            stage = "fetch"
            observer.enter_stage("fetch")
            observer.finish_stage(fetch=_run_fetch(fetch_arguments, run_dir))

        # The gate the resolution above deferred.  Everything the config
        # declares must be on disk before the model starts; whatever was
        # going to supply it has now run.  Named in one refusal rather
        # than discovered one file at a time inside preparation.
        # Only where there is a [case_data] block to gate on.  The
        # prepared route's config declares no inputs -- its chain
        # fetches and binds them itself, and each of its stages refuses
        # what the previous one did not produce.
        missing = [] if data is None else [
            entry for entry in declared_inputs(data)
            if not entry["present"]]
        if missing:
            raise PlanError(layered(
                "declared input(s) this run needs are not on disk: "
                + ", ".join(f"{entry['role']} {entry['path']}"
                            for entry in missing) + ".",
                "The config names them in [case_data].  A plan with a "
                "[fetch] block downloads its own; without one, the data "
                "has to be there before the run starts."))

        # The route is handed a config that is a FILE.  `gpuwm go` takes
        # a path, the prepared chain binds that path's digest into every
        # stage, and an inline config has nowhere to be one -- so it is
        # materialized here, into the run directory, where it is also
        # the run's own provenance afterwards.
        config_path = Path(resolution["plan"]["config_source"])
        if not config_path.is_file():
            config_path = run_dir / GENERATED_CONFIG_NAME
            config_path.write_bytes(plan.config_bytes())
            events.emit(
                "warning", code="inline_config_materialized",
                message=f"the inline config was written to {config_path} "
                        "because this route binds a config file by path",
                path=str(config_path))

        stage = "prepare"
        # The pipeline opens prepare/initialize/forecast itself, through
        # the phases it already reports; the observer maps them.  Only
        # finalize is this front door's own, because the pipeline has no
        # word for it.
        summary = ROUTES[plan.route].execute(
            plan, exp=exp, data=data, config_path=config_path,
            observer=observer)

        stage = "finalize"
        observer.enter_stage("finalize")
        receipts = _receipts(run_dir)
        observer.finish_stage(receipts=receipts)
        heartbeat.complete(_finite_seconds(
            summary.get("completed_seconds")))

        events.emit(
            "completed", dry_run=False, run_dir=str(run_dir),
            receipt_path=receipts.get("certification_capsule",
                                      str(manifest_path)),
            receipts=receipts,
            outputs_committed=observer.outputs_committed,
            summary=dict(summary))
        return 0
    except BaseException as error:  # noqa: BLE001 - every exit is an event
        if observer is not None:
            stage = observer.stage or stage
            observer.finish_stage(outcome="failed")
            observer.failed()
        events.emit(
            "failed", stage=stage, error_class=type(error).__name__,
            message=str(error), run_dir=str(run_dir),
            remedy=_REMEDIES.get(type(error).__name__),
            receipts=_receipts(run_dir))
        if isinstance(error, KeyboardInterrupt):
            return 130
        return 1


def resolve_fetch_cycle(arguments: Sequence[str]
                        ) -> tuple[list[str], list[dict[str, Any]],
                                   list[dict[str, Any]]]:
    """Turn a ``--cycle latest`` into the concrete cycle it means.

    ``gpuwm fetch`` resolves ``latest`` itself, and correctly.  It is
    done HERE anyway, before the fetch runs, for one reason: a plan that
    records ``latest`` records a QUESTION, and the answer changes every
    six hours.  Resolving once, up front, and writing the concrete cycle
    into ``automatic_resolutions`` is what makes the receipt reproducible
    -- and it removes the window in which this front door reports one
    cycle while the fetch downloads the next.

    It is the same rule the wizard already applies to its emitted
    ``[fetch]`` table: "the RESOLVED cycle, never the literal 'latest'"
    (gpuwm/domain_wizard.py:3210).

    ``latest`` is matched case-insensitively.  ``gpuwm fetch`` itself
    compares bare equality while the wizard and the interactive door
    lower-case first, so ``--cycle Latest`` is accepted by two of the
    three front doors today; a machine interface should not inherit
    that coin flip, and the value handed onward is the canonical one.
    """

    from gpuwm.fetch import GFS_CONTAINER_SOURCES, resolve_latest_cycle

    arguments = list(arguments)
    try:
        position = arguments.index("--cycle")
        raw = arguments[position + 1]
    except (ValueError, IndexError):
        return arguments, [], []
    if raw.strip().lower() != "latest":
        return arguments, [], []

    source = "era5"
    if "--source" in arguments:
        source = arguments[arguments.index("--source") + 1]
    last_hour = 0
    if "--hours" in arguments:
        last_hour = int(arguments[arguments.index("--hours") + 1])
    if "--forecast-start-hour" in arguments:
        last_hour += int(
            arguments[arguments.index("--forecast-start-hour") + 1])

    cycle = resolve_latest_cycle(source, last_hour)
    concrete = cycle.strftime("%Y-%m-%dT%H")
    arguments[position + 1] = concrete

    resolutions = [{
        "scope": "fetch", "key": "cycle", "value": concrete,
        "basis": "resolved_latest",
        "note": "the newest cycle whose objects for the final requested "
                "hour are all published; a partially uploaded cycle "
                "cannot win, so the window is complete by construction"}]

    # A cycle that is not the newest one the clock allows means newer
    # cycles exist and are still publishing.  That is normal and not an
    # error -- but it means the run initializes from older data than the
    # caller may assume, which is worth one line.  Reported as a
    # warning, never a refusal.
    warnings: list[dict[str, Any]] = []
    step = 6 if source in GFS_CONTAINER_SOURCES else 1
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    behind = int((now - cycle).total_seconds() // 3600)
    if behind >= 2 * step:
        warnings.append({
            "code": "latest_cycle_is_not_the_newest",
            "message": (
                f"--cycle latest resolved to {concrete}Z for {source}, "
                f"which is about {behind} h old; newer cycles exist but "
                f"are not yet published through hour {last_hour}"),
            "cycle": concrete, "source": source,
            "age_hours": behind, "last_hour": last_hour})
    return arguments, resolutions, warnings


def _run_fetch(arguments: Sequence[str], run_dir: Path) -> dict[str, Any]:
    """Execute the plan's fetch through ``gpuwm fetch``'s own handler.

    Returns what the fetch actually did.  ``fetch_main`` answers only
    with an exit code, but it leaves a ``fetch-manifest.json`` beside
    the data naming the cycle, the hours and the files -- so the report
    comes from the receipt the fetch itself wrote, not from re-deriving
    anything here.
    """

    from gpuwm.cli import build_parser

    args = build_parser().parse_args(["fetch", *arguments])
    code = args.func(args)
    if code:
        raise RuntimeError(
            f"`gpuwm fetch {' '.join(arguments)}` exited {code}")

    from gpuwm.fetch import FETCH_MANIFEST_NAME

    report: dict[str, Any] = {"arguments": list(arguments)}
    out = getattr(args, "out", None)
    if out is not None:
        manifest = Path(out) / FETCH_MANIFEST_NAME
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                report["manifest_path"] = str(manifest)
                for key in ("cycle", "source", "forecast_hours",
                            "payload_bytes"):
                    if key in payload:
                        report[key] = payload[key]
    return report


def _read_json_object(path: Path) -> dict[str, Any]:
    """One JSON object from disk, or ``{}``.

    A stage that has not written its receipt yet is not an error to the
    reader of it, so an absent or half-written file reads as empty
    rather than raising -- the caller decides what a missing signal
    means.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _finite_seconds(value: object) -> float:
    """A model-time figure a heartbeat will accept, from anything.

    ``supervisor.Heartbeat`` requires a finite, non-negative float and
    refuses anything else at construction.  A route that has no number
    to give -- because its work happened in a subprocess, or because a
    receipt was not written -- must not be able to turn a completed run
    into a crash inside the arm that emits ``failed``.  That is exactly
    what happened once, so the coercion lives at the boundary rather
    than in each route's good intentions.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, number)


def _receipts(run_dir: Path) -> dict[str, str]:
    """The receipt artifacts that actually exist, by role.

    Only files present on disk are listed.  A path to a receipt that was
    never written is worse than its absence: the consumer opens it.
    """

    from gpuwm.certify.capsule import CAPSULE_FILENAME
    from gpuwm.supervisor import FAILURE_CAPSULE_NAME, HEARTBEAT_NAME

    known = {
        "certification_capsule": CAPSULE_FILENAME,
        "progress": HEARTBEAT_NAME,
        "failure_capsule": FAILURE_CAPSULE_NAME,
        "manifest": MANIFEST_FILENAME,
        "microphysics_transitions": "microphysics-transitions.json",
        "feedback_provenance": "feedback-provenance.json",
        "initial_perturbation": "initial-perturbation.json",
    }
    found = {}
    for role, filename in known.items():
        path = run_dir / filename
        if path.is_file():
            found[role] = str(path)
    return found


# ---------------------------------------------------------------------------
# Query modes
# ---------------------------------------------------------------------------


def estimate_plan(plan: RunPlan) -> dict[str, Any]:
    """What this plan will cost, from measured machinery only.

    VRAM comes from :mod:`gpuwm.core.preflight`'s itemization -- the
    same arithmetic ``gpuwm check`` reports, on the CPU, with no CUDA
    context created.  Output-frame COUNTS are exact.  Wall time is
    ``null``: this package has no measured rate for an arbitrary
    configuration, and a front end showing an invented duration would
    be showing gpuwm's name on a number gpuwm never measured.
    """

    from gpuwm.core.preflight import estimate_experiment

    resolution, exp, _data = resolve_plan(plan, require_inputs=False)
    estimate = estimate_experiment(exp)
    frames = []
    for domain in exp.domains:
        interval = float(domain.history_interval_s)
        count = (0 if interval <= 0.0
                 else int(exp.run_seconds // interval) + 1)
        frames.append({
            "domain": domain.grid_id,
            "history_interval_s": interval,
            "frames": count,
            "nx": domain.run.nx, "ny": domain.run.ny, "nz": domain.run.nz,
        })
    return {
        "schema": ESTIMATE_SCHEMA,
        "plan": resolution["plan"],
        "vram": {
            "estimate_bytes": int(estimate.alloc_estimate_bytes),
            "estimate_gib": round(
                estimate.alloc_estimate_bytes / 1024 ** 3, 4),
            "basis": "gpuwm.core.preflight.estimate_experiment "
                     "(the estimator `gpuwm check` reports; no device "
                     "context is created)",
        },
        "disk": {
            "frames": frames,
            "total_frames": sum(entry["frames"] for entry in frames),
            "bytes": None,
            "basis": "frame counts are exact from run_seconds and each "
                     "domain's history_interval_s; bytes-per-frame is "
                     "not measured by this package, so no byte figure "
                     "is reported rather than an invented one",
        },
        "download": {
            "bytes": None,
            "basis": ("no [fetch] in this plan" if plan.fetch_arguments
                      is None else
                      "download size is known only to the source mirror "
                      "at fetch time; `gpuwm fetch` reports it there"),
        },
        "wall_time": {
            "seconds": None,
            "basis": "this package publishes no measured rate for an "
                     "arbitrary configuration; the model_progress "
                     "events carry the real one from the first step",
        },
        "automatic_resolutions": resolution["automatic_resolutions"],
    }


def probe_environment(*, readiness: bool = True) -> dict[str, Any]:
    """This machine's device inventory and readiness, as one JSON document.

    The DEVICE half is read through NVML (``nvidia-smi``) and never
    through a CUDA context: capacity is the one device question that
    must be answerable without standing one up, and a front end asking
    "can I run?" must not become a compute contender on the card it is
    asking about.  That half is always safe to poll.

    The READINESS half delegates to :func:`gpuwm.doctor.collect_checks`,
    which verifies the estate for real rather than by presence -- and
    that includes a short-lived subprocess that imports CuPy and runs a
    2x2 matmul, which DOES create a context on the card.  Said plainly
    here because a caller polling a busy card needs to know which half
    costs something: pass ``readiness=False`` (``--no-readiness``) for
    the NVML-only document.
    """

    from gpuwm import __version__

    document: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "gpuwm_version": str(__version__),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pid": os.getpid(),
    }

    devices: list[dict[str, Any]] = []
    device_error = None
    try:
        from gpuwm.core.preflight import (device_physical_total_bytes,
                                          device_wide_used_bytes)
        from gpuwm.supervisor import query_gpus

        total = device_physical_total_bytes()
        used = device_wide_used_bytes()
        for identity in query_gpus():
            devices.append({
                "index": identity.index,
                "uuid": identity.uuid,
                "name": identity.name,
                "driver_version": identity.driver_version,
                "memory_total_bytes": total,
                "memory_used_bytes": used,
                "memory_free_bytes": (None if total is None
                                      else max(0, total - used)),
            })
    except Exception as error:  # noqa: BLE001 - a probe reports, never raises
        device_error = f"{type(error).__name__}: {error}"
    document["devices"] = devices
    document["device_query_error"] = device_error
    document["device_query_basis"] = (
        "NVML via nvidia-smi; no CUDA context is created by this probe. "
        "memory_total is the card's, memory_used is device-wide across "
        "every process, so free is what a new run could actually claim.")

    if not readiness:
        document["readiness"] = {
            "collected": False,
            "basis": "readiness was not requested; the estate check "
                     "creates a CUDA context and this document is the "
                     "poll-safe half"}
    else:
        try:
            from gpuwm.doctor import blocking_gaps, collect_checks

            checks = collect_checks()
            document["readiness"] = {
                "collected": True,
                "checks": [dataclasses.asdict(check)
                           if dataclasses.is_dataclass(check)
                           else dict(check.__dict__) for check in checks],
                "gaps": sum(1 for check in checks
                            if check.status == "missing"),
                "blocking_gaps": len(blocking_gaps(checks)),
                "ready": not blocking_gaps(checks),
                "basis": "gpuwm doctor's own checks, which verify by "
                         "execution and therefore create a CUDA context",
            }
        except Exception as error:  # noqa: BLE001 - a probe reports, never raises
            document["readiness"] = {
                "collected": False,
                "error": f"{type(error).__name__}: {error}"}
    document["routes"] = {
        name: route.summary for name, route in sorted(ROUTES.items())}
    document["schemas"] = {
        "plan": PLAN_SCHEMA, "event": EVENT_SCHEMA,
        "manifest": MANIFEST_SCHEMA, "resolve": RESOLVE_SCHEMA,
        "estimate": ESTIMATE_SCHEMA, "probe": PROBE_SCHEMA}
    return json.loads(json.dumps(document, default=_jsonable))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_plan_main(args: argparse.Namespace) -> int:
    """``gpuwm run-plan`` and ``python -m gpuwm.runplan``.

    Exit codes: 0 when the last event was ``completed``, 1 when it was
    ``failed``, 130 on a Ctrl-C (the shell's 128 + SIGINT, matching
    every other long-running command here).  The query modes exit 0 on
    a printed document and 2 on a refusal, argparse's own convention.
    """

    # Bound before any redirect: this is the machine channel, and the
    # document below is the only thing allowed onto it.
    machine_channel = sys.stdout

    def answer(document) -> int:
        machine_channel.write(json.dumps(
            document, indent=2, sort_keys=True, default=_jsonable) + "\n")
        machine_channel.flush()
        return 0

    if getattr(args, "probe", False):
        with contextlib.redirect_stdout(sys.stderr):
            document = probe_environment(
                readiness=not getattr(args, "no_readiness", False))
        return answer(document)
    if args.plan is None:
        raise PlanError(
            "gpuwm run-plan needs a PLAN.json, or --probe (which needs "
            "no plan)")

    plan = load_plan(args.plan)
    if getattr(args, "resolve", False):
        # The redirect covers resolution, not just execution: an intent
        # plan runs the wizard, and the wizard prints -- its resolved
        # cycle, its gray-zone advisories, its fit notes.  All of that
        # belongs to the reader on stderr; the document is the answer.
        with contextlib.redirect_stdout(sys.stderr):
            resolution, _exp, _data = resolve_plan(
                plan, require_inputs=False)
        return answer(resolution)
    if getattr(args, "estimate", False):
        with contextlib.redirect_stdout(sys.stderr):
            document = estimate_plan(plan)
        return answer(document)

    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # The EventStream binds the REAL stdout here, before the redirect
    # below moves everyone else's to stderr.  That split is the whole
    # promise of this front door: stdout is the machine channel and
    # carries JSONL and nothing else.
    #
    # It is not hypothetical.  The pipeline prints its resolved-config
    # report (runtime.py:1793/1852), its feedback warning, and the
    # wizard prints its resolved cycle -- all with plain print(), all
    # correct for a person, all landing in the middle of the stream a
    # consumer is calling json.loads on line by line.  The dry-run path
    # never reaches any of them, which is exactly why the subprocess
    # test that covered stdout purity passed while a real run did not.
    with EventStream(run_dir / EVENTS_FILENAME) as events:
        with contextlib.redirect_stdout(sys.stderr):
            return execute_plan(plan, events=events)


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Register the machine-facing execution front door."""

    parser = subparsers.add_parser(
        "run-plan",
        help="execute one versioned run plan and emit a structured "
             "event stream (JSONL to <run_dir>/events.jsonl and to "
             "stdout) that a program can consume without parsing any "
             "human output")
    parser.add_argument(
        "plan", type=Path, nargs="?", default=None, metavar="PLAN.json",
        help=f"a {PLAN_SCHEMA} document: which route to execute, which "
             "config to execute it with, and where the outputs land")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resolve", action="store_true",
        help="print the fully resolved configuration plus every "
             "automatic resolution as one JSON document, and run "
             "nothing")
    mode.add_argument(
        "--estimate", action="store_true",
        help="print this plan's VRAM estimate and output-frame counts "
             "as one JSON document, and run nothing")
    mode.add_argument(
        "--probe", action="store_true",
        help="print this machine's device inventory and runtime-estate "
             "readiness as one JSON document; needs no plan.  The "
             "device inventory is NVML only and creates no CUDA "
             "context; the readiness half runs `gpuwm doctor`'s checks, "
             "which verify by execution and do create one")
    parser.add_argument(
        "--no-readiness", dest="no_readiness", action="store_true",
        help="with --probe, report the device inventory only: the "
             "NVML-only half, safe to poll on a card that is busy")
    parser.set_defaults(func=run_plan_main)


def _module_entry(argv: Sequence[str] | None = None) -> int:
    """``python -m gpuwm.runplan PLAN.json``.

    Delegates the WHOLE invocation to :func:`gpuwm.cli.main`, not just
    the parser.  A second entry point that only shared the parser would
    also need its own refusal print boundary, and the first version of
    this function proved why that is a bug rather than duplication: it
    printed the ``[[explain]]`` sentinel straight to the terminal on a
    layered refusal, because choosing a layer is the boundary's job and
    this function was not it.  One boundary, two spellings.
    """

    from gpuwm.cli import main

    tokens = list(sys.argv[1:] if argv is None else argv)
    return main(["run-plan", *tokens])


if __name__ == "__main__":
    sys.exit(_module_entry())


__all__ = [
    "DEFAULT_OUTPUT_ROOT", "ESTIMATE_SCHEMA", "EVENTS_FILENAME",
    "EVENT_SCHEMA", "EVENT_TAGS", "MANIFEST_FILENAME", "MANIFEST_SCHEMA",
    "PLAN_SCHEMA", "PROBE_SCHEMA", "RESOLVE_SCHEMA", "ROUTES", "STAGES",
    "GENERATED_CONFIG_NAME",
    "EventStream", "PlanError", "Route", "RunObserver", "RunPlan",
    "build_plan", "collect_warnings", "declared_inputs",
    "domain_size_floor",
    "estimate_plan", "execute_plan", "generate_intent_config",
    "intent_arguments", "load_plan", "probe_environment", "read_events",
    "register_cli", "resolve_fetch_cycle", "resolve_plan",
    "run_plan_main", "write_manifest",
]
