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
CATALOG_SCHEMA = "gpuwm.run-plan.catalog.v1"

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
    "model_progress", "output_committed", "first_products_ready",
    "warning", "completed", "failed",
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
    # Read by the HRRR preparer as a flag; on the gfs chain the wizard
    # bakes the resulting physics into the config and `go` needs no
    # flag, so this is declared config-delivered and the HRRR arm reads
    # it off the plan directly.
    "physics_profile": "config",
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

#: Sources the prepared route can drive end to end.  gfs runs `gpuwm
#: go`'s chain; hrrr runs its own, which `go` refuses by construction.
_PREPARED_SOURCES = frozenset({"gfs", "hrrr"})

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
    if (not ROUTES[route].needs_case_data
            and source not in _PREPARED_SOURCES):
        raise PlanError(layered(
            f"run plan 'config.intent' names source {source!r}, which "
            f"the {route!r} route cannot drive.",
            "This route drives " + ", ".join(sorted(_PREPARED_SOURCES))
            + ".  An era5 intent belongs on route = \"experiment\", "
            "which consumes its [case_data] declarations directly."))
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
    "physics_profile": None,
    # `gpuwm render --products`' own spec: a comma-separated product
    # list, or `all`, or `none` to skip rendering entirely.  Absent
    # leaves the chain's default set exactly as it was.  NOT an intent
    # key: intent is the wizard's flag list one-for-one, and the wizard
    # writes configs rather than pictures -- there is no --render-products
    # flag for it to mirror.
    "render_products": None,
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
    if key == "render_products":
        return None if value is None else _nonempty_string(value, label)
    if key == "physics_profile":
        return None if value is None else _nonempty_string(value, label)
    if key in ("restart", "data_dir", "geog_root"):
        return None if value is None else str(_absolute(value, base, label))
    raise PlanError(f"{label} is not a run option this build understands")


# ---------------------------------------------------------------------------
# Moving nests: which chain can feed one, and at what price
# ---------------------------------------------------------------------------

#: How a ``[relocation]`` follow source gets its per-move statics on
#: each chain this front door can dispatch to.
#:
#: A moving nest needs child-resolution statics for a footprint nobody
#: knew about at preparation time.  There are exactly two honest ways to
#: have them, and one way to not have them:
#:
#: ``"case_data_ingest"``   the route holds the geography source for the
#:                          whole run and rebuilds each footprint at
#:                          move time.  Nothing to prepare, nothing to
#:                          price.
#: ``"statics_corridor"``   the preparation seals child-resolution
#:                          statics over the whole parent extent and the
#:                          tree runner crops them.  Run-plan's prepare
#:                          stage carries ``--statics-corridor``,
#:                          derived from the very same predicate that
#:                          indexes this table.
#: ``None``                 the chain's preparation emits neither, so a
#:                          follow config is REFUSED at resolve time.
#:                          Without this entry the run reaches the tree
#:                          runner's preflight instead -- after the
#:                          fetch and both preparation stages, which is
#:                          minutes of work to arrive at a refusal that
#:                          was knowable from the config alone.
#:
#: Every chain :func:`_chain_key` can return must appear here; a test
#: fails if one does not, so a chain cannot be added without answering
#: "and where does a moving nest get its statics on it?".
_FOLLOW_STATICS_DELIVERY: dict[str, str | None] = {
    "experiment": "case_data_ingest",
    "prepared:go": "statics_corridor",
    # Was ``None`` until the HRRR chain learned to seal one.  Its
    # corridor comes out of the stage that builds the children and holds
    # the GEOG root (``gpuwm.hrrr_hierarchy_direct --statics-corridor``)
    # rather than out of rw-wps, which is not on this path at all -- but
    # it is the same corridor module, the same sealed artifact and the
    # same digest relay, so the tree runner cannot tell the two apart
    # and this table says the same word for both.
    "prepared:hrrr": "statics_corridor",
}


#: Which stage of each corridor-sealing chain carries the flag.  Named
#: in the resolution note because "the prepare stage" is the wrong
#: sentence on HRRR -- its root preparer seals no corridor and would
#: refuse the flag -- and a reader who went looking for it there would
#: conclude the feature was not wired.  A completeness test requires an
#: entry for every chain whose delivery is ``statics_corridor``.
_CORRIDOR_STAGE = {
    "prepared:go": "the rw-wps prepare stage (gpuwm.source_cli)",
    "prepared:hrrr": "the hierarchy stage (gpuwm.hrrr_hierarchy_direct) "
                     "-- the stage that builds the children and holds "
                     "the geography root --",
}


def _chain_key(route: str, config_source: str | None) -> str:
    """Which chain this plan dispatches to.

    The prepared route is two chains wearing one name:
    :func:`_hrrr_chain` drives the HRRR tools, everything else goes to
    ``gpuwm go``'s rw-wps chain.  :func:`_execute_prepared_route`
    branches on this function rather than on its own copy of the test,
    so the chain that RUNS and the chain the refusal above was decided
    for are the same chain by construction.
    """
    if route != "prepared":
        return route
    return ("prepared:hrrr" if (config_source or "").lower() == "hrrr"
            else "prepared:go")


def follow_statics_decision(exp, *, chain: str) -> dict[str, Any] | None:
    """How a moving nest gets its statics on ``chain``, or ``None``.

    ``None`` means the config declares no moving nest, which is most of
    them: a plan that is not relocating anything is not priced, not
    annotated, and its composed commands are untouched.

    Otherwise a record stating the delivery, whether the preparation
    must seal a corridor, and -- when the chain cannot feed a moving
    nest at all -- the refusal to raise.
    """
    from gpuwm.static.corridor import config_declares_follow_source

    if not config_declares_follow_source(exp):
        return None
    if chain not in _FOLLOW_STATICS_DELIVERY:
        raise PlanError(
            f"run-plan cannot say how a [relocation] follow source is fed "
            f"on chain {chain!r}: it is not in the follow-statics "
            "delivery table, and guessing would either refuse a runnable "
            "config or prepare a bundle its own forecast stage rejects")
    delivery = _FOLLOW_STATICS_DELIVERY[chain]
    grid_id = int(exp.relocation.grid_id)
    return {
        "chain": chain,
        "delivery": delivery,
        "relocation_grid_id": grid_id,
        "statics_corridor": delivery == "statics_corridor",
        "refusal": (None if delivery is not None
                    else _follow_unsupported_refusal(chain, grid_id)),
    }


#: What ``automatic_resolutions`` says about each supported delivery.
#: A caller reads this to know, BEFORE launching, whether the
#: preparation it is about to pay for will seal a corridor.
_CORRIDOR_RESOLUTION_NOTE = {
    "statics_corridor":
        "the config declares a [relocation] follow source on d{grid_id:02d}, "
        "so {stage} is composed with --statics-corridor and the "
        "bundle will carry sealed child-resolution statics over each "
        "child's whole parent extent; without it "
        "gpuwm-prepared-tree-forecast refuses this config at its "
        "preflight.  Derived from the config, not from a run option: "
        "there is no way to ask for a moving nest and separately forget "
        "the statics it moves onto.  See --estimate for the size.",
    "case_data_ingest":
        "the config declares a [relocation] follow source on d{grid_id:02d}, "
        "and this route holds the geography source for the whole run, so "
        "each footprint's statics are rebuilt at move time.  No corridor "
        "is prepared and none is needed",
}


#: Why each chain with no delivery cannot feed a moving nest, in its
#: own words.  A refusal that named another chain's tools would send a
#: reader to the wrong file, so the chain-specific half of the sentence
#: lives beside the chain rather than inside a generic formatter; the
#: completeness test requires an entry for every ``None`` delivery.
#:
#: EMPTY, and that is the current truth rather than a dropped feature:
#: every chain this front door dispatches to can now feed a moving nest.
#: ``prepared:hrrr`` was the one entry, and it went when its hierarchy
#: stage learned to seal a corridor.  The machinery stays because a
#: chain added tomorrow may not be able to, and the generic sentence in
#: :func:`_follow_unsupported_refusal` would then be its stand-in until
#: someone writes it a better one.
_FOLLOW_UNSUPPORTED_DETAIL: dict[str, str] = {}


def _follow_unsupported_refusal(chain: str, grid_id: int) -> str:
    """Why this chain cannot run a moving nest, and what will."""

    detail = _FOLLOW_UNSUPPORTED_DETAIL.get(
        chain, "its preparation seals no child-resolution statics "
               "corridor and it holds no geography source at run time")
    return (
        f"this plan's config declares a [relocation] follow source on "
        f"d{grid_id:02d}, and the {chain!r} chain cannot supply the "
        f"statics a moving nest needs: {detail} -- so it is refused "
        "here instead, before the fetch.\n"
        "  remedy: run this config on a prepared chain that seals a "
        "corridor -- gfs (rw-wps) or hrrr (the hierarchy stage) -- "
        "either of which run-plan composes --statics-corridor for "
        "itself, from this same [relocation] predicate; or run it on "
        "the `experiment` route with a "
        "[case_data] block, which holds the geography source and "
        "rebuilds each footprint's statics at move time; or drop the "
        "follow source for a bounds-only [relocation], which does not "
        "move the nest and needs neither.")


# ---------------------------------------------------------------------------
# [tiles]: which chain streams, and which grid of it
# ---------------------------------------------------------------------------

#: What a configured ``[tiles]`` table reaches on each chain this front
#: door dispatches to.
#:
#: ``"root_only"``  the chain's forecast stage wires
#:                  :func:`gpuwm.core.streaming.builders_for_tree`, so
#:                  ``[tiles]`` genuinely streams -- but only grid 1.
#:                  :func:`~gpuwm.core.streaming.prepared_domain_builder`
#:                  refuses a NEST at build time ("[tiles] fired on grid
#:                  N, which is a NEST"): a nest's lateral forcing is
#:                  rebuilt from its parent every parent step
#:                  (``NestCoupler.force`` -> ``RollingNestBoundaries``)
#:                  rather than tabulated as a ``LateralBoundaries``
#:                  series, so there is nothing for
#:                  ``tile_boundary_tables`` to window.
#: ``"tree"``       the chain's forecast stage wires a builder for EVERY
#:                  grid the config asks to stream -- root through
#:                  :func:`~gpuwm.core.streaming.standalone_domain_builder`
#:                  or :func:`~gpuwm.core.streaming.prepared_domain_builder`,
#:                  nests through the latter's child road (per-buffer
#:                  packed nest tables, ``nest_stream.make_nest_tile_hook``)
#:                  -- and it honours the per-domain ``[tiles]`` table, so
#:                  which end of a coupling edge streams is the config's
#:                  choice.  Only the edge with BOTH ends streamed is
#:                  refused, and the core refuses it.
#: ``"unrouted"``   the chain reads ``exp.tiles`` at NO point, so any
#:                  enabled mode is a request nothing will ever act on.
#:
#: Every chain :func:`_chain_key` can return must appear here; a test
#: fails if one does not, so a chain cannot be added without answering
#: "and what does [tiles] do on it?".
_STREAMING_DELIVERY: dict[str, str] = {
    # WAS "unrouted", and the word was earned: gpuwm.runtime.run_experiment
    # read exp.tiles nowhere and refused it at its own front door.  It
    # wires the builders now -- builders_for_tree on the tree arm,
    # standalone_domain_builder on the single-domain arm -- so this front
    # door must stop relaying a refusal the route no longer raises.  A
    # stale "unrouted" here would refuse at resolve time, before the run
    # directory, a config the route would have run.
    "experiment": "tree",
    "prepared:go": "root_only",
    "prepared:hrrr": "root_only",
}


def streaming_decision(exp, *, chain: str) -> dict[str, Any] | None:
    """What ``[tiles]`` will do on ``chain``, or ``None``.

    ``None`` means the config configures no ``[tiles]``, which is nearly
    all of them: an unconfigured plan is not annotated, not refused, and
    its composed commands are untouched -- the same emptiness contract
    :func:`gpuwm.core.streaming.identity_payload_entry` keeps.

    Otherwise a record naming the chain, what it can stream, the grids
    it cannot, and -- when the combination cannot stream at all -- the
    refusal to raise.

    THE DEFECT THIS ANSWERS.  ``[tiles]`` used to reach the HRRR chain
    and be dropped without a word: the single-domain arm hands its
    forecast the authority the PREPARER published
    (:func:`tools.hrrr_single_domain_benchmark._experiment_tables`),
    which is built in code and has never carried a ``[tiles]`` table, so
    a user's block was read by run-plan, reported in ``--resolve``, and
    then silently replaced by a document that does not mention it.  A
    run configured to stream integrated resident, and the only evidence
    was the absence of a line in the log.  The plumbing is now real (see
    :func:`_hrrr_chain`); this is the other half -- the combinations
    that CANNOT stream, refused here from the config alone rather than
    discovered at the first tile buffer, minutes and two preparations
    downstream.
    """
    from gpuwm.core import streaming

    options = getattr(exp, "tiles", None) or streaming.OFF
    if not options.enabled:
        return None
    if chain not in _STREAMING_DELIVERY:
        raise PlanError(
            f"run-plan cannot say what [tiles] does on chain {chain!r}: "
            "it is not in the streaming delivery table, and guessing "
            "would either refuse a config that streams or accept one "
            "whose forecast stage will never read the table")
    delivery = _STREAMING_DELIVERY[chain]
    root = int(exp.root.grid_id)
    nests = tuple(int(dc.grid_id) for dc in exp.domains if dc.parent_id != 0)
    relocation = getattr(exp, "relocation", None)
    moving = (None if relocation is None or not getattr(
        relocation, "enabled", False) else int(relocation.grid_id))
    return {
        "chain": chain,
        "delivery": delivery,
        "mode": options.mode,
        "store": options.store,
        "streamable_grid_id": None if delivery == "unrouted" else root,
        # The nests this chain CANNOT stream.  On a "tree" delivery that
        # is none of them -- the child road is wired and which end streams
        # is the per-domain [tiles] table's answer, not this table's.
        "resident_grid_ids": [] if delivery == "tree" else list(nests),
        "relocation_grid_id": moving,
        # ``moving`` is reported above but not passed down: the moving
        # domain's own sentence now belongs to the core refusal, which
        # reads exp.relocation itself rather than being told about it.
        "refusal": _streaming_refusal(
            exp, chain, delivery, options, root=root, nests=nests),
    }


def _streaming_refusal(exp, chain: str, delivery: str, options, *, root: int,
                       nests: tuple[int, ...]) -> str | None:
    """Why this ``[tiles]`` cannot run on this chain, or ``None``.

    ``mode = "on"`` is the only mode refused on a chain that streams,
    and that asymmetry is
    :func:`gpuwm.core.streaming.refuse_unrouted_streaming`'s, not a new
    one: ``on`` streams unconditionally and is therefore decidable from
    the config alone -- on a tree it forces BOTH ends of every coupling
    edge streamed, the shape the coupler refuses -- while ``auto`` asks
    :mod:`tilestream.autoplan` about a specific card and legitimately
    answers "resident" or "streamed" per domain.  Asking that question
    here would stand a CUDA primary context up inside a front door that
    has not decided to use the device yet; the walk itself
    (:func:`gpuwm.core.streaming.steppers_for_tree`) refuses a
    both-streamed edge at decision time, before anything is built.

    BOTH arms now call the core rather than restating it.  The nest arm
    used to carry its own copy of the sentence, written when
    ``build_experiment`` did not ask the question -- so a config loaded
    by any door OTHER than run-plan reached the tile builder anyway, and
    two spellings of one refusal could drift.
    """
    if delivery == "unrouted":
        # The core's own sentence, raised as this front door's refusal.
        # Calling it rather than restating it is the point: a route that
        # learns to stream stops refusing here on the same day it stops
        # refusing there, without anyone remembering this file exists.
        from types import SimpleNamespace

        from gpuwm.core import streaming

        try:
            streaming.refuse_unrouted_streaming(
                SimpleNamespace(tiles=options), f"{chain!r}",
                consults_the_seam=False)
        except streaming.StreamingRefused as refusal:
            return str(refusal)
        # It did not refuse, so the core no longer agrees that this
        # chain is unrouted -- which means the route learned to read
        # [tiles] and _STREAMING_DELIVERY was not updated with it.
        # Refused rather than accepted: this function's caller has no
        # resolution note for a delivery that does not exist, and an
        # accepted plan here would be one whose forecast stage nobody
        # has checked reads the table.
        return layered(
            f"run-plan lists the {chain!r} chain as reading [tiles] at "
            "no point, but gpuwm.core.streaming.refuse_unrouted_streaming "
            f"accepted mode = '{options.mode}' for it.",
            "The two disagree, so one of them is stale.  If that route "
            "now wires a streamed-domain builder, give it a delivery in "
            "gpuwm.runplan._STREAMING_DELIVERY and a note in "
            "_STREAMING_RESOLUTION_NOTE saying which grid it can stream.")
    if options.mode != "on" or not nests:
        return None
    # The core's own sentence again, on the same terms as the unrouted arm
    # above.  ``build_experiment`` raises this for every door, so by the
    # time a run-plan holds an ExperimentConfig the refusal has already
    # fired and ``resolve_plan`` has translated it into a PlanError; what
    # remains reachable here is a caller that assembled an experiment
    # itself and asks this function directly -- and it must get the same
    # words, not a second author's paraphrase of them.
    from gpuwm.core import streaming

    try:
        streaming.refuse_streamed_nests(exp, source="this config")
    except streaming.StreamingRefused as refusal:
        return str(refusal)
    # It did not refuse, and that is now a LEGITIMATE answer rather than a
    # disagreement.  While [tiles] was tree-wide, a mode = 'on' config with
    # nests in it put both ends of every coupling edge on the streamed
    # road and the core always refused -- so reaching this line meant one
    # of the two readers was stale, and refusing was right.  The table is
    # per-domain now: `mode = "on"` over the tree with `tiles = { mode =
    # "off" }` on the nest is "stream the parent, keep the child
    # resident", which is a shape the core deliberately admits.  The core
    # asks the question about the EDGE and it is the only reader that can;
    # counting nests here cannot tell the two configs apart, so this arm
    # defers instead of second-guessing it.
    return None


#: What ``automatic_resolutions`` says about a configured ``[tiles]``.
#: A caller reads this to know, BEFORE launching, which grid will
#: actually stream -- because the run itself cannot tell them: a grid
#: that declined to stream is ABSENT from the stepper dict, and absent
#: is exactly what an unconfigured grid looks like.
_STREAMING_RESOLUTION_NOTE = {
    "root_only":
        "[tiles] mode = '{mode}' (store = '{store}') is carried to the "
        "forecast stage of the {chain} chain, which wires "
        "streaming.builders_for_tree and streams for real.  Only "
        "d{root:02d} can stream: a nest's forcing is rebuilt from its "
        "parent rather than tabulated, so prepared_domain_builder "
        "refuses one.{nests}  Nothing here binds the restart identity -- "
        "streaming.identity_payload_entry contributes nothing on purpose, "
        "so a checkpoint written streamed resumes resident and one "
        "written resident resumes streamed",
    "tree":
        "[tiles] mode = '{mode}' (store = '{store}') is carried to the "
        "forecast stage of the {chain} chain, which wires "
        "streaming.builders_for_tree over the whole domain tree and "
        "streams for real.  ANY grid can stream here, d{root:02d} "
        "included: the root through its own tabulated boundaries, a nest "
        "through the child road (per-buffer packed nest tables refilled "
        "when the rolling generation moves).  Which end of a coupling "
        "edge streams is this config's choice -- put `tiles = {{ mode = "
        "\"off\" }}` on the [[domain]] you want resident -- and mode = "
        "'auto' answers it by pricing every domain against one budget, "
        "reserving what the domains below it need before a streamed "
        "domain picks its tile.  The only refused shape is an edge with "
        "BOTH ends streamed.{nests}  Nothing here binds the restart "
        "identity -- streaming.identity_payload_entry contributes nothing "
        "on purpose, so a checkpoint written streamed resumes resident "
        "and one written resident resumes streamed",
}


def corridor_estimate(exp, decision: Mapping[str, Any] | None
                      ) -> dict[str, Any]:
    """What the sealed corridor will cost, priced before it is built.

    The corridor is the one preparation artifact whose size a caller
    cannot infer from the domain sizes it already has -- it is
    parent-extent at CHILD resolution, so a modest nest on a large
    parent is hundreds of megabytes.  A front end that launches a
    moving-nest plan without showing that number is hiding the largest
    single thing the preparation will write.

    Priced through the preparation's OWN child selection
    (:func:`gpuwm.static.corridor.validated_corridor_selection`) and the
    corridor module's own arithmetic
    (:func:`gpuwm.static.corridor.corridor_cost`), so the figure shown
    before the run and the artifact written during it come from one
    source rather than from an estimate that agrees with it today.

    Chain-agnostic by construction: nothing here reads the chain, only
    the experiment's own domain tree, so the same arithmetic prices a
    GFS corridor and an HRRR one.  The ``decision`` is consulted for
    WHETHER a corridor is sealed, never for how big it is.
    """
    if decision is None or not decision["statics_corridor"]:
        return {
            "domains": [], "host_bytes": 0, "host_gib": 0.0,
            "basis": ("this config declares no [relocation] follow "
                      "source, so no statics corridor is prepared"
                      if decision is None else
                      "a moving nest on this chain is fed by "
                      f"{decision['delivery']}, which seals no corridor"),
        }
    from gpuwm.static.corridor import (corridor_cost,
                                       validated_corridor_selection)

    # `--statics-corridor` is passed bare, which the preparation reads
    # as "every child domain" -- so every child is priced, not only the
    # one [relocation] names.
    by_id = {int(domain.grid_id): domain for domain in exp.domains}
    domains = []
    for grid_id in validated_corridor_selection(exp, "all"):
        child = by_id[grid_id]
        cost = corridor_cost(child, by_id[int(child.parent_id)].run)
        cost["domain"] = f"d{grid_id:02d}"
        domains.append(cost)
    total = sum(entry["host_bytes"] for entry in domains)
    return {
        "domains": domains,
        "host_bytes": int(total),
        "host_gib": round(total / 1024 ** 3, 4),
        "basis": (
            "each child's corridor is its parent's full extent at the "
            "child's resolution (parent_nx*ratio x parent_ny*ratio "
            "cells) carrying the native static contract's "
            f"{domains[0]['planes_per_cell']} float64 planes, so "
            f"{domains[0]['bytes_per_cell']} bytes per corridor cell; "
            "counted from the same field inventory the build is "
            "shape-checked against.  DISK and HOST are the same figure "
            "to within container headers: the cache is an uncompressed "
            "NPZ of exactly these arrays.  No GPU residency -- a "
            "corridor is cropped on the host, so the VRAM estimate "
            "above is unchanged by it."),
    }


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


def _last_sequence(path: Path) -> int:
    """The highest sequence already in an event file, or 0.

    Tolerant by design: this runs before anything is written, against a
    file that may not exist, may be empty, or may end in a torn line
    from a killed run.  None of those is a reason to refuse to START a
    stream -- they are things :func:`read_events` reports to whoever
    reads it.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return 0
    highest = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        sequence = record.get("sequence") if isinstance(record, dict) else None
        if isinstance(sequence, int) and sequence > highest:
            highest = sequence
    return highest


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
        # Continue an existing stream rather than restarting its
        # numbering.  The file is opened for APPEND, so a second run
        # into the same directory -- a resume, or a caller that reused a
        # run_dir -- would otherwise write a record numbered 1 after a
        # record numbered 7, and read_events would refuse the whole file
        # as reordered.  Counting what is already there keeps the
        # sequence dense across the join, which is the one property
        # every reader of this stream depends on.
        self._sequence = _last_sequence(self.path)
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
                 root_domain: int = 1, accepted_wall: float | None = None):
        self._events = events
        self._heartbeat = heartbeat
        self._root_domain = int(root_domain)
        #: The monotonic reading taken when ``plan_accepted`` was
        #: emitted.  Passed in rather than read here because this object
        #: is built several steps into ``execute_plan``, and every wall
        #: time this run reports is measured from the instant the plan
        #: was accepted -- not from the instant an observer happened to
        #: exist.  Defaulted for callers that build one directly.
        self._accepted_wall = (time.perf_counter() if accepted_wall is None
                               else float(accepted_wall))
        #: Set by :meth:`arm_first_products` when a prepared chain wants
        #: its first frame rendered as it lands.  ``None`` -- the default
        #: and the whole of the ``experiment`` route -- means the
        #: finalize stage is the only render there has ever been.
        self._first_products = None
        #: Time to first plot, once there is one.  Kept so the run's
        #: ``completed`` event can carry the headline number too: a
        #: reader comparing runs should not have to scan the stream for
        #: the one line that has it.
        self._first_products_seconds: float | None = None
        self._stage: str | None = None
        self._stage_phases: list[str] = []
        self._stage_started_wall = 0.0
        self._forecast_started_wall: float | None = None
        self._forecast_started_model: float | None = None
        self._committed = 0
        self._progress_events = 0
        #: The last model time this observer saw.  The chain summary's
        #: only route-independent source for how far the run got: the
        #: single-domain runner publishes it in progress.json and the
        #: tree runner does not publish it at all.
        self._last_model_seconds: float | None = None

    # -- stage bookkeeping --------------------------------------------

    @property
    def stage(self) -> str | None:
        """The stage currently open, or ``None`` before/after the run."""

        return self._stage

    @property
    def outputs_committed(self) -> int:
        return self._committed

    @property
    def last_model_seconds(self) -> float | None:
        return self._last_model_seconds

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
        self._last_model_seconds = model_seconds
        if self._stage != "forecast":
            self.enter_stage("forecast", phase=phase)
        if self._forecast_started_wall is None:
            # Armed on the FIRST progress call, not on entering the
            # stage.  Every prepared chain opens `forecast` itself
            # before handing the runner over, so keying this off the
            # stage transition meant it never armed there: a live
            # nested run published 181 progress events with speed_x
            # null and wall_seconds 0.0 on every one of them.
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
        # `domain` above is the ROOT clock, and on a tree that is only
        # part of the answer: the nests advance on their own clocks.
        # Present only when there is more than one, so the single-domain
        # route's events are unchanged and a consumer can treat the key's
        # absence as "the root IS the tree".
        clocks = extra.get("domain_clocks")
        if isinstance(clocks, dict) and len(clocks) > 1:
            payload["domains"] = [
                {"domain": int(grid_id), "model_seconds": float(seconds)}
                for grid_id, seconds in sorted(clocks.items())]
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

    # -- time to first plot -------------------------------------------

    @property
    def first_products(self):
        """The armed early render, or ``None``.

        The finalize stage reads this off whatever observer it was given
        -- directly or through :class:`_GoObserver` -- to collect the
        render before deciding what is left to draw.
        """

        return self._first_products

    @property
    def first_products_seconds(self) -> float | None:
        """Time to first plot, or ``None`` if no early render published."""

        return self._first_products_seconds

    def arm_first_products(self, render_plan) -> None:
        """Render the first committed frame as it lands, not at finalize.

        ``render_plan`` is the dict the finalize stage will hand
        ``go_cli._render_stage``: the same output directory and the same
        product spec, so the early render and the late one cannot drift
        apart in what they draw or where they put it.

        Silently does nothing when this run named no products, which is
        the default.  A caller therefore does not have to ask whether
        the feature applies before arming -- the answer lives in one
        place, :func:`gpuwm.first_products.early_render_requested`.
        """

        from gpuwm.first_products import (FirstProducts,
                                          early_render_requested)

        if not early_render_requested(render_plan.get("render_products")):
            return
        self._first_products = FirstProducts(
            render_plan, report=self._first_products_ready, warn=self.warn)

    def _first_products_ready(self, receipt) -> None:
        """The early render published.  This is the TTFP number.

        Emitted from the render's own worker thread, which is why
        ``EventStream`` holds a lock: this and ``model_progress`` from
        the forecast genuinely do reach it at once.
        """

        elapsed = round(time.perf_counter() - self._accepted_wall, 6)
        self._first_products_seconds = elapsed
        self._events.emit(
            "first_products_ready",
            domain=receipt["domain"], valid_time=receipt["valid_time"],
            frame=receipt["frame"], paths=list(receipt["paths"]),
            render_products=receipt["render_products"],
            render_seconds=receipt["render_seconds"],
            seconds_from_plan_accepted=elapsed)

    def output_committed(self, *, domain: int, valid_time, path) -> None:
        """One wrfout is durable on disk.  Raised from the writer.

        On the domain-tree route this arrives on the per-domain writer
        thread, after ``WrfoutWriter.close`` has fsynced, self-validated
        and renamed the temporary onto its final name -- so the event is
        emitted for a file that exists and passes its own inventory
        check, never for one that is merely queued.

        When an early render is armed, the FIRST root-domain frame also
        dispatches it.  That frame is the analysis: the history alarm is
        true at t = 0, so it is written before a single step is
        integrated, and it is the picture a reader has been waiting the
        whole download and preparation for.
        """

        self._committed += 1
        self._events.emit(
            "output_committed", domain=int(domain),
            valid_time=(valid_time.isoformat()
                        if isinstance(valid_time, datetime)
                        else str(valid_time)),
            path=str(path))
        trigger = self._first_products
        if trigger is None or int(domain) != self._root_domain:
            return
        # Guarded here as well as inside the trigger.  This method is
        # reached from `runtime._output_committed`, which -- unlike the
        # async writer's own call site -- does not wrap the callback, so
        # anything raised here would land in the model loop.
        try:
            trigger.frame_committed(
                domain=domain, valid_time=valid_time, path=path)
        except Exception as error:  # noqa: BLE001 - telemetry never fails
            self.warn(
                "first_products_not_dispatched",
                "the early render of the first frame could not be "
                f"started ({type(error).__name__}: {error}); the finalize "
                "stage is unaffected")

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
    """Every ``ExperimentConfig`` field the document did not spell.

    Mechanical, from the dataclass itself: a field with a declared
    default that the TOML did not set was chosen by the schema, not by
    the author, and that is exactly what ``automatic_resolutions``
    exists to say out loud.

    "Did not set" spans BOTH places an author can write one of these
    fields.  Several are authored as their own top-level table --
    ``[relocation]``, ``[projection]``, ``[perturbation]`` -- and reading
    only inside ``[experiment]`` reported them as schema defaults with
    the schema's value attached, which for a moving-nest plan says
    ``relocation.enabled = false`` about a config whose nest follows a
    storm.  The run was right and the ``configuration`` block in the same
    event was right; only this list lied, and it lied to exactly the
    programmatic caller it exists for.
    """

    from gpuwm.experiment import ExperimentConfig

    table = raw.get("experiment")
    spelled = set(table) if isinstance(table, dict) else set()
    # A top-level table whose name IS a field name is the author spelling
    # that field.  Taken from the document rather than from a second list
    # of "the tabular ones", so a table added later needs no edit here.
    spelled |= {name for name in raw if name != "experiment"}
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

        # Imported here rather than at module scope, on this file's own
        # convention for gpuwm.core.streaming: the run-plan front door is
        # reached by every route, and the streaming module must stay a
        # thing a resident plan never pays for.
        from gpuwm.core.streaming import StreamingRefused

        warnings: list[dict[str, str]] = list(warnings_generated)
        try:
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
        except StreamingRefused as refusal:
            # build_experiment refuses [tiles] mode = 'on' over a nested
            # tree, and it does so as StreamingRefused -- a RuntimeError,
            # which this front door would print as a traceback.  Every
            # other refusal here travels as PlanError (a ValueError) so
            # that gpuwm.cli.main prints one sentence and exits 2, and a
            # config-shaped refusal must not be the exception.  The
            # message is carried verbatim: it is the core's sentence, and
            # restating it here is exactly the drift _streaming_refusal
            # below already refuses to introduce.
            raise PlanError(str(refusal)) from None
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

    chain = _chain_key(plan.route, (raw.get("fetch") or {}).get("source"))

    # A moving nest, decided and REPORTED before anything is fetched.
    # The chain is read off the config's own [fetch] table, which is
    # what `_execute_prepared_route` dispatches on, so the chain judged
    # here is the chain that runs.  A plan with no follow source adds
    # nothing to this document.
    decision = follow_statics_decision(exp, chain=chain)
    if decision is not None:
        if decision["refusal"] is not None:
            raise PlanError(decision["refusal"])
        # Published without the refusal slot: reaching here means there
        # was none, and a null field inviting a caller to test it would
        # imply this document ever carries a live one.
        decision = {key: value for key, value in decision.items()
                    if key != "refusal"}
        resolutions.append({
            "scope": "preparation", "key": "statics_corridor",
            "value": decision["statics_corridor"],
            "basis": "relocation_follow",
            "note": _CORRIDOR_RESOLUTION_NOTE[decision["delivery"]].format(
                grid_id=decision["relocation_grid_id"],
                stage=_CORRIDOR_STAGE.get(decision["chain"],
                                          "the preparation"))})

    # [tiles], on the same terms and in the same place.  A configured
    # streaming mode that the chain cannot honour is refused HERE --
    # from the config, before the fetch -- rather than at the first tile
    # buffer, which on the HRRR chain is after a download, two
    # preparations and a whole resident tree construction.  A plan that
    # configures no [tiles] adds nothing to this document.
    tiles = streaming_decision(exp, chain=chain)
    if tiles is not None:
        if tiles["refusal"] is not None:
            raise PlanError(tiles["refusal"])
        tiles = {key: value for key, value in tiles.items()
                 if key != "refusal"}
        resolutions.append({
            # NOT "tiles".  `_schema_default_resolutions` already emits
            # that key -- scope "experiment", value the whole default
            # StreamingOptions -- for a config that spells no [tiles].
            # The two are mutually exclusive today, so a consumer keyed
            # on the name would never see both; it would just see one
            # key whose value is sometimes a table of defaults and
            # sometimes a mode, which is a shape it cannot parse.
            "scope": "execution", "key": "tiles_delivery",
            "value": tiles["mode"],
            "basis": "experiment_config",
            "note": _STREAMING_RESOLUTION_NOTE[tiles["delivery"]].format(
                mode=tiles["mode"], store=tiles["store"], chain=chain,
                root=tiles["streamable_grid_id"],
                nests=("" if not tiles["resident_grid_ids"] else
                       "  d"
                       + ", d".join(f"{grid:02d}" for grid
                                    in tiles["resident_grid_ids"])
                       + " therefore run resident, and mode = 'auto' "
                         "reaches that same refusal for any of them "
                         "autoplan says does not fit."))})

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
        # The moving-nest decision as a record, beside the sentence
        # `automatic_resolutions` carries: a front end that wants to
        # draw a corridor toggle reads this, and one that just prints
        # the resolutions gets the same fact in prose.  ``null`` when
        # the config moves no nest.
        "moving_nest": decision,
        # The [tiles] decision as a record, for the same reason and on
        # the same terms: ``null`` when the config configures none, and
        # otherwise the grid that CAN stream beside the grids that
        # cannot.  A run cannot answer this for itself -- a grid that
        # declined to stream is absent from the stepper dict, and absent
        # is what an unconfigured grid looks like too -- so a front end
        # that wants to show which domain will stream has to be told
        # before the run rather than after it.
        "tiles": tiles,
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

    # -- time to first plot, forwarded verbatim -----------------------

    @property
    def first_products(self):
        return self._observer.first_products

    def arm_first_products(self, render_plan) -> None:
        self._observer.arm_first_products(render_plan)

    def starting(self) -> None:
        self._observer.starting()

    def complete(self, model_elapsed_seconds: float) -> None:
        self._observer.complete(model_elapsed_seconds)

    def failed(self) -> None:
        self._observer.failed()


def _fetch_arguments_from_hints(hints: Mapping[str, Any],
                                *, out: Path) -> list[str]:
    """The ``gpuwm fetch`` argv one ``[fetch]`` hints table spells.

    The wizard's own words for that table are "keys mirror `gpuwm fetch`
    flags", so the mapping is mechanical: ``forecast_start_hour``
    becomes ``--forecast-start-hour``.  Nothing is filtered by a list
    kept here -- the argv is handed to gpuwm's real fetch parser, which
    accepts exactly what `gpuwm fetch` accepts and refuses the rest.
    """

    arguments: list[str] = []
    for key in sorted(hints):
        if key == "out":
            continue          # run-plan owns where the data lands
        value = hints[key]
        if isinstance(value, bool):
            if value:
                arguments.append("--" + key.replace("_", "-"))
            continue
        arguments += ["--" + key.replace("_", "-"), str(value)]
    return arguments + ["--out", str(out)]


def _hrrr_chain(plan: RunPlan, *, config_path: Path, exp,
                observer: RunObserver, run_dir: Path) -> Mapping[str, Any]:
    """The documented HRRR chain: fetch, prepare, forecast.

    HRRR is not ``gpuwm go``'s.  ``go`` refuses the source by
    construction (``ORCHESTRATED_SOURCES = ("gfs",)``), preparation is
    ``tools/prepare_hrrr_wrf`` rather than rw-wps, and the HRRR tools
    read the four namelist/JSON files the wizard writes beside the
    config rather than the TOML itself.  So the stages and their order
    here are the wizard's own printed chain
    (:func:`gpuwm.domain_wizard.hrrr_route_commands`), driven rather
    than printed, with each stage's refusals left entirely alone.

    ONE thing is added to that chain: ``--wps-namelist``.  The runner's
    HRRR manifest role inventory requires a ``wps_namelist`` role
    (prepared_single_domain_forecast.py:2181 -- the prepared-cache
    identity's ``namelist_sha256`` IS that file's digest on this
    route), and the preparer only records the role if it is handed the
    file.  The printed chain never passed it, so the bundle it produced
    could not be read by the single-domain runner at all, and HRRR was
    sent to a benchmark script instead.  Passing it makes a
    single-domain HRRR bundle structurally identical to a GFS one at
    the run step -- same runner, same digests, same observer.
    """

    from gpuwm.fetch import sha256_file
    from gpuwm.go_cli import proof_digests, run_stage
    from gpuwm.hrrr_route_inputs import route_input_paths

    inputs = route_input_paths(config_path)
    absent = sorted(role for role, path in inputs.items()
                    if not path.is_file())
    if absent:
        raise PlanError(
            f"the HRRR route reads {absent} beside {config_path.name}, "
            "and `gpuwm domain` writes them at emission; this config was "
            "not emitted for the HRRR route")

    raw = tomllib.load(io.BytesIO(config_path.read_bytes()))
    hints = dict(raw.get("fetch") or {})
    intent = plan.config_intent or {}
    data_dir = Path(plan.run_options.get("data_dir")
                    or intent.get("data_dir") or (run_dir / "data"))
    geog_root = plan.run_options.get("geog_root") or intent.get("geog_root")
    if geog_root is None:
        from gpuwm.geog_assets import default_geog_root

        geog_root = default_geog_root()
    # Not created here, and deliberately so: the preparer refuses an
    # --output-root that already exists ("refusing existing output
    # root"), which is its own create-only guarantee.
    prep_root = run_dir / "chain" / "hrrr-root-prep"
    forecast_dir = run_dir / "chain" / "run"

    # -- fetch ---------------------------------------------------------
    observer.enter_stage("fetch", phase="fetch")
    fetch_report = _run_fetch(
        _fetch_arguments_from_hints(hints, out=data_dir), run_dir)
    manifest = data_dir / "SHA256SUMS"
    if not manifest.is_file():
        raise PlanError(
            f"the HRRR fetch wrote no {manifest}, so the preparation "
            "stage has nothing to bind against")
    observer.finish_stage(fetch=fetch_report)

    # -- prepare -------------------------------------------------------
    observer.enter_stage("prepare", phase="prepare")
    # The two stages spell the same instant differently: [fetch] carries
    # YYYY-MM-DDTHH (what `gpuwm fetch` takes) and the preparer takes
    # YYYY-MM-DD_HH:MM:SS (what the wizard's printed chain passes it).
    # Converted through the real parser rather than by slicing the
    # string, so a cadence rule the parser enforces is enforced here.
    from gpuwm.fetch import parse_cycle

    cycle = parse_cycle(
        str(hints["cycle"]), "hrrr").strftime("%Y-%m-%d_%H:%M:%S")
    cadence = int(exp.domains[0].history_interval_s)
    prepare = [
        sys.executable, "-m", "tools.prepare_hrrr_wrf",
        "--source-root", str(data_dir),
        "--source-manifest", str(manifest),
        "--source-manifest-sha256", sha256_file(manifest),
        "--domain-spec", str(inputs["target_domain"]),
        "--namelist-input", str(inputs["namelist_input"]),
        # THE addition.  Without it the emitted bundle has no
        # wps_namelist role and the runner refuses it outright.
        "--wps-namelist", str(inputs["wps_namelist"]),
        "--geog-root", str(geog_root),
        "--cycle", cycle,
        "--run-seconds", str(int(exp.run_seconds)),
        "--history-interval-seconds", str(cadence),
        "--skip-stock-wrf-export",
        "--output-root", str(prep_root),
    ]
    profile = (plan.run_options.get("physics_profile")
               or intent.get("physics_profile"))
    if profile:
        # Passed only when the plan states it.  The route owns its own
        # physics gate and the emitted TOML records physics as numbers
        # rather than a profile id, so there is nothing to recover from
        # a config.path plan -- and a default invented at this layer
        # would silently outrank the preparer's own.  Where the two
        # disagree the runner's identity check refuses, loudly, which is
        # the gate doing its job.
        prepare += ["--physics-profile", str(profile)]
    lead = hints.get("forecast_start_hour")
    if lead:
        prepare += ["--forecast-start-hour", str(int(lead))]
    run_stage("prepare", prepare, explain=False,
              progress=prep_root / "progress.json",
              observer=_GoObserver(observer))

    # Armed before either forecast arm, with the very dict `_hrrr_render`
    # will hand the finalize stage below, so the frame rendered as it
    # lands and the frames rendered at the end agree on both the output
    # directory and the product spec by construction.
    observer.arm_first_products(
        _hrrr_render_plan(plan, forecast_dir=forecast_dir, run_dir=run_dir))

    if len(exp.domains) > 1:
        # A nested HRRR run takes a THIRD stage between the root
        # preparation and the forecast, and then a different runner.
        # Everything above -- fetch, and the root preparation this
        # branch shares -- is identical, which is why the split is here
        # and not at the top of the function.
        from gpuwm.static.corridor import config_declares_follow_source

        tree_root = _hrrr_hierarchy_stage(
            prep_root=prep_root, inputs=inputs, hints=hints,
            geog_root=geog_root, manifest=manifest, cycle=cycle,
            run_dir=run_dir, observer=observer,
            # THE predicate, not a copy of it: the same function
            # `follow_statics_decision` consulted at resolve time, so a
            # plan reported as corridor-bearing seals one and a plan
            # reported as still does not.
            statics_corridor=config_declares_follow_source(exp))
        _hrrr_tree_forecast(
            tree_root=tree_root, config_path=config_path,
            forecast_dir=forecast_dir, observer=observer)
        return _hrrr_render(plan, forecast_dir=forecast_dir,
                            run_dir=run_dir, observer=observer)

    # The preparer publishes its own handoff -- prepared_root, the three
    # digests, and the PUBLISHED authority paths -- into
    # public-wrapper-result.json.  Read it rather than re-deriving any
    # of it: that is the same relay-from-the-artifact rule `go` follows,
    # and every part of it is a value this layer must not compute.
    #
    # Three things here are not guessable and were each wrong first
    # time:
    #   * proof.json lives at the OUTPUT ROOT, not inside the prepared
    #     cache -- the bundle root IS --output-root;
    #   * --prepared-root is therefore that root too, because the
    #     runner's HRRR_BUNDLE_PATHS are relative to it;
    #   * --experiment-config / --wps-namelist must be the PUBLISHED
    #     copies (experiment.toml, namelist.wps) and not the wizard's
    #     originals, because the runner checks each supplied file's
    #     NAME and digest against the portable source manifest.
    wrapper = _read_json_object(prep_root / "public-wrapper-result.json")
    handoff = wrapper.get("portable_bundle")
    if not isinstance(handoff, dict):
        raise PlanError(layered(
            f"the HRRR preparation published no portable bundle in "
            f"{prep_root / 'public-wrapper-result.json'}.",
            "That bundle -- proof.json, the role-keyed source manifest "
            "and the experiment authority -- is what the forecast stage "
            "binds, and the preparer only publishes it when handed "
            "--wps-namelist, which this chain does pass.  Its own "
            "output is above."))
    prepared_root = Path(handoff["prepared_root"])
    # Cross-check the relayed digests against the proof on disk, using
    # go's own reader.  Cheap, and it catches a handoff that does not
    # describe the artifacts beside it.
    digests = proof_digests(prepared_root)
    for key, relayed in (("proof", "proof_sha256"),
                         ("source_manifest", "source_manifest_sha256"),
                         ("prepared_content", "prepared_content_sha256")):
        if handoff.get(relayed) != digests[key]:
            raise PlanError(
                f"the HRRR preparation's published {relayed} does not "
                f"match {prepared_root / 'proof.json'}; the bundle and "
                "the proof beside it disagree")
    observer.finish_stage(prepared_root=str(prepared_root),
                          bundle=str(prep_root /
                                     "public-wrapper-result.json"))

    # -- forecast ------------------------------------------------------
    # In process, so the runner's per-step progress and its per-wrfout
    # landing hook reach the observer.  Same runner and same flags the
    # GFS chain uses; only --source and the bundle differ.
    argv = [
        "--source", "hrrr",
        "--prepared-root", str(prepared_root),
        "--proof-sha256", digests["proof"],
        "--source-manifest-sha256", digests["source_manifest"],
        "--prepared-content-sha256", digests["prepared_content"],
        "--experiment-config", str(handoff["experiment_config"]),
        "--wps-namelist", str(handoff["wps_namelist"]),
        "--io-mode", "history", "--outdir", str(forecast_dir),
    ]
    # [tiles], and the ONE flag on this argv that does not come out of
    # the bundle.
    #
    # THE DEFECT.  --experiment-config above is the authority the
    # PREPARER published, not the config the user wrote: it is rendered
    # from tables that stage builds in code
    # (tools/hrrr_single_domain_benchmark.py _experiment_tables), and
    # those tables are {experiment, projection, shared, domain} -- there
    # has never been a [tiles] among them.  So a user's [tiles] block
    # reached run-plan, was validated, was reported by --resolve, and
    # then vanished at exactly this line: the forecast loaded a document
    # that does not mention it and ran resident, with nothing in the log
    # to say the mode had been asked for.
    #
    # Forwarded as a FLAG rather than published into that document on
    # purpose.  The published authority is hash-bound -- the runner
    # compares its name and sha256 against the portable source manifest
    # -- so putting [tiles] in it would make the execution mode part of
    # the prepared bundle's identity, and a bundle prepared streamed
    # could then not be re-run resident.  That is the exact coupling
    # streaming.identity_payload_entry exists to prevent: it contributes
    # NOTHING to the restart identity so that a forecast which outgrew
    # its card can resume on the machine it outgrew.
    #
    # Only when the mode is enabled, so an ordinary run composes the
    # argv it has always composed, token for token.
    if exp.tiles.enabled:
        argv += ["--tiles", json.dumps(dataclasses.asdict(exp.tiles),
                                       sort_keys=True)]
    observer.enter_stage("forecast", phase="forecast")
    from gpuwm import prepared_single_domain_forecast as runner

    code = runner.main(argv, observer=observer)
    if code:
        raise RuntimeError(
            f"the HRRR forecast stage exited {code}; its own refusal is "
            "above")

    # -- render --------------------------------------------------------
    # Shared with the tree arm above: `render_products` -- including
    # `none` -- must mean exactly the same thing however the forecast
    # was produced, and the wizard's printed HRRR chain has no render
    # step at all, so this is the one place the sources are made to
    # agree.
    return _hrrr_render(plan, forecast_dir=forecast_dir, run_dir=run_dir,
                        observer=observer)


def _hrrr_hierarchy_stage(*, prep_root: Path, inputs: Mapping[str, Path],
                          hints: Mapping[str, Any], geog_root,
                          manifest: Path, cycle: str, run_dir: Path,
                          observer: RunObserver,
                          statics_corridor: bool = False) -> Path:
    """Build d02..dNN from the sealed root preparation.

    The stage the GFS tree does not have.  rw-wps is not on this path at
    all: the root preparation above is ``tools.prepare_hrrr_wrf``, and
    this turns its sealed d01 into a hierarchy the tree runner can
    execute.

    Nine required flags, and they are not the preparer's -- notably
    ``--stock-wrf-namelist-input``, which rw-wps rejects outright.  That
    is why this composes its own argv rather than sharing a builder with
    either neighbour: two tools that take *almost* the same flags are
    exactly where a shared builder starts passing one of them something
    it refuses.

    ``statics_corridor`` adds the tenth, and it is where the HRRR chain
    answers a moving nest.  On the GFS chain the corridor flag rides the
    rw-wps prepare stage; here the root preparer knows nothing of the
    children, so the flag belongs to THIS stage -- the one holding
    ``--geog-root`` and the child geometries.  The caller derives the
    boolean from the corridor module's own follow predicate, so the
    plan that was resolved as corridor-bearing is the plan that seals
    one.
    """

    from gpuwm.fetch import sha256_file
    from gpuwm.go_cli import run_stage

    tree_root = run_dir / "chain" / "hrrr-hierarchy"
    command = [
        sys.executable, "-m", "gpuwm.hrrr_hierarchy_direct",
        "--root-preparation", str(prep_root),
        "--root-domain-spec", str(inputs["target_domain"]),
        "--wps-namelist", str(inputs["wps_namelist"]),
        "--namelist-input", str(inputs["namelist_input"]),
        # rw-wps has no such flag; this stage requires it.
        "--stock-wrf-namelist-input", str(inputs["stock_namelist_input"]),
        "--geog-root", str(geog_root),
        "--source-manifest", str(manifest),
        "--source-manifest-sha256", sha256_file(manifest),
        "--cycle", cycle,
        "--output-root", str(tree_root),
    ]
    if statics_corridor:
        # Bare, exactly as the GFS chain passes it: the preparation
        # reads that as "every child domain", which is also what
        # corridor_estimate priced for this plan.
        command.append("--statics-corridor")
    # Only when nonzero.  This stage raises on a negative lead, and
    # raises again if a lead is passed beside the deprecated
    # --valid-time; passing a bare 0 is legal but says nothing, and the
    # chain reads better without it.
    lead = hints.get("forecast_start_hour")
    if lead:
        command += ["--forecast-start-hour", str(int(lead))]

    observer.enter_stage("prepare", phase="hierarchy")
    run_stage("hierarchy", command, explain=False,
              progress=tree_root / "progress.json",
              observer=_GoObserver(observer))
    if not tree_root.is_dir():
        raise PlanError(
            f"the HRRR hierarchy stage wrote no {tree_root}; its own "
            "output is above")
    observer.finish_stage(hierarchy_root=str(tree_root))
    return tree_root


def _hrrr_tree_forecast(*, tree_root: Path, config_path: Path,
                        forecast_dir: Path,
                        observer: RunObserver) -> None:
    """The same tree runner the GFS tree route drives.

    The relay is the same shape too -- one preparation-receipt digest
    plus the experiment config's own -- and it reaches the same
    schema-matched document resolver.  Only the filename underneath
    differs: this hierarchy writes ``receipt.json`` where rw-wps writes
    ``proof.json``, and because the resolver matches on SCHEMA rather
    than on filename order it needed no change to find it.

    The tool also prints ``preparation_receipt_sha256`` on stdout.  It
    is not read: the tool computes that value as the sha256 of
    ``receipt.json``'s bytes, so hashing the artifact gives the
    identical digest without making a printed line load-bearing.

    ``[tiles]`` NEEDS NO FLAG HERE, and that is a fact about this argv
    rather than an omission.  ``--experiment-config`` below is the
    USER'S config -- the file they wrote or the wizard emitted, bound by
    its own digest on the next line -- so ``exp.tiles`` at the tree
    runner is already the table they typed, verbatim.  The single-domain
    arm needs ``--tiles`` only because IT hands over the authority the
    preparer published instead, and that document has no [tiles] table
    to carry.  Adding a flag here as well would give one table two
    sources on one route, which is how the two arms end up streaming
    differently from the same config.  Pinned by a test rather than left
    to this comment.
    """

    from gpuwm.fetch import sha256_file
    from gpuwm.go_cli import _hierarchy_document

    receipt = _hierarchy_document(tree_root)
    argv = [
        "--prepared-root", str(tree_root),
        "--preparation-receipt-sha256", sha256_file(receipt),
        "--experiment-config", str(config_path),
        "--experiment-config-sha256", sha256_file(config_path),
        "--io-mode", "history", "--outdir", str(forecast_dir),
    ]
    observer.enter_stage("forecast", phase="forecast")
    from gpuwm import prepared_domain_tree_forecast as runner

    code = runner.main(argv, observer=observer)
    if code:
        raise RuntimeError(
            f"the HRRR tree forecast stage exited {code}; its own "
            "refusal is above")


def _hrrr_render_plan(plan: RunPlan, *, forecast_dir: Path,
                      run_dir: Path) -> dict[str, Any]:
    """The plan dict this chain's render stage runs on.

    One function because it now has two readers: the finalize render
    below, and the early render armed before the forecast.  Two copies
    of this literal is exactly how a run ends up publishing its first
    frame into one directory and the rest into another.
    """

    return {"run": forecast_dir, "render": run_dir / "chain" / "png",
            "render_products": plan.run_options.get("render_products")}


def _hrrr_render(plan: RunPlan, *, forecast_dir: Path, run_dir: Path,
                 observer: RunObserver) -> Mapping[str, Any]:
    """go's render stage against this chain's output, then the summary."""

    from gpuwm.go_cli import _render_stage

    observer.enter_stage("finalize", phase="render")
    _render_stage(
        _hrrr_render_plan(plan, forecast_dir=forecast_dir, run_dir=run_dir),
        explain=False, observer=_GoObserver(observer))
    return _chain_summary(run_dir / "chain", observer=observer)


def _chain_summary(chain: Path, *,
                   observer: "RunObserver | None" = None) -> dict[str, Any]:
    """A finished chain's completion signals, from its own artifacts.

    The two runners do not leave the same receipts.  The single-domain
    one writes ``progress.json`` and ``report.json``; the tree one
    writes a ``certification-capsule.json`` and neither of the others.
    Reading only the first pair made a completed nested run report
    ``completed_seconds: 0.0`` and ``status: null`` -- and, because the
    heartbeat is fed from this summary, published a ``complete``
    heartbeat whose model time was zero beside an outer_step of 180.

    ``completed_seconds`` therefore comes from the observer, which saw
    every step on either route.  Nothing is inferred: where a receipt
    does not state a status, this says so rather than assuming a PASS
    from the absence of a failure.
    """

    forecast = chain / "run"
    progress = _read_json_object(forecast / "progress.json")
    report = _read_json_object(forecast / "report.json")
    capsule_path = forecast / "certification-capsule.json"
    capsule = _read_json_object(capsule_path)
    wrfouts = sorted(forecast.glob("**/wrfout_*"))

    frames = capsule.get("output", {}).get("frames")
    completed_seconds = progress.get("model_elapsed_seconds")
    if not isinstance(completed_seconds, (int, float)) and observer is not None:
        completed_seconds = observer.last_model_seconds
    status = report.get("status") or progress.get("status")
    return {
        "wrfout_count": (len(wrfouts) if wrfouts
                         else len(frames) if isinstance(frames, list)
                         else int(progress.get("frame_count") or 0)),
        "completed_seconds": _finite_seconds(completed_seconds),
        "nan_free": None if status is None else status == "PASS",
        "status": status,
        "status_basis": (
            "the runner's own report" if status is not None
            else "this runner publishes a certification capsule rather "
                 "than a status report; the capsule's presence is not "
                 "read as a verdict here"),
        "chain_root": str(chain),
        "report": (str(forecast / "report.json")
                   if (forecast / "report.json").is_file() else None),
        "certification_capsule": (str(capsule_path)
                                  if capsule_path.is_file() else None),
        "restarted": False,
    }


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

    # Which chain this config is on.  Read from the config's own [fetch]
    # table rather than from the plan, so a config.path plan lands on the
    # same chain its emission targeted.
    raw = tomllib.load(io.BytesIO(Path(config_path).read_bytes()))
    # Through _chain_key, not a second copy of the same test: the
    # follow-statics refusal in resolve_plan was decided for whichever
    # chain that function names, and this is where the naming becomes a
    # dispatch.  One function, so a config cannot be judged as one chain
    # and then run as the other.
    if _chain_key(plan.route,
                  (raw.get("fetch") or {}).get("source")) == "prepared:hrrr":
        return _hrrr_chain(plan, config_path=Path(config_path), exp=exp,
                           observer=observer, run_dir=plan.run_dir)

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
    # Not a `gpuwm go` CLI flag: it is stamped onto the namespace that
    # go_main reads, the same way go_main reads --outdir.  Adding a flag
    # to `gpuwm go` for it is a separate decision about that command's
    # surface, and this front door does not get to make it.
    args.render_products = plan.run_options.get("render_products")
    # Trees allowed: this front door dispatches to the tree runner, so
    # the interactive refusal `gpuwm go` keeps does not apply.
    code = go_main(args, observer=_GoObserver(observer), allow_tree=True)
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
    return _chain_summary(plan.run_dir / "chain", observer=observer)


ROUTES: dict[str, Route] = {
    "prepared": Route(
        name="prepared",
        summary="the native prepared-cache route: authority, fetch, "
                "manifest, preparation, forecast and render, in the "
                "documented order (what `gpuwm go CONFIG` executes) -- "
                "for sources the config-driven route cannot decode",
        run_options=frozenset({*_RUN_OPTION_DEFAULTS, "data_dir",
                               "geog_root", "physics_profile",
                               "render_products"}),
        needs_case_data=False,
        execute=_execute_prepared_route),
    "experiment": Route(
        name="experiment",
        summary="the config-driven experiment route: one experiment TOML "
                "with its [case_data] inputs, prepared and integrated in "
                "this process (what `gpuwm run CONFIG` executes)",
        run_options=frozenset(_RUN_OPTION_DEFAULTS)
        - {"data_dir", "geog_root", "physics_profile",
           "render_products"},
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

    from gpuwm.provenance_gate import receipt_block
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
        # WHICH TREE is executing this plan.  A front end reattaching to
        # a run, or comparing two runs, has to be able to answer that
        # from the manifest alone -- the pid and the run_id say which
        # process, and nothing here said which CODE until this field.
        "provenance": receipt_block(),
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


#: Remedies for the failure classes whose cause REALLY is known from the
#: class alone: a plan document that failed validation, and a declared
#: input that is not where the config says.  Absent is ``None``, never a
#: guess: a wrong remedy costs a reader more than no remedy.
#:
#: ``ModuleNotFoundError`` is deliberately NOT here any more.  It used
#: to map to the CuPy install line, so a plan run that died on a missing
#: ``wrf``, ``scipy`` or ``shapefile`` told the caller's event stream to
#: install a GPU wheel -- a remedy that is wrong, and wrong in the one
#: channel a front end shows a user verbatim.  Import failures are
#: answered by :func:`gpuwm.capabilities.remedy_for_error`, which reads
#: the MODULE the failure names.
_REMEDIES = {
    "PlanError": "fix the plan document and re-run; nothing was started",
    "FileNotFoundError": "a declared input is not at the path the config "
                         "names; `gpuwm check CONFIG` names all of them",
}


def _remedy(error: BaseException) -> str | None:
    """The remedy for a failed plan run: what is missing, then the class.

    Order matters.  An import failure is asked about FIRST and answered
    from the module it names, so the class table can never speak for a
    dependency it cannot see.
    """

    from gpuwm import capabilities

    derived = capabilities.remedy_for_error(error)
    if derived is not None:
        return derived
    return _REMEDIES.get(type(error).__name__)


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

    from gpuwm.provenance_gate import receipt_block

    events.emit(
        "plan_accepted", name=plan.name, route=plan.route,
        plan_source=plan.source, plan_sha256=plan.sha256,
        run_dir=str(run_dir), manifest_path=str(manifest_path),
        events_path=str(events.path), pid=os.getpid(), run_id=run_id,
        # The first line of the stream names the tree, so a consumer
        # that only ever tails events.jsonl never has to open the
        # manifest to learn which code produced what follows.
        provenance=receipt_block())
    # Time to first plot is measured from HERE -- the instant this run
    # was accepted -- because that is the instant the person who launched
    # it started waiting.  Taken immediately after the event so the two
    # cannot drift by whatever the next few resolution steps cost.
    accepted_wall = time.perf_counter()

    stage = "preflight"
    observer: RunObserver | None = None
    try:
        # BEFORE the fetch, the resolve and the device.  A plan run that
        # cannot import the runtime it is about to integrate on must say
        # so on the first line of its event stream, not after it has
        # spent the caller's bandwidth -- and it must say so with the
        # remedy, because the caller here is usually a program relaying
        # our words to somebody else.
        #
        # `dry_run` is exempt by the same rule `gpuwm go --dry-run` is:
        # it resolves the plan and stops before any device work, so it
        # is exactly the thing a reader with no runtime should be able
        # to run.
        if not plan.run_options.get("dry_run"):
            from gpuwm import capabilities

            capabilities.require(
                "gpuwm run-plan", capabilities.GPU_RUNTIME,
                before=("Refusing at plan acceptance, before the fetch "
                        "stage downloads anything and before a device is "
                        "selected."))
        stage = "prepare"
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
            events, heartbeat=heartbeat, root_domain=exp.root.grid_id,
            accepted_wall=accepted_wall)
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
            # Time to first plot, repeated here from
            # `first_products_ready` so the headline number is on the
            # line every reader already reads.  Null when this run
            # published no products early.
            first_products_seconds=observer.first_products_seconds,
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
            remedy=_remedy(error),
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
    # Taken from the resolution rather than re-derived, so the corridor
    # a caller was told about and the corridor it is quoted a price for
    # are one decision.  A chain that cannot feed a moving nest has
    # already refused inside resolve_plan, above.
    corridor = corridor_estimate(exp, resolution["moving_nest"])
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
            "domains": len(exp.domains),
            "estimate_bytes": int(estimate.alloc_estimate_bytes),
            "estimate_gib": round(
                estimate.alloc_estimate_bytes / 1024 ** 3, 4),
            # The figure a nested plan must be judged on.  alloc_estimate
            # is the POOL REQUEST; the envelope is what the machine has
            # to have free, and it is the tree-aware one --
            # machine_peak_envelope_bytes adds a per-nest term
            # (nests = domains - 1) and, on WDDM, takes the measured
            # footprint floor.  A tree priced on alloc_estimate alone
            # reads as fitting a card it does not fit.
            "peak_envelope_bytes": int(estimate.peak_envelope_bytes),
            "peak_envelope_gib": round(
                estimate.peak_envelope_bytes / 1024 ** 3, 4),
            "basis": "gpuwm.core.preflight.estimate_experiment, which "
                     "sums every domain and shares the scratch arena "
                     "across a tree; the envelope is its "
                     "peak_envelope_bytes (the estimator `gpuwm check` "
                     "reports; no device context is created)",
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
        # The preparation's largest single artifact when a nest moves,
        # and absent-by-arithmetic when none does.  Disk AND host: the
        # runner loads the corridor whole at preflight.
        "corridor": corridor,
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


def render_catalog() -> dict[str, Any]:
    """What may be put in ``render_products``, as JSON.

    The renderer's own answer, asked rather than transcribed.  Which
    engine speaks is part of the answer, not an implementation detail:
    the rust catalog is much larger than the matplotlib fallback's five,
    so a picker built against one and run against the other would offer
    products that do not exist.  The engine is named in the document.

    ``render.py`` already refuses to keep a second copy of the rust
    catalog for exactly this reason; this keeps that promise across the
    machine seam too.
    """

    from gpuwm.render import PRODUCTS, _resolve_engine, fallback_notice

    engine, why = _resolve_engine("auto")
    document: dict[str, Any] = {
        "schema": CATALOG_SCHEMA,
        "engine": engine,
        "engine_notice": fallback_notice(engine, why),
        "spec": "a comma-separated list of the names below, or 'all'; "
                "'none' skips rendering entirely",
        "skip_token": "none",
    }
    if engine == "matplotlib":
        document["products"] = [{"name": name} for name in PRODUCTS]
        document["source"] = "gpuwm.render.PRODUCTS (matplotlib engine)"
        return document

    import subprocess

    from gpuwm import rustwx

    try:
        renderer = rustwx.find_renderer()
        result = subprocess.run(
            [str(renderer), "--list-products"], capture_output=True,
            text=True, errors="replace", env=rustwx.renderer_env(),
            timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        document["products"] = None
        document["error"] = f"{type(error).__name__}: {error}"
        return document
    if result.returncode != 0:
        document["products"] = None
        document["error"] = (result.stderr or "").strip() or             f"renderer exited {result.returncode}"
        return document
    # The renderer INDENTS its product lines and leaves its header and
    # footers flush left.  That is the discriminator, not a guess about
    # which words look like slugs -- and the footer declares the count,
    # so the parse is CHECKED rather than trusted.  A disagreement is
    # reported instead of silently returning a short list to a picker.
    products, groups, declared = [], [], None
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        if line.startswith((" ", "	")):
            products.append({"name": line.strip()})
            continue
        head, sep, tail = line.partition(":")
        if sep and head.strip() == "group keywords":
            groups = [word.strip() for word in tail.split(",") if word.strip()]
            continue
        name, sep, value = line.partition("=")
        if sep and name.strip() == "selectable_slugs":
            try:
                declared = int(value.strip())
            except ValueError:
                declared = None
    document["products"] = products
    document["group_keywords"] = groups
    document["source"] = "the rust renderer's own --list-products"
    if declared is not None and declared != len(products):
        document["parse_warning"] = (
            f"the renderer declared {declared} selectable slugs and this "
            f"read {len(products)}; the raw output is carried below")
        document["raw"] = result.stdout
    return document


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
    from gpuwm.provenance_gate import receipt_block

    document: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "gpuwm_version": str(__version__),
        # "Can I run?" is half a question without "what would run?".
        # A front end that probes one box and then launches on it needs
        # both, and gpuwm_version above is the metadata claim, not the
        # tree.
        "provenance": receipt_block(),
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
            # Not asked is not ready.  Null, explicitly, for the same
            # reason the error arm below is null: a consumer must be
            # able to tell "unknown" from "no".
            "ready": None,
            "basis": "readiness was not requested; the estate check "
                     "creates a CUDA context and this document is the "
                     "poll-safe half.  `ready` is null, meaning UNKNOWN"}
    else:
        try:
            from gpuwm import capabilities
            from gpuwm.doctor import blocking_gaps, collect_checks

            checks = collect_checks()
            # READY MEANS VERIFIED READY.  This field used to be
            # `not blocking_gaps(checks)`, and doctor carries the CuPy
            # check as non-blocking on purpose (an install that has not
            # opted into a GPU wheel is not a broken install) -- so a
            # bare install answered `"ready": true, "blocking_gaps": 0`
            # to a front end whose very next call is `gpuwm run`, which
            # then refuses.  A probe that prints a green light over a
            # hole is worse than one that says nothing.
            #
            # The requirements a RUN needs are asked here directly, by
            # the same registry the run's own front door refuses with,
            # so the two cannot disagree.
            unmet = capabilities.unmet_run_requirements()
            document["readiness"] = {
                "collected": True,
                "checks": [dataclasses.asdict(check)
                           if dataclasses.is_dataclass(check)
                           else dict(check.__dict__) for check in checks],
                "gaps": sum(1 for check in checks
                            if check.status == "missing"),
                "blocking_gaps": len(blocking_gaps(checks)),
                "ready": not blocking_gaps(checks) and not unmet,
                "unmet_run_requirements": [
                    {"module": item.module,
                     "distribution": item.distribution,
                     "extras": list(item.extras),
                     "needed_for": item.unlocks,
                     "remedy": item.remedy} for item in unmet],
                "basis": "gpuwm doctor's own checks, which verify by "
                         "execution and therefore create a CUDA context, "
                         "plus the run front door's own capability "
                         "requirements; `ready` is true only when both "
                         "are satisfied",
            }
        except Exception as error:  # noqa: BLE001 - a probe reports, never raises
            # UNKNOWN, never ready.  `ready` is present and null so a
            # consumer reading the field gets a third answer rather than
            # a missing key it might treat as false -- or, worse, an
            # absent-means-fine default.
            document["readiness"] = {
                "collected": False,
                "ready": None,
                "error": f"{type(error).__name__}: {error}",
                "basis": "readiness could not be established; `ready` is "
                         "null, which means UNKNOWN and never READY"}
    document["routes"] = {
        name: route.summary for name, route in sorted(ROUTES.items())}
    document["schemas"] = {
        "plan": PLAN_SCHEMA, "event": EVENT_SCHEMA,
        "manifest": MANIFEST_SCHEMA, "resolve": RESOLVE_SCHEMA,
        "estimate": ESTIMATE_SCHEMA, "probe": PROBE_SCHEMA,
        "catalog": CATALOG_SCHEMA}
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

    if getattr(args, "catalog", False):
        with contextlib.redirect_stdout(sys.stderr):
            document = render_catalog()
        return answer(document)
    if getattr(args, "probe", False):
        with contextlib.redirect_stdout(sys.stderr):
            document = probe_environment(
                readiness=not getattr(args, "no_readiness", False))
        return answer(document)
    if args.plan is None:
        raise PlanError(
            "gpuwm run-plan needs a PLAN.json, or one of --probe / "
            "--catalog (which need no plan)")

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
        "--catalog", action="store_true",
        help="print the renderer's product catalog as one JSON "
             "document -- what may be put in the render_products run "
             "option -- and run nothing; needs no plan")
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
    "CATALOG_SCHEMA", "DEFAULT_OUTPUT_ROOT", "ESTIMATE_SCHEMA", "EVENTS_FILENAME",
    "EVENT_SCHEMA", "EVENT_TAGS", "MANIFEST_FILENAME", "MANIFEST_SCHEMA",
    "PLAN_SCHEMA", "PROBE_SCHEMA", "RESOLVE_SCHEMA", "ROUTES", "STAGES",
    "GENERATED_CONFIG_NAME",
    "EventStream", "PlanError", "Route", "RunObserver", "RunPlan",
    "build_plan", "collect_warnings", "corridor_estimate",
    "declared_inputs", "domain_size_floor",
    "estimate_plan", "execute_plan", "follow_statics_decision",
    "generate_intent_config", "render_catalog",
    "intent_arguments", "load_plan", "probe_environment", "read_events",
    "register_cli", "resolve_fetch_cycle", "resolve_plan",
    "run_plan_main", "streaming_decision", "write_manifest",
]
