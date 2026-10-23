"""Forecast-scoped NextGen settings and parent-only attachment."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import importlib
import os
from pathlib import Path
import threading
import time
import tomllib

from gpuwm.ensemble.optional_routing import activation_settings, activation_receipt
from . import door as _door

_SETTINGS = ContextVar("arwen_nextgen_settings", default=None)
ENV_SETTINGS = "GPUWM_NEXTGEN_SETTINGS"
ENV_ENABLED = "GPUWM_NEXTGEN_ENABLED"


def read_settings(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    settings = document.get("hydro", {})
    if not isinstance(settings, dict):
        raise ValueError("[hydro] must be a TOML table")
    settings = dict(settings)
    settings["settings_path"] = str(path)
    return activation_settings(settings)


@contextmanager
def settings_scope(settings):
    token = _SETTINGS.set(settings)
    try:
        yield
    finally:
        _SETTINGS.reset(token)


def cli_disabled():
    """Inspect the launch scope without opening an inherited settings file."""
    return (_SETTINGS.get() or {}).get("activation_reason") == "cli_off"


def current_settings():
    # A covering fabric still preparing beside the fetch is this launch's
    # answer too: every reader waits for it (see join_beside).
    join_beside()
    settings = _SETTINGS.get()
    if os.environ.get(ENV_ENABLED) == "0":
        return {"enabled": False, "activation_reason": "cli_off"}
    if settings is None:
        inherited = os.environ.get(ENV_SETTINGS)
        if inherited:
            settings = read_settings(inherited)
            if os.environ.get(ENV_ENABLED) == "1":
                settings["enabled"] = True
                settings["activation_reason"] = "launcher_enabled"
    return settings


def current_member_id():
    """Read the ensemble's existing preparation scope; never own its clock."""
    from gpuwm.ensemble.runtime_context import current_capture
    capture = current_capture()
    if capture is None:
        return None
    member_id = capture.member_id
    if type(member_id) is not int or member_id < 0:
        raise ValueError("NextGen member identity must be a nonnegative integer")
    return member_id


def forecast_scope(function):
    """Bind a prepared forecast's experiment without sharing member state."""
    @wraps(function)
    def wrapped(inputs, *args, **kwargs):
        if current_settings() is not None:
            return function(inputs, *args, **kwargs)
        experiment = getattr(inputs, "experiment_config", None)
        settings = read_settings(experiment) if experiment is not None else None
        if (settings is not None and _door.inherited_decision() is None
                and not _door.explicitly_prepared(settings)
                and settings.get("activation_reason") != "configuration_off"):
            # A runner started without a launcher decides for itself, once.
            settings, door_path, decision = _door.resolve_launch(
                settings, requested=None, config_path=experiment)
            if door_path is not None:
                settings = read_settings(door_path)
            previous = os.environ.get(_door.ENV_DECISION)
            _door.publish_decision(decision)
            try:
                with settings_scope(settings):
                    return function(inputs, *args, **kwargs)
            finally:
                if previous is None:
                    os.environ.pop(_door.ENV_DECISION, None)
                else:
                    os.environ[_door.ENV_DECISION] = previous
        with settings_scope(settings):
            return function(inputs, *args, **kwargs)
    return wrapped


def _launch_settings(requested, config, *, plan_only=False, prepared_launch=False, beside=False):
    """The door decision for one launch: (settings, settings_path, decision).

    ``plan_only`` (``gpuwm go --dry-run``) decides without building the
    covering fabric: a dry run prints the plan and launches nothing.
    ``beside`` (a default-on ``gpuwm go`` launch): a covering fabric is
    prepared on a thread beside the launch's fetch and preparation, and the
    bound answer is provisional until :func:`join_beside` (see _Beside).
    """
    if requested == "on":
        requested = ""
    path = config if requested in (None, "", "off") else requested
    if requested not in (None, "", "off") and not Path(path).is_file():
        raise ValueError(f"hydro settings file does not exist: {path}")
    # Explicit opt-out must not depend on the validity or availability of
    # hydro preparation. The atmosphere runner validates its own config.
    if requested == "off":
        settings = {"enabled": False, "activation_reason": "cli_off"}
    elif path is not None and Path(path).is_file():
        settings = read_settings(path)
    else:
        settings = activation_settings({})
    if requested is not None:
        settings["enabled"] = requested != "off"
        settings["activation_reason"] = "cli_off" if requested == "off" else "cli_enabled"
    # Default-on door: with no prepared configuration, find a fabric that
    # covers the parent grid (CONUS recipe, then the global builder) or
    # record the named gap. Explicit settings and --hydro off pass through.
    decided = {}
    # `--hydro on` ("") builds beside too: hydro is opt-in from 2.9.4.1, so the requested launch is the one that
    # builds; a domain no fabric covers still refuses on entry (resolve_launch below), before the fetch.
    if beside and not plan_only and requested in (None, ""):
        decision = _door.decide(settings=settings, requested=requested, config_path=config)
        if decision["reason"] == "fabric_covered":
            return _Beside.start(settings, path, decision, requested=requested, config=config,
                                 prepared_launch=prepared_launch)
        decided["decision"] = decision
    return _finish_launch(*_door.resolve_launch(
        settings, requested=requested, config_path=config, plan_only=plan_only,
        prepared_launch=prepared_launch, **decided), path=path)


def _finish_launch(settings, door_path, decision, *, path):
    if door_path is not None:
        path = door_path
        settings = read_settings(door_path)
    if settings.get("enabled"):
        settings = _numerics_settings(settings)
    return settings, path, decision


#: The bound reason while a covering fabric prepares beside the launch.
FABRIC_PREPARING = "fabric_preparing"


class _Beside:
    """A covering fabric preparing beside one launch's fetch and preparation.

    Breakage it prevents: R15's first run on a bare install spent
    273 s building the Oklahoma fabric (149 s downloading, 156 s building)
    before the fetch started, so the chained root stage came ready 366 s after
    launch against its 300 s bound; nothing in the fetch or the preparation
    reads the fabric. The door decides on entry as before (cheap, no build);
    the build runs on this thread and the launch binds a provisional answer
    (hydro not yet enabled, the decision marked pending). Every reader of the
    final answer joins first: :func:`current_settings` (in-process forecasts,
    members, receipts) and ``gpuwm go``'s forecast stage before it spawns its
    runner (go_cli._run_forecast). The go memory gate does not wait: it prices
    the atmosphere and says the fabric is preparing; the hydro allocation is
    admitted at construction (memory.admit_plan) and by the runner's own gate.
    The receipt keeps the build seconds plus how long the forecast waited.
    """

    _current = None
    _lock = threading.Lock()

    def __init__(self, settings, path, decision, *, requested, config, prepared_launch):
        self.requested, self.path = requested, path
        self.provisional = dict(settings, enabled=False, activation_reason=FABRIC_PREPARING)
        self.decision = dict(_door.public(decision), build={
            "pending": "preparing beside the fetch and the preparation"})
        self.started = time.perf_counter()
        self.result = self.error = None
        self.finished_at = None
        self.joined = False

        def build():
            try:
                self.result = _door.resolve_launch(
                    dict(settings), requested=requested, config_path=config,
                    prepared_launch=prepared_launch, decision=decision)
            except BaseException as error:  # noqa: BLE001 - raised where the answer is read
                self.error = error
            finally:
                self.finished_at = time.perf_counter()

        self.thread = threading.Thread(target=build, name="hydro-fabric-beside", daemon=True)

    @classmethod
    def start(cls, settings, path, decision, **kwargs):
        # One build beside a launch at a time: a nested launch first takes
        # the answer its enclosing launch is still preparing.
        join_beside()
        beside = cls(settings, path, decision, **kwargs)
        with cls._lock:
            cls._current = beside
        beside.thread.start()
        return beside.provisional, None, beside.decision

    @classmethod
    def pending(cls):
        return cls._current is not None

    @classmethod
    def abandon(cls, keep=None):
        """Forget an unjoined build (a launch that stopped before its forecast).

        ``keep``: the enclosing launch's own build, which stays pending.
        """
        with cls._lock:
            cls._current = keep if keep is not None and not keep.joined else None


def beside_pending():
    """True while a launch's covering fabric is still unjoined (no wait)."""
    return _Beside.pending()


def join_beside():
    """Wait for the fabric preparing beside this launch and bind its answer.

    Returns the final door decision, or None when nothing was pending. The
    environment children read (GPUWM_NEXTGEN_SETTINGS/ENABLED, the published
    decision) and the provisional settings bound in the launch scope are
    replaced by the build's answer, so a runner spawned after this call sees
    exactly what a synchronous door would have bound.
    """
    if _Beside._current is None:
        return None
    with _Beside._lock:
        beside = _Beside._current
        if beside is None:
            return None
        waited = time.perf_counter()
        beside.thread.join()
        waited = time.perf_counter() - waited
        beside.joined = True
        _Beside._current = None
    if beside.error is not None:
        raise beside.error
    settings, path, decision = _finish_launch(*beside.result, path=beside.path)
    build = decision.setdefault("build", {})
    build.update(beside=True, waited_seconds=round(waited, 3),
                 seconds=round(beside.finished_at - beside.started, 3))
    _bind_launch(settings, path, decision, beside.requested)
    # The launch scope bound the provisional dict; every context holding it
    # now reads the final answer.
    beside.provisional.clear()
    beside.provisional.update(settings)
    if waited > 1.0:
        print(f"  .. waited {waited:.0f} s for the hydro fabric preparing beside the fetch "
              f"({beside.finished_at - beside.started:.0f} s in all)", flush=True)
    print(_door.launch_line(_door.inherited_decision()), flush=True)
    return decision


def _bind_launch(settings, path, decision, requested):
    """Publish the decision and the subprocess environment for this launch."""
    _door.publish_decision(decision)
    # Only a launcher mutates the subprocess environment. Member workers
    # inherit this path and create their own device state at preparation.
    if settings.get("enabled"):
        if path is not None:
            os.environ[ENV_SETTINGS] = str(Path(path).resolve())
        os.environ[ENV_ENABLED] = "1"
    elif requested == "off":
        os.environ.pop(ENV_SETTINGS, None)
        os.environ[ENV_ENABLED] = "0"
    else:
        # An unrelated inherited launcher must not resurrect a disabled
        # forecast in its child processes.
        os.environ.pop(ENV_SETTINGS, None)
        os.environ.pop(ENV_ENABLED, None)


def _plan_only(args):
    """A launch that answers and runs nothing decides without building.

    ``--dry-run`` prints a plan and ``--readiness`` prints a JSON readiness
    document. Breakage it prevents: `gpuwm go CONFIG --members 2 --readiness`
    on a covered domain fetched the hydro sources and built a fabric (197 s on
    a 5090 host) and its progress lines landed on stdout ahead of the readiness
    JSON, so the answer did not parse.
    """
    return bool(getattr(args, "dry_run", False) or getattr(args, "readiness", False))


#: The reason bound on entry while ``gpuwm go --cycle`` has yet to re-time
#: CONFIG: the door decides once the run's own cycle is in the file.
CYCLE_PENDING = "cycle_pending"


def _cycle_pending(args, requested, config):
    """True when go_main will apply ``--cycle`` to CONFIG after go_scope's entry.

    Breakage it prevents: `gpuwm go RECIPE.toml --cycle X` decided from the
    file's stored [fetch] cycle. For a stored 21Z RAP-boundary recipe the
    door's experiment load looked for RAP 18Z f51, which no archive holds, and
    reported hydro off with no_domain_geometry instead of the run's own answer;
    on a covered domain it would have spun the fabric up for the stored dates.
    """
    return (getattr(args, "cycle", None) is not None and requested != "off"
            and config is not None and Path(config).is_file())


#: The reason bound while an ensemble's disk and card admission is pending:
#: the door builds the covering fabric only once that admission passes.
ADMISSION_PENDING = "admission_pending"

#: Bytes a cold hydro preparation adds per square degree of domain, measured on
#: the 10.2 x 6.4 degree WoFS Oklahoma box (a 5070 Ti host, 2026-10-21): fabric and
#: parameters 2.34 GB, a later start on the same domain 0.97 GB, source tiles
#: 2.25 GB on a cold tile cache.
HYDRO_FULL_BYTES_PER_DEG2 = 36e6
HYDRO_START_BYTES_PER_DEG2 = 15e6
HYDRO_TILE_BYTES_PER_DEG2 = 35e6


def _admission_pending(args):
    """True for a launch whose members are admitted (disk, card) after go_main
    starts: the ensemble and recipe routes. Breakage it prevents: the 4-member
    WoFS ensemble spent 369 s fetching hydro sources and 276 s on initial water,
    then its disk admission refused the run and said nothing was spent."""
    if _plan_only(args) or getattr(args, "hydro", None) == "off":
        return False
    members = getattr(args, "members", None) or 1
    return (getattr(args, "command", None) == "ensemble" or members > 1
            or getattr(args, "recipe", None) is not None)


def _admission_decision(requested):
    return {"schema": _door.SCHEMA, "requested": requested, "enabled": False,
            "reason": ADMISSION_PENDING,
            "gaps": [{"owner": "hydro-world-door",
                      "gap": "decided once the ensemble's disk and card admission passes; "
                             "this launch stopped before that"}]}


def _bind_pending(args, requested, decision):
    settings = activation_settings({}) | {"enabled": False, "activation_reason": decision["reason"]}
    _bind_launch(settings, None, decision, requested)
    _SETTINGS.set(settings)
    return decision


def hydro_disk_estimate(config, args=None):
    """(bytes, sentence): what the default-on hydro preparation would add on
    disk for CONFIG, decided plan-only (nothing built), and where it lands."""
    requested = getattr(args, "hydro", None) if args is not None else None
    if requested == "off" or config is None or not Path(config).is_file():
        return 0, None
    try:
        settings, _path, decision = _launch_settings(requested, config, plan_only=True)
    except Exception:  # noqa: BLE001 - an unanswerable door prices nothing; it decides later
        return 0, None
    if decision.get("reason") != "fabric_covered":
        return 0, None
    foot = decision.get("footprint") or {}
    try:
        width = (foot["east"] - foot["west"]) % 360.0 or 360.0
        area = max(width * (foot["north"] - foot["south"]), 0.0)
    except (KeyError, TypeError):
        return 0, None
    root = _door.cache_root(config) / str(decision.get("provider"))
    static = any(root.glob("static-*/static-ready.json")) if root.is_dir() else False
    fabric = area * (HYDRO_START_BYTES_PER_DEG2 if static else HYDRO_FULL_BYTES_PER_DEG2)
    tiles = 0.0 if static else area * HYDRO_TILE_BYTES_PER_DEG2
    total = int(fabric + tiles)
    return total, (f"hydro preparation about {total / 2**30:.1f} GiB over {area:.0f} square degrees "
                   f"({'a later start on a prepared domain' if static else 'fabric, parameters and source tiles'}; "
                   f"cache {root.parent})")


def _pending_decision(requested):
    return {"schema": _door.SCHEMA, "requested": requested, "enabled": False,
            "reason": CYCLE_PENDING,
            "gaps": [{"owner": "hydro-world-door",
                      "gap": "decided once --cycle re-times the configuration; "
                             "this launch stopped before that"}]}


def rebind_launch(args, *, cycle_applied=False, admitted=False, beside=False):
    """Re-ask the door once the launch's configuration is final.

    go_scope decides on entry, before go_main materializes the domain
    configuration of a --source/--init/--center launch (no CONFIG), so that
    entry decision is no_domain_geometry. Breakage it prevents: every member
    of `gpuwm ensemble --source hrrr --center ...` ran with hydro off on a
    CONUS domain the CONUS fabric covers. A ``gpuwm go --cycle`` launch is
    decided here too, with ``cycle_applied=True``, once go has re-timed CONFIG
    (see :func:`_cycle_pending`). Call it inside go_scope only; leaving
    go_scope restores the environment and the bound settings.
    """
    if _SETTINGS.get() is None:
        return None
    decision = _door.inherited_decision()
    config = getattr(args, "config", None)
    requested = getattr(args, "hydro", None)
    reasons = ((ADMISSION_PENDING,) if admitted else (CYCLE_PENDING,) if cycle_applied
               else ("no_domain_geometry",))
    if (decision is None or decision.get("reason") not in reasons
            or requested == "off" or config is None or not Path(config).is_file()):
        return None
    if not admitted and _admission_pending(args):
        # The ensemble door admits its members (disk, card) first; the
        # fabric is built after that (rebind_launch(args, admitted=True)).
        return _bind_pending(args, requested, _admission_decision(requested))
    settings, path, decision = _launch_settings(
        requested, config, plan_only=_plan_only(args),
        prepared_launch=getattr(args, "prepared_root", None) is not None,
        beside=beside and not admitted)
    _bind_launch(settings, path, decision, requested)
    _SETTINGS.set(settings)
    return decision


@contextmanager
def run_scope(requested, config=None):
    """``gpuwm run --hydro off|SETTINGS.toml``: go's hydro opt-out on the run door.

    ``off`` binds the cli_off decision (no fabric probe, no build) and the
    environment every forecast worker reads (GPUWM_NEXTGEN_ENABLED=0); a
    settings file binds those settings as go binds them. With no flag the
    run door keeps deciding in its own worker (forecast_scope). Breakage it
    prevents: `gpuwm run` had no hydro opt-out, so the R6 identity arms (whose
    reference predates hydro on that domain) could not run with hydro off.
    """
    if requested is None:
        yield None
        return
    if requested == "on":
        requested = ""
    settings, path, decision = _launch_settings(requested, config)
    saved = {name: os.environ.get(name) for name in (_door.ENV_DECISION, ENV_SETTINGS, ENV_ENABLED)}
    token = None
    try:
        _bind_launch(settings, path, decision, requested)
        token = _SETTINGS.set(settings)
        yield decision
    finally:
        if token is not None:
            _SETTINGS.reset(token)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def go_scope(function=None, *, beside=False):
    """Carry CLI settings through in-process runners and child processes.

    ``@go_scope(beside=True)`` (``gpuwm go``): a covering fabric prepares
    beside the launch's fetch and preparation (_Beside); the launch joins it
    before its forecast, and on return. A launch that raises does not wait.
    """
    if function is None:
        return lambda target: go_scope(target, beside=beside)

    @wraps(function)
    def wrapped(args, *positional, **kwargs):
        requested = getattr(args, "hydro", None)
        outer = _Beside._current
        # A door with no CONFIG (the public branch-from ensemble, whose
        # members carry their parents' own hydro settings, or a source-only
        # launch before go_main writes its domain; see rebind_launch) has no
        # settings file to read and no parent grid for the door to cover yet.
        config = getattr(args, "config", None)
        if (not _cycle_pending(args, requested, config) and _admission_pending(args)
                and config is not None and Path(config).is_file()):
            settings, path, decision = (activation_settings({}) | {"enabled": False,
                                        "activation_reason": ADMISSION_PENDING},
                                        None, _admission_decision(requested))
        elif _cycle_pending(args, requested, config):
            # Decided by rebind_launch(args, cycle_applied=True) once go has
            # re-timed CONFIG to --cycle; until then hydro is bound off.
            settings, path, decision = (activation_settings({}) | {"enabled": False,
                                        "activation_reason": CYCLE_PENDING},
                                        None, _pending_decision(requested))
        else:
            settings, path, decision = _launch_settings(
                requested, config, plan_only=_plan_only(args),
                prepared_launch=getattr(args, "prepared_root", None) is not None,
                beside=beside)
        previous_decision = os.environ.get(_door.ENV_DECISION)
        previous = os.environ.get(ENV_SETTINGS)
        previous_enabled = os.environ.get(ENV_ENABLED)
        token = None
        enclosing = outer
        try:
            _bind_launch(settings, path, decision, requested)
            # A plain token (not settings_scope) so rebind_launch can replace
            # the bound settings and this reset still restores the caller's.
            token = _SETTINGS.set(settings)
            result = function(args, *positional, **kwargs)
            # A --prepare-only launch (or one whose forecast never read the
            # answer) still leaves the domain's fabric complete in the cache.
            if _Beside._current is not enclosing:
                join_beside()
            return result
        finally:
            _Beside.abandon(keep=enclosing)
            if token is not None:
                _SETTINGS.reset(token)
            if previous_decision is None:
                os.environ.pop(_door.ENV_DECISION, None)
            else:
                os.environ[_door.ENV_DECISION] = previous_decision
            if previous is None:
                os.environ.pop(ENV_SETTINGS, None)
            else:
                os.environ[ENV_SETTINGS] = previous
            if previous_enabled is None:
                os.environ.pop(ENV_ENABLED, None)
            else:
                os.environ[ENV_ENABLED] = previous_enabled
    return wrapped


def _numerics_settings(settings):
    """Inherit the process mode unless the prepared configuration names one."""
    from gpuwm import numerics
    settings = dict(settings)
    settings["numerics"] = (numerics.select(settings["numerics"])
                            if "numerics" in settings else numerics.mode())
    return settings


def resolve_settings(settings):
    """Expand composition once before device setup or duration validation."""
    if settings.get("enabled") is not True:
        raise ValueError("hydro preparation requires [hydro] enabled=true")
    # Launch decisions belong to activation receipts, not the realization
    # sealed into checkpoints. Preserve the caller's decision metadata.
    settings = {key: value for key, value in settings.items() if key != "activation_reason"}
    # Bind before configure seals the realization, so factories, compiler and
    # checkpoint composition all describe the same process arithmetic.
    settings = _numerics_settings(settings)
    if settings.get("realization") is not None:
        from .config import configure
        settings = configure(settings)
    return settings


def prepare(settings, *, state=None, physics=None, device_id=None, mode="standalone"):
    return _prepare_resolved(resolve_settings(settings), state=state, physics=physics,
                             device_id=device_id, mode=mode)


def _prepare_resolved(settings, *, state=None, physics=None, device_id=None, mode="standalone"):
    settings = _numerics_settings(settings)
    if device_id is None:
        import cupy as cp
        device_id = cp.cuda.runtime.getDevice()
    target = settings.get("driver_factory", "gpuwm.hydro.assembly:prepare")
    module, separator, name = target.partition(":")
    if not separator or not module.startswith("gpuwm."):
        raise ValueError("hydro driver_factory must name a gpuwm module:callable")
    factory = getattr(importlib.import_module(module), name)
    driver = factory(settings=settings, state=state, physics=physics,
                     device_id=int(device_id), mode=mode)
    from .driver import ResidentHydroDriver
    if not isinstance(driver, ResidentHydroDriver) or driver.device_id != int(device_id):
        raise ValueError("hydro factory returned no resident driver on the parent card")
    if driver.member_id != current_member_id():
        raise ValueError("hydro factory returned a driver prepared for another member")
    return driver


def attach_parent(physics, state, cfg):
    settings = current_settings()
    if not settings or not settings.get("enabled") or int(getattr(cfg, "grid_id", 1)) != 1:
        return None
    from gpuwm.core.physics_inventory import resolve_physics_slot
    if resolve_physics_slot("sf_surface_physics", cfg.sf_surface_physics) is None:
        raise ValueError("coupled NextGen requires an active parent LSM cadence")
    reference = next(iter(physics.fields.values()))
    # The parent hooks wrap every active LSM. Component factories validate
    # their own field contracts (the Noah-MP runoff exchange still requires
    # Noah-MP); meteorological forcing/runoff modules need no such restriction.
    if physics.noahmp_geometry is not None:
        physics.noahmp_geometry.device_static_fields(
            (cfg.ny, cfg.nx), physics.noahmp_soil_thickness_m)
    hydro = prepare(settings, state=state, physics=physics,
                    device_id=reference.device.id, mode="coupled")
    if hydro.interval_seconds != float(physics.bldt_seconds):
        raise ValueError("NextGen realization interval must equal the parent LSM cadence")
    if bool(getattr(cfg, "use_adaptive_time_step", False)):
        # The adaptive clock moves the LSM interval every step; the stages run
        # on their own fixed time intervals instead (driver docstring).
        hydro.allow_variable_interval()
    physics.hydro = hydro
    from .cost import attach as _attach_cost
    _attach_cost(physics)
    return hydro
