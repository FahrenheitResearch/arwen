"""Forecast-scoped NextGen settings and parent-only attachment."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import importlib
import os
from pathlib import Path
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


def go_scope(function):
    """Carry CLI settings through in-process runners and child processes."""
    @wraps(function)
    def wrapped(args, *positional, **kwargs):
        requested = getattr(args, "hydro", None)
        # A door with no CONFIG (the public branch-from ensemble, whose
        # members carry their parents' own hydro settings) has no settings
        # file to read and no parent grid for the door to cover.
        config = getattr(args, "config", None)
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
        settings, door_path, decision = _door.resolve_launch(
            settings, requested=requested, config_path=config)
        if door_path is not None:
            path = door_path
            settings = read_settings(door_path)
        previous_decision = os.environ.get(_door.ENV_DECISION)
        _door.publish_decision(decision)
        if settings.get("enabled"):
            settings = _numerics_settings(settings)
        previous = os.environ.get(ENV_SETTINGS)
        previous_enabled = os.environ.get(ENV_ENABLED)
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
        try:
            with settings_scope(settings):
                return function(args, *positional, **kwargs)
        finally:
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
    physics.hydro = hydro
    from .cost import attach as _attach_cost
    _attach_cost(physics)
    return hydro
