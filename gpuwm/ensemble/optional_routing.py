"""Metadata-only optional routing hooks retained by generic ensemble builds.

Absent/disabled routing never imports a provider or its product collectors.
An explicit request cannot silently become an atmosphere-only forecast.
"""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import json
import sys


def activation_settings(settings):
    """Default prepared hydro on, without resolving fabric or loading providers.

    A realization/factory/stage list is preparation metadata, not evidence of
    physical coverage or admission. Missing configuration remains a named gap.
    """
    if settings is None:
        return None
    settings = deepcopy(settings)
    if "enabled" in settings and type(settings["enabled"]) is not bool:
        raise ValueError("[hydro] enabled must be a boolean")
    if "enabled" not in settings:
        settings["enabled"] = any(settings.get(key) for key in
                                  ("realization", "stages", "driver_factory"))
        settings["activation_reason"] = ("prepared_configuration_default" if settings["enabled"]
                                          else "no_prepared_hydro_configuration")
    else:
        settings.setdefault("activation_reason", "configuration_enabled" if settings["enabled"]
                            else "configuration_off")
    return settings


def activation_receipt(settings):
    """Startup decision only; never claim coverage, execution or a measured cost."""
    settings = activation_settings(settings) or {}
    return {"schema": "gpuwm-hydro-activation.v1",
            "enabled": settings.get("enabled", False),
            "reason": settings.get("activation_reason", "no_prepared_hydro_configuration"),
            "settings_path": settings.get("settings_path"),
            "realization": settings.get("realization"),
            "scope": "activation policy; fabric coverage and runtime admission are separate",
            # The default-on door's decision (provider, footprint, named gaps),
            # published by the launcher for every member and runner process.
            "door": _door_decision()}


def _door_decision():
    text = os.environ.get("GPUWM_HYDRO_DOOR_DECISION_JSON")
    return None if not text else json.loads(text)


def _file_settings(path):
    if path is None or not Path(path).is_file():
        return None
    import tomllib
    with Path(path).open("rb") as source:
        settings = tomllib.load(source).get("hydro")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("[hydro] must be a TOML table")
    return activation_settings(dict(settings, settings_path=str(Path(path).expanduser().resolve())))


def member_settings(inputs, *, inherited_settings=None):
    """Resolve fresh metadata; a member's table overrides launcher defaults."""
    runtime = sys.modules.get("gpuwm.hydro.runtime")
    # The command's explicit opt-out applies to the entire ensemble, including
    # a worker with a member-specific enabled table or a copied settings snapshot.
    if (os.environ.get("GPUWM_NEXTGEN_ENABLED") == "0"
            or getattr(runtime, "cli_disabled", lambda: False)()
            or (inherited_settings or {}).get("activation_reason") == "cli_off"):
        return {"enabled": False, "activation_reason": "cli_off"}
    settings = _file_settings(getattr(inputs, "experiment_config", None))
    if settings is not None:
        return settings
    if inherited_settings is not None:
        return activation_settings(inherited_settings)
    # In-process launchers may already hold settings without an environment
    # path. Never import a numerical provider just to detect this feature.
    settings = None if runtime is None else runtime.current_settings()
    if settings is None:
        settings = _file_settings(os.environ.get("GPUWM_NEXTGEN_SETTINGS"))
        if os.environ.get("GPUWM_NEXTGEN_ENABLED") == "1":
            settings = dict(settings or {}, enabled=True, activation_reason="launcher_enabled")
    return activation_settings(settings)


def enabled_for(inputs, *, inherited_settings=None):
    settings = member_settings(inputs, inherited_settings=inherited_settings)
    return settings is not None and settings.get("enabled") is True


def mass_member_settings(parents, members, *, products_requested=False):
    """Snapshot selected parent metadata before spawning card workers.

    Mixed routing rosters cannot produce full-roster probabilities. Reject
    them, and explicit products with no routing, before a forecast or lease.
    No provider is imported on the ordinary atmosphere-only path.
    """
    from types import SimpleNamespace
    selected = {row["parent_index"] for row in members}
    settings = {index: member_settings(SimpleNamespace(
        experiment_config=parents[index].get("experiment_config"))) for index in selected}
    result = {row["member_id"]: deepcopy(settings[row["parent_index"]]) for row in members}
    enabled = [value is not None and value.get("enabled") is True for value in result.values()]
    if (products_requested or any(enabled)) and (not enabled or not all(enabled)):
        raise ValueError("NextGen ensemble products require hydro enabled for every member")
    return result


@contextmanager
def member_settings_scope(inputs, *, inherited_settings=None):
    """Bind only this forecast's settings, resetting even after a failed run."""
    settings = member_settings(inputs, inherited_settings=inherited_settings)
    runtime = sys.modules.get("gpuwm.hydro.runtime")
    if settings is None or (settings.get("enabled") is not True and runtime is None):
        yield
        return
    if runtime is None:
        try:
            from gpuwm.hydro import runtime
        except ModuleNotFoundError as error:
            if error.name not in {"gpuwm.hydro", "gpuwm.hydro.runtime"}:
                raise
            raise ValueError("NextGen runtime is unavailable in this build; "
                             "enabled member routing cannot be advanced") from error
    with runtime.settings_scope(settings):
        yield


def product_options(value):
    """Validate explicit optional products only when their consumer ships."""
    try:
        from gpuwm.ensemble.nextgen_points import product_options as validate
    except ModuleNotFoundError as error:
        if error.name != "gpuwm.ensemble.nextgen_points":
            raise
        raise ValueError("NextGen products are unavailable in this build; "
                         "requested reach probabilities cannot be produced") from error
    return validate(value)


def read_product_options(config):
    """The mass launcher's absent-policy path needs no optional package."""
    if config is None:
        return None
    import tomllib
    with Path(config).open("rb") as source:
        policy = tomllib.load(source).get("ensemble", {}).get("hydro")
    return None if policy is None else product_options(policy)
