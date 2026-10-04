"""Observation-calibration admission for public ensemble requests."""
from __future__ import annotations

# The reason, the selector names and the config scan live in a module the
# config loaders can import without the ensemble runtime; this module is
# where the ensemble doors have always found them.
from gpuwm.ensemble_admission import (
    RANDOM_SELECTORS, UNCALIBRATED_SPREAD_REASON, refuse_configured_random)


def stochastic_requested(controls):
    """Recognize active controls without importing numerical or GPU code."""
    if not isinstance(controls, dict):
        return controls is not None and controls is not False
    if controls.get("spp_configs"):
        return True
    if controls.get("sppt") is not None and controls.get("sppt") is not False:
        return True
    skebs = controls.get("skebs")
    if isinstance(skebs, dict):
        if any(skebs.get(name, True) is not None and skebs.get(name, True) is not False
               for name in ("psi", "theta")):
            return True
    elif skebs is not None and skebs is not False:
        return True
    spp = controls.get("spp")
    return any(bool(value) for value in spp.values()) if isinstance(spp, dict) else bool(spp)


def random_descriptor_requested(perturbation):
    """Recognize this line's new random perturbation descriptors only.

    An absent key, ``"none"`` and the 2.8.4 string references
    (``"gpuwm.da.perturb"``, ``"experimental-stub"``) are not new random
    providers: they keep the behaviour, paths and provenance warnings
    their own owners gave them in 2.8.4.  Refusing them would remove a
    shipped, reachable feature.  The new descriptor is a table.
    """
    return perturbation is not None and not isinstance(perturbation, str)


def refuse_uncalibrated_random(*, perturbation=None, stochastic=None):
    """No caller-supplied amplitude can authorize public random spread."""
    if random_descriptor_requested(perturbation) or stochastic_requested(stochastic):
        raise ValueError(UNCALIBRATED_SPREAD_REASON)


def refuse_native_random(directory):
    """Inspect authored switches before native headers, preprocessing or CUDA."""
    from pathlib import Path
    path = Path(directory) / "namelist.input"
    if path.is_file():
        from gpuwm.fortran_namelist import parse_namelist
        refuse_configured_random(parse_namelist(path))


def refuse_public_arguments(args):
    """Check declared random controls before capability and provenance probes.

    Existing readers still validate the complete config or run-plan schema
    before any admitted request executes.
    """
    import json
    from pathlib import Path
    import tomllib

    def ensemble(value):
        if isinstance(value, dict):
            refuse_uncalibrated_random(perturbation=value.get("perturbation"),
                                      stochastic=value.get("stochastic"))

    def configuration(raw):
        if isinstance(raw, dict):
            refuse_configured_random(raw)
            ensemble(raw.get("ensemble"))

    def config_file(path):
        if path is not None and Path(path).is_file():
            from gpuwm.config_authority import read_config_authority
            configuration(tomllib.loads(read_config_authority(path).payload.decode("utf-8")))

    command = getattr(args, "command", None)
    # `branch` writes a new run folder before its worker starts, so it is
    # checked here like the other doors that take a config.
    if command in {"go", "ensemble", "run", "resume", "branch"}:
        config_file(getattr(args, "config", None))
    config_file(getattr(args, "experiment_config", None))
    for name in ("wrfinput", "met_em"):
        directory = getattr(args, name, None)
        if directory is not None:
            refuse_native_random(directory)
    request = getattr(args, "ensemble_request", None)
    if request is not None:
        ensemble(json.loads(request) if isinstance(request, str) else request)
    if command == "run-plan":
        path = getattr(args, "plan", None)
        if path is not None and Path(path).is_file():
            raw = json.loads(Path(path).read_bytes())
            if isinstance(raw, dict):
                options = raw.get("run_options", {})
                if isinstance(options, dict):
                    ensemble(options.get("ensemble"))
                config = raw.get("config", {})
                if isinstance(config, dict):
                    if isinstance(config.get("inline"), str):
                        configuration(tomllib.loads(config["inline"]))
                    if isinstance(config.get("path"), str):
                        config_file(Path(path).parent / config["path"])
                    configuration(config.get("intent"))


def refuse_unopened_table(path, *, door):
    """Refuse an ``[ensemble]`` table at a forecast door with no ensemble session.

    Breakage it prevents: ``gpuwm sim`` and the prepared runners run the
    one forecast their prepared inputs hold.  Started on their own they
    open no ensemble session, so an ``[ensemble]`` table in the config
    would be dropped and one forecast would run under its name.  A runner
    hosted by a door that did open a session (``gpuwm go``, ``gpuwm
    ensemble``, ``gpuwm run-plan``) inherits it, and is not refused.
    """
    from pathlib import Path
    import tomllib

    if path is None or not Path(path).is_file():
        return
    from gpuwm.ensemble.runtime_context import current_session
    if current_session() is not None:
        return
    from gpuwm.config_authority import read_config_authority
    raw = tomllib.loads(read_config_authority(path).payload.decode("utf-8"))
    if isinstance(raw, dict) and raw.get("ensemble") is not None:
        from gpuwm.ensemble_admission import table_not_honoured
        raise ValueError(table_not_honoured(Path(path).name, door, prepared=True))


def refuse_explicit_config_argv(argv):
    """Check the prepared runners' named config before their startup probes."""
    if "--help" in argv or "-h" in argv:
        return
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment-config")
    args, _ = parser.parse_known_args(argv)
    refuse_public_arguments(args)
    refuse_unopened_table(args.experiment_config, door="the prepared forecast runner")
