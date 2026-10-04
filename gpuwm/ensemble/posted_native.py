"""Reuse the checked ordinary source preparation for native posted members.

These helpers share geometry, solved land/water inputs and source authority.
Each changed member atmosphere still passes through the ordinary native real
initializer and boundary construction owned by its source adapter.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from gpuwm.ensemble.physical_store import digest_file
from gpuwm.ensemble.posted_physical import _link_or_copy

# These are the atmospheric fields changed by native recentering. None is an
# input to the land/skin/soil preparation whose solved arrays are shared here.
ATMOSPHERIC_FIELDS = frozenset({
    "TT", "SPFH", "RH", "UU", "VV", "GHT", "PSFC",
    "T2", "Q2", "RH2", "U10", "V10",
})


def writer_source_wait(writer):
    """Observe a shared source wait using only this member's lifecycle.

    Cancellation stops the member. It never writes a stop marker into the
    common source, which may still be supplying other ensemble members.
    """
    def observe(report):
        writer.check_stop()
        if report is None or report.get("cause") is None:
            writer.source_arrived()
        else:
            writer.waiting_for_source(report["cause"])
    return observe


def checked_source_inputs(context, *, source, experiment_config, wps_namelist,
                          physics_profile=None, expert_acknowledgements=(),
                          history_interval_seconds=None):
    """Run the ordinary source head preflight with its exact original pin."""
    from gpuwm.stage_cli import SINGLE_DOMAIN_RUNNER, missing_forecast_runners
    if SINGLE_DOMAIN_RUNNER in missing_forecast_runners():
        # The standalone RW-WPS wheel stages no forecast runner, and this
        # check IS the runner's preflight.  Without the refusal a provider
        # member there dies on a bare ModuleNotFoundError after its source
        # has been opened, or, beside an editable gpuwm checkout, imports
        # that checkout's runner against this package and dies inside it.
        raise RuntimeError(
            "a posted physical provider member checks its source head with "
            "the forecast runner's preflight, which this preparation-only "
            "installation does not carry; prepare provider members with the "
            "full gpuwm distribution")
    from gpuwm.prepared_single_domain_forecast import preflight_prepared_forecast
    from gpuwm.experiment import load_experiment
    context.verify()
    experiment = load_experiment(Path(experiment_config))
    if history_interval_seconds is None:
        history_interval_seconds = experiment.root.history_interval_s
    result = preflight_prepared_forecast(
        source=source, prepared_root=context.prepared_root,
        prepared_head_sha256=context.prepared_head["head_sha256"],
        source_manifest_sha256=None, experiment_config=Path(experiment_config),
        wps_namelist=Path(wps_namelist), physics_profile=physics_profile,
        expert_acknowledgements=tuple(expert_acknowledgements),
        run_seconds=experiment.run_seconds,
        history_interval_seconds=history_interval_seconds)
    context.verify()
    return result


def _same_array(left, right):
    left, right = np.ascontiguousarray(left), np.ascontiguousarray(right)
    return (left.shape == right.shape and left.dtype == right.dtype
            and np.array_equal(left.view(np.uint8).reshape(-1), right.view(np.uint8).reshape(-1)))


def shared_surface(checked, *, base_met, member_met):
    """Reuse solved native land/water arrays only for identical surface inputs.

    The checked head retains its canonical surface and the meteorological
    fields required by restoration; some raw soil arrays are intentionally
    omitted there. The captured base physical frame retains all raw fields,
    so every member surface ingredient and water metadata is compared against
    that frame as well. Atmospheric changes are initialized normally later.
    """
    if (base_met.valid_time != member_met.valid_time
            or set(base_met.fields) != set(member_met.fields)
            or not _same_array(base_met.levels_hpa, member_met.levels_hpa)):
        raise ValueError("shared native surface needs the same physical time, coordinate and field inventory")
    fields = base_met.fields
    reader = checked.cache_reader
    for name in reader.header["metadata"].get("met_fields", ()):
        if name not in fields or not _same_array(reader.read_array("met/"+name), fields[name]):
            raise ValueError(f"shared native surface base field {name} differs from its checked prepared head")
    for name in fields.keys()-ATMOSPHERIC_FIELDS:
        if not _same_array(fields[name], member_met.fields[name]):
            raise ValueError(f"shared native surface input {name} changed and must be prepared from its own source")
    for name in ("water_temperature", "water_temperature_source", "soil_no_source_land"):
        left, right = getattr(base_met, name, None), getattr(member_met, name, None)
        if (left is None) != (right is None) or left is not None and not _same_array(left, right):
            raise ValueError(f"shared native surface {name} differs from its captured base")
    for name in ("water_temperature_receipt", "horizontal_operators", "masked_field_repairs"):
        if getattr(base_met, name, None) != getattr(member_met, name, None):
            raise ValueError(f"shared native surface {name} authority differs from its captured base")
    names = reader.header["metadata"].get("surface_fields")
    if not names:
        raise ValueError("ordinary source head has no solved native surface to share")
    surface = {name: reader.read_array("surface/"+name) for name in names}
    for value in surface.values():
        value.flags.writeable = False
    # The same dictionaries are used by the native restored-surface path.
    # Keep recorded soil repairs and texture operations with the shared state.
    user = reader.header["metadata"].get("user", {})
    proof = checked.proof
    return SimpleNamespace(fields=MappingProxyType(surface),
        soil_texture_downscale=deepcopy(user.get("soil_texture_downscale", proof.get("soil_texture_downscale"))),
        soil_temperature_repair=deepcopy(user.get("soil_temperature_repair", proof.get("soil_temperature_repair"))))


def copy_common_artifacts(context, checked, destination, *, static_name="native-static.npz",
                          geometry_name="geometry-receipt.json"):
    """Copy verified static/geometry artifacts without rebuilding geography."""
    context.verify()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (static_name, geometry_name):
        if (not isinstance(name, str) or not name or name in (".", "..") or ":" in name
                or Path(name).name != name or "/" in name or "\\" in name):
            raise ValueError("shared native artifact name must stay within the member preparation root")
    paths = {"static": (Path(checked.static_path), static_name),
             "geometry": (Path(checked.geometry_receipt_path), geometry_name)}
    records = {}
    for role, (source, name) in paths.items():
        observed = digest_file(source)
        if role == "static" and observed != checked.cache_identity["static_cache_sha256"]:
            raise ValueError("shared native static bytes changed after ordinary source preflight")
        if role == "geometry":
            import json
            if json.loads(source.read_bytes()) != dict(checked.geometry_receipt):
                raise ValueError("shared native geometry changed after ordinary source preflight")
        target = destination/name
        _link_or_copy(source, target)
        if digest_file(target) != observed:
            raise ValueError("shared native artifact changed while copying its checked bytes")
        records[role] = {"path": target, "sha256": observed}
    context.verify()
    return records


def relay_source_segment(context, index, writer, interval):
    """Reuse raw posted/decoded authorities while writing member endpoints."""
    marker = context.require_interval(index)
    return writer.write_segment(index, interval, relay_marker=marker)
