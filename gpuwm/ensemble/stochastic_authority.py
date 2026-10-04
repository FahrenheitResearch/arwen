"""Declared forecast-only stochastic selectors over verified prepared inputs."""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace, asdict
from collections.abc import Mapping
import struct

SPP_FIELDS = ("spp_conv", "spp_pbl", "spp_lsm")


def _same_configuration(original, selected):
    """Compare the original typed config, including exact Python float words."""
    if original is selected:
        return True
    if type(original) is not type(selected):
        return False
    if is_dataclass(original) and not isinstance(original, type):
        return all(_same_configuration(getattr(original, field.name), getattr(selected, field.name))
                   for field in fields(original))
    if isinstance(original, Mapping):
        return original.keys() == selected.keys() and all(
            _same_configuration(value, selected[name]) for name, value in original.items())
    if isinstance(original, (tuple, list)):
        return len(original) == len(selected) and all(
            _same_configuration(before, after) for before, after in zip(original, selected))
    if isinstance(original, float):
        return struct.pack("=d", original) == struct.pack("=d", selected)
    result = original == selected
    if not isinstance(result, bool):
        raise TypeError("runtime stochastic authority requires scalar typed configuration, not model arrays")
    return result


def configure_member_inputs(provider, inputs):
    """Apply only declared SPP switches before the original physics initializer.

    Prepared atmosphere/soil/boundaries, source and append-only head authority
    stay owned by the original verified inputs. Full ordinary input/header
    verification still runs. The selected switches and process parameters
    belong to the member's forecast and lossless stochastic restart owner.
    """
    original = inputs.experiment
    selected = provider.configure_experiment(original)
    if not provider.enabled_for_experiment(selected):
        # Default/off has no new prepared owner or config echo.
        if not _same_configuration(original, selected):
            raise ValueError("disabled stochastic controls changed the ordinary experiment configuration")
        return inputs, None
    if not is_dataclass(original) or not is_dataclass(inputs):
        raise TypeError("active runtime stochastic overlay requires typed original prepared inputs and experiment")
    if len(original.domains) != len(selected.domains):
        raise ValueError("runtime stochastic overlay changed the original domain roster")
    rows, restored_domains = [], []
    for before, after in zip(original.domains, selected.domains):
        baseline = {name: getattr(before.run, name) for name in SPP_FIELDS}
        values = {name: getattr(after.run, name) for name in SPP_FIELDS}
        restored_domains.append(replace(after, run=replace(after.run, **baseline)))
        rows.append({"grid_id": int(before.grid_id), "prepared_flags": baseline,
                     "selected_flags": values})
    restored = replace(selected, domains=tuple(restored_domains))
    if not _same_configuration(original, restored):
        raise ValueError("runtime stochastic overlay may change only declared spp_conv, spp_pbl and spp_lsm; "
                         "physics schemes, clock, domain geometry and source preparation must remain original")
    changed = any(row["prepared_flags"] != row["selected_flags"] for row in rows)
    bound = replace(inputs, experiment=selected) if changed else inputs
    if any(getattr(bound, field.name) is not getattr(inputs, field.name)
           for field in fields(inputs) if field.name != "experiment"):
        raise AssertionError("stochastic overlay replaced an original prepared input authority or array owner")
    def config(value):
        return None if value is None else asdict(value)
    receipt = {"schema": "gpuwm-ensemble-runtime-stochastic-authority.v1",
        "configuration_match": "all original typed configuration fields and float words; only declared SPP flags may differ",
        "prepared_authority_policy": "original source/static/cache/head verification retained without header edits",
        "runtime_overlay": "forecast-only; selected before original PhysicsDriver construction",
        "spp_selectors": rows,
        "process_configurations": {"sppt": config(provider.sppt),
            "skebs_psi": config(provider.skebs_psi), "skebs_theta": config(provider.skebs_theta),
            "spp": {name: config(value) for name, value in (provider.spp_configs or {}).items()},
            "spp_omitted_policy": "WRF reference settings"},
        "prepared_authority_sha256": dict(getattr(inputs, "authority_sha256", {}) or {}),
        "prepared_head_sha256": getattr(inputs, "prepared_head_sha256", None),
        "domains": [{"grid_id": int(domain.grid_id),
            "authority_sha256": dict(getattr(domain, "authority_sha256", {}) or {})}
            for domain in getattr(inputs, "domains", ()) if hasattr(domain, "grid_id")],
        "checkpoint_policy": "member spectrum header binds exact selected process config, seed, shape and consumer levels before restore/forecast"}
    labels = getattr(provider, "wrf_seed_labels", None)
    if labels is not None:
        from gpuwm.ensemble.stochastic_seeds import seed_label_receipt
        receipt["wrf_seed_labels"] = seed_label_receipt(labels)
    return bound, receipt


__all__ = ["configure_member_inputs"]
