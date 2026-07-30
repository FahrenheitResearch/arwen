"""Semantic CPU/CUDA parity for direct native WRF input products.

WRF stores pressure, geopotential, and dry mass as base plus perturbation
fields.  Its lateral boundary arrays are additionally dry-mass coupled.
Comparing those serialized components with one generic tolerance is not a
physical comparison: benign FP32 cancellation can make a perturbation look
large, while a large coupled value can make a primitive-field error look
small.  This module reconstructs the quantities consumed by WRF before
applying frozen, unit-aware backend tolerances.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Mapping

import netCDF4
import numpy as np

from gpuwm.ingest.backend_contract import ArrayParityRule


WRF_PARITY_SCHEMA = "gpuwm-native-wrf-backend-parity-v1"

PRESSURE_RULE = ArrayParityRule(rtol=3.0e-5, atol=5.0e-3)
GEOPOTENTIAL_RULE = ArrayParityRule(rtol=3.0e-5, atol=1.0e-1)
WIND_RULE = ArrayParityRule(rtol=3.0e-5, atol=5.0e-3)
TEMPERATURE_RULE = ArrayParityRule(rtol=3.0e-5, atol=5.0e-3)
MOISTURE_RULE = ArrayParityRule(rtol=3.0e-5, atol=1.0e-6)
SPECIFIC_VOLUME_RULE = ArrayParityRule(rtol=3.0e-5, atol=5.0e-6)
EXACT_RULE = ArrayParityRule(mode="byte_exact", rtol=0.0, atol=0.0)

_WIND_FIELDS = {"U", "V", "W", "U10", "V10"}
_TEMPERATURE_FIELDS = {
    "T", "THM", "T_INIT", "T2", "TH2", "TSK", "TSLB", "TMN", "SST",
}
_MOISTURE_FIELDS = {
    "Q2", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
}
_PRESSURE_FIELDS = {"P_HYD", "PSFC"}
_SEMANTIC_WRFINPUT_RAW = {"P", "PH", "MU"}
_LBC_LOGICAL = {
    "U": "u", "V": "v", "T": "theta", "PH": "phi",
    "MU": "mu", "QVAPOR": "qv",
}
_SIDE_SUFFIX = {
    "XS": "west", "XE": "east", "YS": "south", "YE": "north",
}


def _attribute_value(value):
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attributes(dataset_or_variable) -> dict[str, object]:
    return {
        name: _attribute_value(dataset_or_variable.getncattr(name))
        for name in sorted(dataset_or_variable.ncattrs())
    }


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _array_record(left, right, rule: ArrayParityRule) -> dict[str, object]:
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    result: dict[str, object] = {
        "rule": asdict(rule),
        "reference_shape": list(left.shape),
        "candidate_shape": list(right.shape),
        "reference_dtype": left.dtype.str,
        "candidate_dtype": right.dtype.str,
    }
    if left.shape != right.shape or left.dtype != right.dtype:
        return {**result, "status": "FAIL", "reason": "shape_or_dtype",
                "max_abs": None, "max_rel": None,
                "rms_abs": None, "p99_abs": None, "violations": None}
    if np.issubdtype(left.dtype, np.number):
        finite_left = np.isfinite(left)
        finite_right = np.isfinite(right)
        if not bool(finite_left.all() and finite_right.all()):
            return {
                **result, "status": "FAIL", "reason": "non_finite",
                "non_finite_reference": int((~finite_left).sum()),
                "non_finite_candidate": int((~finite_right).sum()),
                "max_abs": None, "max_rel": None,
                "rms_abs": None, "p99_abs": None, "violations": None,
            }
    if rule.mode == "byte_exact":
        passed = left.tobytes(order="C") == right.tobytes(order="C")
        return {
            **result, "status": "PASS" if passed else "FAIL",
            "reason": "byte_exact", "max_abs": 0.0 if passed else None,
            "max_rel": 0.0 if passed else None,
            "rms_abs": 0.0 if passed else None,
            "p99_abs": 0.0 if passed else None,
            "violations": 0 if passed else None,
        }
    left64 = left.astype(np.float64, copy=False)
    right64 = right.astype(np.float64, copy=False)
    delta = np.abs(left64 - right64)
    scale = np.maximum(np.abs(left64), np.abs(right64))
    bound = rule.atol + rule.rtol * scale
    relative = np.divide(
        delta, scale, out=np.zeros_like(delta), where=scale != 0.0)
    flat = delta.reshape(-1)
    worst_flat_index = int(np.argmax(flat)) if flat.size else 0
    worst_index = ([int(value) for value in
                    np.unravel_index(worst_flat_index, delta.shape)]
                   if flat.size else [])
    ratio = np.divide(
        delta, bound, out=np.zeros_like(delta), where=bound != 0.0)
    ratio_flat_index = int(np.argmax(ratio)) if flat.size else 0
    violations = int(np.count_nonzero(delta > bound))
    return {
        **result, "status": "PASS" if violations == 0 else "FAIL",
        "reason": "numeric", "max_abs": float(delta.max(initial=0.0)),
        "max_rel": float(relative.max(initial=0.0)),
        "rms_abs": float(np.sqrt(np.mean(delta * delta))) if flat.size else 0.0,
        "p99_abs": float(np.quantile(flat, 0.99)) if flat.size else 0.0,
        "max_abs_index": worst_index,
        "reference_at_max_abs": (
            float(left64[tuple(worst_index)]) if flat.size else None),
        "candidate_at_max_abs": (
            float(right64[tuple(worst_index)]) if flat.size else None),
        "max_bound_ratio": float(ratio.max(initial=0.0)),
        "max_bound_ratio_index": (
            [int(value) for value in
             np.unravel_index(ratio_flat_index, ratio.shape)]
            if flat.size else []),
        "violations": violations,
    }


def _wrfinput_rule(name: str) -> ArrayParityRule:
    if name in _WIND_FIELDS:
        return WIND_RULE
    if name in _TEMPERATURE_FIELDS:
        return TEMPERATURE_RULE
    if name in _MOISTURE_FIELDS:
        return MOISTURE_RULE
    if name in _PRESSURE_FIELDS:
        return PRESSURE_RULE
    if name == "AL":
        return SPECIFIC_VOLUME_RULE
    return EXACT_RULE


def _dimensions(dataset) -> dict[str, int]:
    return {name: len(value) for name, value in dataset.dimensions.items()}


def _metadata_record(reference, candidate) -> dict[str, object]:
    reference_dimensions = _dimensions(reference)
    candidate_dimensions = _dimensions(candidate)
    dimensions_match = reference_dimensions == candidate_dimensions
    attributes_match = _attributes(reference) == _attributes(candidate)
    reference_variables = set(reference.variables)
    candidate_variables = set(candidate.variables)
    inventory_match = reference_variables == candidate_variables
    variable_attributes_match = inventory_match and all(
        _attributes(reference.variables[name]) ==
        _attributes(candidate.variables[name])
        for name in reference_variables)
    status = all((dimensions_match, attributes_match, inventory_match,
                  variable_attributes_match))
    return {
        "status": "PASS" if status else "FAIL",
        "dimensions_match": dimensions_match,
        "reference_dimensions": reference_dimensions,
        "candidate_dimensions": candidate_dimensions,
        "global_attributes_match": attributes_match,
        "variable_inventory_match": inventory_match,
        "reference_only_variables": sorted(
            reference_variables - candidate_variables),
        "candidate_only_variables": sorted(
            candidate_variables - reference_variables),
        "variable_attributes_match": variable_attributes_match,
    }


def _field(dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    value = np.asarray(variable[:])
    if variable.dimensions and variable.dimensions[0] == "Time":
        if value.shape[0] != 1:
            raise ValueError(f"wrfinput {name} must have exactly one Time")
        value = value[0]
    return np.asarray(value)


def _compare_wrfinput(reference, candidate) -> dict[str, object]:
    metadata = _metadata_record(reference, candidate)
    fields: dict[str, object] = {}
    common = sorted(set(reference.variables) & set(candidate.variables))
    for name in common:
        if name in _SEMANTIC_WRFINPUT_RAW:
            continue
        fields[name] = _array_record(
            np.asarray(reference.variables[name][:]),
            np.asarray(candidate.variables[name][:]),
            _wrfinput_rule(name),
        )
    required = {"P", "PB", "PH", "PHB", "MU", "MUB"}
    missing = sorted(required - set(common))
    if not missing:
        fields["SEMANTIC_TOTAL_PRESSURE"] = _array_record(
            _field(reference, "P").astype(np.float64)
            + _field(reference, "PB").astype(np.float64),
            _field(candidate, "P").astype(np.float64)
            + _field(candidate, "PB").astype(np.float64),
            PRESSURE_RULE,
        )
        fields["SEMANTIC_TOTAL_GEOPOTENTIAL"] = _array_record(
            _field(reference, "PH").astype(np.float64)
            + _field(reference, "PHB").astype(np.float64),
            _field(candidate, "PH").astype(np.float64)
            + _field(candidate, "PHB").astype(np.float64),
            GEOPOTENTIAL_RULE,
        )
        fields["SEMANTIC_TOTAL_DRY_MASS"] = _array_record(
            _field(reference, "MU").astype(np.float64)
            + _field(reference, "MUB").astype(np.float64),
            _field(candidate, "MU").astype(np.float64)
            + _field(candidate, "MUB").astype(np.float64),
            PRESSURE_RULE,
        )
    status = (metadata["status"] == "PASS" and not missing
              and all(value["status"] == "PASS" for value in fields.values()))
    return {
        "status": "PASS" if status else "FAIL",
        "metadata": metadata,
        "missing_semantic_variables": missing,
        "failed_fields": [name for name, value in fields.items()
                          if value["status"] != "PASS"],
        "fields": fields,
    }


def _boundary_native(value: np.ndarray, logical: str,
                     side: str) -> np.ndarray:
    """Invert ``wrf_direct._lbc_to_wrf`` for one time record."""
    value = np.asarray(value)
    x_side = side in {"west", "east"}
    if logical == "mu":
        if value.ndim != 2:
            raise ValueError("WRF MU boundary record must be 2-D")
        return (value.T if x_side else value)[None]
    if value.ndim != 3:
        raise ValueError("WRF 3-D boundary record must have three dimensions")
    return (np.transpose(value, (1, 2, 0)) if x_side
            else np.transpose(value, (1, 0, 2)))


def _strip(field: np.ndarray, side: str, width: int) -> np.ndarray:
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError("boundary strip source must be two dimensional")
    if side == "west":
        return field[:, :width]
    if side == "east":
        return field[:, -width:][:, ::-1]
    if side == "south":
        return field[:width, :]
    if side == "north":
        return field[-width:, :][::-1, :]
    raise ValueError(f"unknown boundary side {side!r}")


def _u_face_mass(mass: np.ndarray, side: str) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float32)
    if side in {"west", "east"}:
        result = np.empty_like(mass)
        result[:, 0] = mass[:, 0]
        result[:, 1:] = np.asarray(
            np.float32(0.5) * (mass[:, 1:] + mass[:, :-1]),
            dtype=np.float32)
        return result
    result = np.empty((mass.shape[0], mass.shape[1] + 1), dtype=np.float32)
    result[:, 0] = mass[:, 0]
    result[:, -1] = mass[:, -1]
    result[:, 1:-1] = np.asarray(
        np.float32(0.5) * (mass[:, 1:] + mass[:, :-1]), dtype=np.float32)
    return result


def _v_face_mass(mass: np.ndarray, side: str) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float32)
    if side in {"south", "north"}:
        result = np.empty_like(mass)
        result[0, :] = mass[0, :]
        result[1:, :] = np.asarray(
            np.float32(0.5) * (mass[1:, :] + mass[:-1, :]),
            dtype=np.float32)
        return result
    result = np.empty((mass.shape[0] + 1, mass.shape[1]), dtype=np.float32)
    result[0, :] = mass[0, :]
    result[-1, :] = mass[-1, :]
    result[1:-1, :] = np.asarray(
        np.float32(0.5) * (mass[1:, :] + mass[:-1, :]), dtype=np.float32)
    return result


def _uncouple_endpoint(
        logical: str, side: str, coupled: np.ndarray,
        mup: np.ndarray, *, mub: np.ndarray,
        c1h: np.ndarray, c2h: np.ndarray,
        c1f: np.ndarray, c2f: np.ndarray,
        mapfac_u: np.ndarray, mapfac_v: np.ndarray,
) -> np.ndarray:
    """Return the primitive WRF field represented by one coupled endpoint."""
    coupled = np.asarray(coupled, dtype=np.float64)
    mup = np.asarray(mup, dtype=np.float64)
    if mup.shape[0] != 1:
        raise ValueError("MU boundary endpoint must have one vertical level")
    width = mup.shape[-1] if side in {"west", "east"} else mup.shape[-2]
    total_mass = np.asarray(
        _strip(mub, side, width) + mup[0], dtype=np.float32)
    if logical == "mu":
        return total_mass.astype(np.float64)
    if logical == "u":
        face_mass = _u_face_mass(total_mass, side)
        weight = np.asarray(
            c1h[:, None, None] * face_mass[None] + c2h[:, None, None],
            dtype=np.float32)
        map_factor = _strip(mapfac_u, side, width)
        return coupled * map_factor[None].astype(np.float64) / weight
    if logical == "v":
        face_mass = _v_face_mass(total_mass, side)
        weight = np.asarray(
            c1h[:, None, None] * face_mass[None] + c2h[:, None, None],
            dtype=np.float32)
        map_factor = _strip(mapfac_v, side, width)
        return coupled * map_factor[None].astype(np.float64) / weight
    if logical in {"theta", "qv"}:
        weight = np.asarray(
            c1h[:, None, None] * total_mass[None] + c2h[:, None, None],
            dtype=np.float32)
    elif logical == "phi":
        weight = np.asarray(
            c1f[:, None, None] * total_mass[None] + c2f[:, None, None],
            dtype=np.float32)
    else:
        raise ValueError(f"unknown coupled boundary field {logical!r}")
    return coupled / weight


def _timestamp_record(value: np.ndarray) -> str:
    raw = np.asarray(value)
    if raw.dtype.kind == "S":
        return b"".join(raw.tolist()).decode("ascii")
    return "".join(str(item) for item in raw.tolist())


def _boundary_durations(dataset) -> list[float]:
    current_name = "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
    next_name = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
    if current_name not in dataset.variables or next_name not in dataset.variables:
        raise ValueError("wrfbdy lacks explicit current/next boundary timestamps")
    current = np.asarray(dataset.variables[current_name][:])
    following = np.asarray(dataset.variables[next_name][:])
    if current.shape != following.shape or current.ndim != 2:
        raise ValueError("wrfbdy boundary timestamp arrays are malformed")
    durations = []
    for index in range(current.shape[0]):
        left = datetime.strptime(
            _timestamp_record(current[index]), "%Y-%m-%d_%H:%M:%S")
        right = datetime.strptime(
            _timestamp_record(following[index]), "%Y-%m-%d_%H:%M:%S")
        duration = (right - left).total_seconds()
        if duration <= 0.0:
            raise ValueError("wrfbdy boundary duration must be positive")
        durations.append(duration)
    return durations


def _input_coupling(input_dataset) -> dict[str, np.ndarray]:
    required = {"MUB", "C1H", "C2H", "C1F", "C2F",
                "MAPFAC_U", "MAPFAC_V"}
    missing = sorted(required - set(input_dataset.variables))
    if missing:
        raise ValueError(f"wrfinput lacks boundary coupling fields {missing}")
    return {
        "mub": np.asarray(_field(input_dataset, "MUB"), dtype=np.float32),
        "c1h": np.asarray(_field(input_dataset, "C1H"), dtype=np.float32),
        "c2h": np.asarray(_field(input_dataset, "C2H"), dtype=np.float32),
        "c1f": np.asarray(_field(input_dataset, "C1F"), dtype=np.float32),
        "c2f": np.asarray(_field(input_dataset, "C2F"), dtype=np.float32),
        "mapfac_u": np.asarray(
            _field(input_dataset, "MAPFAC_U"), dtype=np.float32),
        "mapfac_v": np.asarray(
            _field(input_dataset, "MAPFAC_V"), dtype=np.float32),
    }


def _boundary_rule(logical: str) -> ArrayParityRule:
    if logical in {"u", "v"}:
        return WIND_RULE
    if logical == "theta":
        return TEMPERATURE_RULE
    if logical == "phi":
        return GEOPOTENTIAL_RULE
    if logical == "qv":
        return MOISTURE_RULE
    if logical == "mu":
        return PRESSURE_RULE
    raise ValueError(logical)


def _active_boundary_names() -> set[str]:
    names = set()
    for wrf_name in _LBC_LOGICAL:
        for suffix in _SIDE_SUFFIX:
            names.add(f"{wrf_name}_B{suffix}")
            names.add(f"{wrf_name}_BT{suffix}")
    return names


def _compare_wrfbdy(reference, candidate, reference_input,
                    candidate_input) -> dict[str, object]:
    metadata = _metadata_record(reference, candidate)
    fields: dict[str, object] = {}
    common = sorted(set(reference.variables) & set(candidate.variables))
    active = _active_boundary_names()
    for name in common:
        if name in active:
            continue
        fields[name] = _array_record(
            np.asarray(reference.variables[name][:]),
            np.asarray(candidate.variables[name][:]), EXACT_RULE)
    missing = sorted(active - set(common))
    try:
        reference_durations = _boundary_durations(reference)
        candidate_durations = _boundary_durations(candidate)
    except ValueError as error:
        return {
            "status": "FAIL", "metadata": metadata,
            "missing_semantic_variables": missing,
            "boundary_time_error": str(error), "failed_fields": [],
            "fields": fields,
        }
    if reference_durations != candidate_durations:
        return {
            "status": "FAIL", "metadata": metadata,
            "missing_semantic_variables": missing,
            "boundary_time_error": "reference/candidate durations differ",
            "failed_fields": [], "fields": fields,
        }
    reference_coupling = _input_coupling(reference_input)
    candidate_coupling = _input_coupling(candidate_input)
    if not missing:
        for record, duration in enumerate(reference_durations):
            for suffix, side in _SIDE_SUFFIX.items():
                ref_mu_value = _boundary_native(
                    reference.variables[f"MU_B{suffix}"][record], "mu", side)
                ref_mu_tendency = _boundary_native(
                    reference.variables[f"MU_BT{suffix}"][record], "mu", side)
                cand_mu_value = _boundary_native(
                    candidate.variables[f"MU_B{suffix}"][record], "mu", side)
                cand_mu_tendency = _boundary_native(
                    candidate.variables[f"MU_BT{suffix}"][record], "mu", side)
                ref_mu_endpoints = (
                    ref_mu_value,
                    ref_mu_value.astype(np.float64)
                    + ref_mu_tendency.astype(np.float64) * duration,
                )
                cand_mu_endpoints = (
                    cand_mu_value,
                    cand_mu_value.astype(np.float64)
                    + cand_mu_tendency.astype(np.float64) * duration,
                )
                for wrf_name, logical in _LBC_LOGICAL.items():
                    ref_value = _boundary_native(
                        reference.variables[f"{wrf_name}_B{suffix}"][record],
                        logical, side)
                    ref_tendency = _boundary_native(
                        reference.variables[f"{wrf_name}_BT{suffix}"][record],
                        logical, side)
                    cand_value = _boundary_native(
                        candidate.variables[f"{wrf_name}_B{suffix}"][record],
                        logical, side)
                    cand_tendency = _boundary_native(
                        candidate.variables[f"{wrf_name}_BT{suffix}"][record],
                        logical, side)
                    ref_endpoints = (
                        ref_value,
                        ref_value.astype(np.float64)
                        + ref_tendency.astype(np.float64) * duration,
                    )
                    cand_endpoints = (
                        cand_value,
                        cand_value.astype(np.float64)
                        + cand_tendency.astype(np.float64) * duration,
                    )
                    for endpoint, label in enumerate(("current", "future")):
                        ref_primitive = _uncouple_endpoint(
                            logical, side, ref_endpoints[endpoint],
                            ref_mu_endpoints[endpoint], **reference_coupling)
                        cand_primitive = _uncouple_endpoint(
                            logical, side, cand_endpoints[endpoint],
                            cand_mu_endpoints[endpoint], **candidate_coupling)
                        key = (f"record{record:03d}/{label}/{logical}/{side}")
                        fields[key] = _array_record(
                            ref_primitive, cand_primitive,
                            _boundary_rule(logical))
    status = (metadata["status"] == "PASS" and not missing
              and all(value["status"] == "PASS" for value in fields.values()))
    return {
        "status": "PASS" if status else "FAIL",
        "metadata": metadata,
        "missing_semantic_variables": missing,
        "boundary_durations_seconds": reference_durations,
        "failed_fields": [name for name, value in fields.items()
                          if value["status"] != "PASS"],
        "fields": fields,
    }


def compare_wrf_backend_directories(
        reference_directory: Path | str,
        candidate_directory: Path | str) -> dict[str, object]:
    """Recompute semantic parity for two direct native WRF directories."""
    reference_directory = Path(reference_directory)
    candidate_directory = Path(candidate_directory)
    required = ("wrfinput_d01", "wrfbdy_d01")
    missing = {
        "reference": [name for name in required
                      if not (reference_directory / name).is_file()],
        "candidate": [name for name in required
                      if not (candidate_directory / name).is_file()],
    }
    if missing["reference"] or missing["candidate"]:
        return {"schema": WRF_PARITY_SCHEMA, "status": "FAIL",
                "missing_files": missing, "files": {}}
    with (
        netCDF4.Dataset(reference_directory / "wrfinput_d01") as ref_input,
        netCDF4.Dataset(candidate_directory / "wrfinput_d01") as cand_input,
        netCDF4.Dataset(reference_directory / "wrfbdy_d01") as ref_bdy,
        netCDF4.Dataset(candidate_directory / "wrfbdy_d01") as cand_bdy,
    ):
        for dataset in (ref_input, cand_input, ref_bdy, cand_bdy):
            dataset.set_auto_mask(False)
        files = {
            "wrfinput_d01": _compare_wrfinput(ref_input, cand_input),
            "wrfbdy_d01": _compare_wrfbdy(
                ref_bdy, cand_bdy, ref_input, cand_input),
        }
    return {
        "schema": WRF_PARITY_SCHEMA,
        "status": "PASS" if all(
            value["status"] == "PASS" for value in files.values()) else "FAIL",
        "reference_directory": str(reference_directory),
        "candidate_directory": str(candidate_directory),
        "evidence": {
            "reference": {
                name: _file_identity(reference_directory / name)
                for name in required
            },
            "candidate": {
                name: _file_identity(candidate_directory / name)
                for name in required
            },
        },
        "rules": {
            "pressure_pa": asdict(PRESSURE_RULE),
            "geopotential_m2_s2": asdict(GEOPOTENTIAL_RULE),
            "wind_m_s": asdict(WIND_RULE),
            "temperature_k": asdict(TEMPERATURE_RULE),
            "moisture_kg_kg": asdict(MOISTURE_RULE),
            "specific_volume_m3_kg": asdict(SPECIFIC_VOLUME_RULE),
            "backend_inert": asdict(EXACT_RULE),
        },
        "files": files,
    }


__all__ = [
    "EXACT_RULE", "GEOPOTENTIAL_RULE", "MOISTURE_RULE", "PRESSURE_RULE",
    "SPECIFIC_VOLUME_RULE", "TEMPERATURE_RULE", "WIND_RULE",
    "WRF_PARITY_SCHEMA", "compare_wrf_backend_directories",
]
