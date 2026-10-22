"""Opt-in tuning members through the original hash-bound forecast runner.

Python selects and records existing interfaces. Surface and urban field
arithmetic runs on the GPU. A selector change needs its own preparation;
this adapter never replaces a prepared experiment or edits its receipts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace


SCHEMA = "gpuwm-tuning-member.v1"
SELECTORS = ("sf_surface_physics", "num_soil_layers", "bl_mynn_mixlength", "sf_urban_physics")
SURFACE = {"soil_moisture_scale": (0.65, 1.15, 1.0), "sst_offset_k": (-1.0, 1.0, 0.0)}
URBAN = {"fraction_scale": (0.5, 1.25, 1.0), "heat_capacity_scale": (0.7, 1.3, 1.0),
         "conductivity_scale": (0.7, 1.3, 1.0)}
_CLAIMED_MEMBER = None
_GPU_MODULE = None


class TuningMemberError(ValueError):
    """The declared setting cannot execute through its named interfaces."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _float32(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TuningMemberError(f"{name} must be a finite FP32 number")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error):
        raise TuningMemberError(f"{name} must be a finite FP32 number") from None
    if not math.isfinite(result):
        raise TuningMemberError(f"{name} must be a finite FP32 number")
    return result


def _controls(value, table, group):
    if not isinstance(value, dict) or set(value) - table.keys():
        raise TuningMemberError(f"{group} accepts only {sorted(table)}")
    result = {}
    for name, (low, high, default) in table.items():
        selected = _float32(value.get(name, default), f"{group}.{name}")
        if not _float32(low, name) <= selected <= _float32(high, name):
            raise TuningMemberError(f"{group}.{name} outside {low}..{high}")
        result[name] = selected
    return result


def validate_member(member):
    """Detach a declaration and validate it without importing CUDA code."""
    if not isinstance(member, dict):
        raise TuningMemberError("member must be an object")
    known = {"id", "seed", "physics", "surface", "urban", "physics_params_file", "product"}
    if set(member) - known:
        raise TuningMemberError(f"unknown member fields: {sorted(set(member) - known)}")
    identifier = member.get("id")
    if not isinstance(identifier, str) or not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier):
        raise TuningMemberError("member.id must be a nonempty path-safe label")
    seed = member.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
        raise TuningMemberError("member.seed must be an unsigned 64-bit integer")
    product = member.get("product")
    if product not in {"city", "research"}:
        raise TuningMemberError("tuning applies only to city or research products; HRRR-exact settings remain pinned")
    physics = member.get("physics")
    if not isinstance(physics, dict) or set(physics) != set(SELECTORS):
        raise TuningMemberError(f"member.physics must declare every selector: {list(SELECTORS)}")
    for key, allowed in (("sf_surface_physics", {2, 3, 4}), ("num_soil_layers", {4, 6, 9}),
                         ("bl_mynn_mixlength", {1, 2}), ("sf_urban_physics", {0, 1})):
        if isinstance(physics[key], bool) or not isinstance(physics[key], int) or physics[key] not in allowed:
            raise TuningMemberError(f"member.physics.{key} must be one of {sorted(allowed)}")
    if physics["sf_surface_physics"] == 3 and physics["sf_urban_physics"]:
        raise TuningMemberError("RUC never calls the urban canopy; sf_urban_physics must be 0 for RUC")
    if physics["sf_surface_physics"] in {2, 4} and physics["num_soil_layers"] != 4:
        raise TuningMemberError("Noah and Noah-MP use four prepared soil layers; a different ladder cannot reuse those inputs")
    surface = _controls(member.get("surface", {}), SURFACE, "surface")
    urban = _controls(member.get("urban", {}), URBAN, "urban")
    if not physics["sf_urban_physics"] and any(value != 1.0 for value in urban.values()):
        raise TuningMemberError("urban controls require sf_urban_physics=1; otherwise their recorded coefficients would never act")
    params = member.get("physics_params_file")
    if params is not None and (not isinstance(params, str) or not params.strip()):
        raise TuningMemberError("physics_params_file must name a set file")
    return {"id": identifier, "seed": seed, "product": product, "physics": dict(physics),
            "surface": surface, "urban": urban, **({} if params is None else {"physics_params_file": params})}


def _requires_surface(member):
    return any(value != SURFACE[key][2] for key, value in member["surface"].items())


def _requires_urban(member):
    return any(value != 1.0 for value in member["urban"].values())


def dependency_capabilities(member):
    """Check named seams before GPU acquisition; no numerical fallback."""
    member = validate_member(member)
    needed = {"gpuwm.ensemble.runtime_context": ("MemberOutputCapture", "member_output_scope")}
    if _requires_surface(member):
        needed["gpuwm.ensemble.surface_recipe"] = ("surface_initialization_callback",)
    if member.get("physics_params_file"):
        needed["gpuwm.physics_params"] = ("environment_set", "document", "check_schemes")
    if _requires_urban(member):
        needed["gpuwm.core.urban_ucm"] = ("_packed", "_device_params", "UCM_TABLE_COLUMNS")
    result = {}
    for name, attributes in needed.items():
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            raise TuningMemberError(f"missing {name}: the declared tuning control has no runtime interface ({error})") from None
        missing = [attribute for attribute in attributes if not hasattr(module, attribute)]
        if missing:
            raise TuningMemberError(f"missing {name} interfaces {missing}: the declared tuning control would not reach the forecast")
        result[name] = {"interfaces": list(attributes), "source_sha256": _sha_file(module.__file__)}
    return result


@contextmanager
def parameter_environment(member):
    """Set before experiment parsing; clear inherited sets for controls."""
    variable = "GPUWM_PHYSICS_PARAMS"
    prior = os.environ.get(variable)
    path = member.get("physics_params_file")
    if path:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise TuningMemberError("physics_params_file does not exist; using defaults would mislabel this member")
        os.environ[variable] = str(resolved)
    else:
        os.environ.pop(variable, None)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = prior


def preflight_member(member, inputs, *, check_dependencies=True):
    """Require original preparation selectors; never rebind loaded inputs."""
    member = validate_member(member)
    assertion = getattr(inputs, "physics_profile_assertion", None)
    if assertion:
        raise TuningMemberError("a named source-exact physics profile cannot be tuned; prepare an explicit city or research configuration")
    experiment = getattr(inputs, "experiment", None)
    if _requires_urban(member) and float(getattr(experiment, "restart_interval_s", 0.0)) != 0.0:
        raise TuningMemberError("temporary urban GPU settings are not in checkpoint identity; disable restart output to prevent a tuned checkpoint resuming with the base table")
    domains = getattr(experiment, "domains", ())
    if not domains:
        raise TuningMemberError("member inputs carry no original experiment domains")
    rows = []
    for domain in domains:
        run = domain.run
        actual = {name: getattr(run, name, None) for name in SELECTORS}
        if actual != member["physics"]:
            raise TuningMemberError(f"domain {domain.grid_id} prepared selectors {actual} differ from member selectors {member['physics']}; prepare this selector group from the shared source instead of replacing hash-bound inputs")
        if int(getattr(run, "bl_pbl_physics", -1)) != 5:
            raise TuningMemberError("MYNN mixing-length tuning requires bl_pbl_physics=5; another PBL would ignore the setting")
        rows.append({"grid_id": int(domain.grid_id), "physics": actual})
    params = getattr(experiment, "physics_params", None)
    parameter_record = None
    if member.get("physics_params_file"):
        from gpuwm.physics_params import document, environment_set
        expected = environment_set()
        if expected is None or params is None or expected.sha256() != params.sha256():
            raise TuningMemberError("the parameter set was not bound before preparation preflight; do not attach a set to already-loaded inputs")
        parameter_record = document(params)
    elif params is not None:
        raise TuningMemberError("the control unexpectedly carries a physics parameter set")
    return {"member": member, "domains": rows, "physics_params": parameter_record,
            "dependencies": dependency_capabilities(member) if check_dependencies else {}}


_URBAN_SOURCE = r'''
extern "C" __global__ void urban_tuning_validate(const float *fraction,
    const int *mask, unsigned long long count, int *bad) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && mask[i] && (!isfinite(fraction[i]) || fraction[i] < 0.0f || fraction[i] > 1.0f)) atomicOr(bad, 1);
}
extern "C" __global__ void urban_tuning_fraction(float *fraction,
    const int *mask, unsigned long long count, float scale) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && mask[i]) fraction[i] = fminf(1.0f, fmaxf(0.0f, __fmul_rn(fraction[i], scale)));
}
extern "C" __global__ void urban_tuning_table(float *table,
    unsigned long long count, int width, int c0, int c1, int c2,
    int k0, int k1, int k2, float capacity, float conductivity) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int column = i % width;
    if (column == c0 || column == c1 || column == c2) table[i] = __fmul_rn(table[i], capacity);
    if (column == k0 || column == k1 || column == k2) table[i] = __fmul_rn(table[i], conductivity);
}
'''


def _device_sha(array):
    return hashlib.sha256(array.get().tobytes()).hexdigest()


def _apply_urban(state, member, domain_id):
    """Clone each domain's pinned table binding, then edit only GPU words."""
    global _GPU_MODULE
    from gpuwm.core import urban_ucm
    import cupy as cp
    import numpy as np

    identity = hashlib.sha256(_canonical(member).encode()).hexdigest()
    prior = getattr(state, "_tuning_urban", None)
    if prior is not None:
        if prior["identity"] != identity:
            raise TuningMemberError("initialized urban fields already belong to a different tuning member")
        return prior["receipt"]
    if float(getattr(state, "elapsed_seconds", 0.0)) > float(getattr(state, "domain_start_offset", 0.0)):
        raise TuningMemberError("urban tuning requires a fresh initialized domain; changing an advanced state would corrupt continuation")
    urban = getattr(state.physics, "urban", None)
    if urban is None or int(urban.option) != 1:
        raise TuningMemberError("urban GPU controls require an initialized single-layer UCM binding")
    for attribute in ("_DEVICE_PARAMS", "_PACKED"):
        if not hasattr(urban_ucm, attribute):
            raise TuningMemberError("the UCM table cache interface changed; private member table words cannot be bound safely")
    fraction = urban.frc_urb2d
    mask = cp.ascontiguousarray(urban.urban_mask, dtype=cp.int32)
    if str(fraction.dtype) != "float32" or fraction.shape != mask.shape or not fraction.flags.c_contiguous:
        raise TuningMemberError("urban fraction must be contiguous FP32 over the initialized urban mask")
    if _GPU_MODULE is None:
        from gpuwm.certify.kernel_manifest import record_module
        from gpuwm.core.kernels import MODULE_KEY_ROOT
        module = cp.RawModule(code=_URBAN_SOURCE, options=("--std=c++11",), name_expressions=(
            "urban_tuning_validate", "urban_tuning_fraction", "urban_tuning_table"))
        record_module(f"{MODULE_KEY_ROOT}:tuning_urban", source=_URBAN_SOURCE, options=("--std=c++11",),
                      module=module)
        _GPU_MODULE = module
    bad = cp.zeros(1, dtype=cp.int32)
    blocks = (int(fraction.size) + 255) // 256
    _GPU_MODULE.get_function("urban_tuning_validate")((blocks,), (256,),
        (fraction, mask, np.uint64(fraction.size), bad))
    if int(bad.get()[0]):
        raise TuningMemberError("initial urban fraction is nonfinite or outside 0..1; scaling would corrupt the canopy partition")
    original_params = urban.params
    source = urban_ucm._packed(original_params)
    source_buffers = urban_ucm._device_params(source)
    private_params = replace(original_params)
    private_packed = replace(source)
    buffers = tuple(array.copy() for array in source_buffers)
    table = buffers[0]
    columns = urban_ucm.UCM_TABLE_COLUMNS
    capacity_columns = [columns.index(name) for name in ("capr", "capb", "capg")]
    conductivity_columns = [columns.index(name) for name in ("aksr", "aksb", "aksg")]
    before = {"fraction_sha256": _device_sha(fraction), "table_sha256": _device_sha(table)}
    controls = member["urban"]
    _GPU_MODULE.get_function("urban_tuning_table")(((int(table.size) + 255) // 256,), (256,),
        (table, np.uint64(table.size), np.int32(len(columns)),
         *(np.int32(index) for index in capacity_columns + conductivity_columns),
         np.float32(controls["heat_capacity_scale"]), np.float32(controls["conductivity_scale"])))
    _GPU_MODULE.get_function("urban_tuning_fraction")((blocks,), (256,),
        (fraction, mask, np.uint64(fraction.size), np.float32(controls["fraction_scale"])))
    after = {"fraction_sha256": _device_sha(fraction), "table_sha256": _device_sha(table)}
    # The original loader verified table hashes. Separate parameter identities
    # keep another domain's cache and the untouched original buffers intact.
    urban_ucm._PACKED[id(private_params)] = (private_params, private_packed)
    urban_ucm._DEVICE_PARAMS[id(private_packed)] = (private_packed, buffers)
    urban.params = private_params
    receipt = {"domain_id": domain_id, "controls": controls, "before": before, "after": after,
               "base_table_sha256": original_params.sha256, "fraction_mask": "initialized urban mask",
               "fraction_cap": "0..1", "capacity_columns": ["capr", "capb", "capg"],
               "conductivity_columns": ["aksr", "aksb", "aksg"],
               "kernel_source_sha256": hashlib.sha256(_URBAN_SOURCE.encode()).hexdigest()}
    state._tuning_urban = {"identity": identity, "receipt": receipt}
    return receipt


def initialization_callback(member, *, record):
    """Compose the dependency's surface hook with the temporary UCM seam."""
    member = validate_member(member)
    surface = None
    if _requires_surface(member):
        from gpuwm.ensemble.surface_recipe import surface_initialization_callback
        surface = surface_initialization_callback(
            SimpleNamespace(perturbation={"kind": "surface-state", **member["surface"]}),
            member_id=member["id"], seed=member["seed"],
            record=lambda receipt: record({"surface": receipt}))
    if not _requires_urban(member):
        return surface

    def initialize(*, model=None, prepared_case=None, state=None, cfg=None, grid=None, clock=None):
        if surface is not None:
            surface(model=model, prepared_case=prepared_case, state=state, cfg=cfg, grid=grid, clock=clock)
        nodes = ([(getattr(node.cfg, "grid_id", None), node.state)
                  for node in model.walk_parent_first() if node.state is not None and getattr(node, "_started", True)]
                 if model is not None else [(getattr(cfg, "grid_id", None), state)])
        if any(domain_state is None for _, domain_state in nodes):
            raise TuningMemberError("urban initialization seam supplied no initialized domain")
        record({"urban": [_apply_urban(domain_state, member, domain_id) for domain_id, domain_state in nodes]})
    return initialize


def run_member(member, inputs, *, output_directory, receipt_path, runner=None):
    """Run one declared member per process, preserving ordinary histories."""
    global _CLAIMED_MEMBER
    member = validate_member(member)
    if _CLAIMED_MEMBER is not None:
        raise TuningMemberError("each tuning member requires a fresh process; compiled constants and table caches cannot switch members")
    preflight = preflight_member(member, inputs)
    _CLAIMED_MEMBER = member["id"]
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree, claim_output_directory
    output = claim_output_directory(Path(output_directory), protected_roots=(inputs.prepared_root, inputs.experiment_config))
    receipts = []
    capture = MemberOutputCapture(callback=lambda **event: None, member_id=0, keep_member_files=True,
                                  initialize_callback=initialization_callback(member, record=receipts.append))
    report = {"schema": SCHEMA, "status": "RUNNING", "preflight": preflight,
              "output_directory": str(output), "initialization": receipts,
              "execution": "ordinary hash-bound prepared tree; one process per setting",
              "temporary_adapter": "UCM fraction and private GPU table scaling" if _requires_urban(member) else None}
    _atomic_json(receipt_path, report)
    try:
        with member_output_scope(capture):
            result = (runner or run_prepared_tree)(inputs, output_directory=output, io_mode="history")
        if _requires_surface(member) and not any("surface" in item for item in receipts):
            raise TuningMemberError("the forecast runner did not call the surface initialization seam; the recorded setting never acted")
        if _requires_urban(member) and not any("urban" in item for item in receipts):
            raise TuningMemberError("the forecast runner did not call the urban initialization seam; the recorded setting never acted")
        if isinstance(result, dict) and result.get("status") not in {None, "PASS", "SUCCESS", "SUCCEEDED", "COMPLETE"}:
            raise TuningMemberError(f"forecast runner returned {result.get('status')}")
        report.update(status="PASS", runner_result=result)
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        _atomic_json(receipt_path, report)
        raise
    _atomic_json(receipt_path, report)
    return report


def preflight_job(job):
    """Open the original preparation through its actual CPU-only reader."""
    member = validate_member(job["member"])
    dependency_capabilities(member)
    prepared = dict(job["prepared"])
    allowed = {"prepared_root", "experiment_config", "experiment_config_sha256", "preparation_receipt_sha256", "prepared_head_sha256"}
    if set(prepared) - allowed:
        raise TuningMemberError(f"unsupported preparation arguments: {sorted(set(prepared) - allowed)}")
    for name in ("prepared_root", "experiment_config"):
        prepared[name] = Path(prepared[name])
    from gpuwm.prepared_domain_tree_forecast import preflight_prepared_tree
    inputs = preflight_prepared_tree(**prepared)
    return inputs, preflight_member(member, inputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    receipt_path = job.get("receipt_path", str(args.job.with_suffix(".receipt.json")))
    attempted = False
    try:
        member = validate_member(job["member"])
        with parameter_environment(member):
            inputs, preflight = preflight_job(job)
            if args.preflight_only:
                _atomic_json(receipt_path, {"schema": SCHEMA, "status": "PREFLIGHT_PASS", "preflight": preflight, "gpu_forecast_run": False})
            else:
                attempted = True
                run_member(member, inputs, output_directory=job["output_directory"], receipt_path=receipt_path)
    except Exception as error:
        existing = None
        if attempted and Path(receipt_path).is_file():
            existing = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("status") != "FAIL":
            _atomic_json(receipt_path, {"schema": SCHEMA, "status": "FAIL", "error": f"{type(error).__name__}: {error}", "gpu_forecast_run": attempted})
        print(f"tuning member: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
