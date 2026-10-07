"""One ordinary RTE+RRTMGP call over member-owned radiation columns.

Column joining does not construct a dycore grid. The caller owns and prices
the packed atmosphere, surface and microphysics arrays. This binding owns
latitude/longitude banks, one stream, and the original native workspace
policy. Admission also reserves full-column and selected solver transients.
"""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from math import prod
from operator import index
import inspect
import json
import struct
import threading
from types import SimpleNamespace

import numpy as np

from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage
from gpuwm.ensemble.batch_state import _temporal_key


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    result = index(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _scalar(value):
    """Strict metadata identity, preserving clock types and float words."""
    calendar = _temporal_key(value)
    if calendar is not None:
        return {"calendar": calendar}
    if isinstance(value, np.datetime64):
        return {"numpy_calendar": {"dtype": value.dtype.str, "words": value.tobytes().hex()}}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return {"float64": struct.pack("!d", value).hex()}
    if isinstance(value, dict):
        return {str(key): _scalar(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_scalar(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return {"ratio": [value.numerator, value.denominator]}
    raise TypeError(f"RRTMGP member metadata cannot encode {type(value).__name__}")


def _config_identity(cfg):
    raw = asdict(cfg) if is_dataclass(cfg) else vars(cfg)
    return _scalar({key: value for key, value in raw.items() if not key.startswith("_")})


def _adapter_metadata(adapter):
    workspace = getattr(adapter, "chunk_workspace", None)
    workspace_policy = None if workspace is None else {
        "class": f"{type(workspace).__module__}.{type(workspace).__qualname__}",
        "nz": int(workspace.nz), "column_chunk": int(workspace.column_chunk),
        "p_top": _scalar(float(workspace.p_top))}
    return {"start_time": adapter.start_time.isoformat(),
            "spectra": [bool(adapter.longwave), bool(adapter.shortwave)],
            "column_chunk": int(adapter.column_chunk),
            "validation_mode": adapter.validation_mode,
            "trace_gas_overrides": _scalar(adapter.trace_gas_overrides),
            "trace_vmr": _scalar(adapter.trace_vmr),
            "surface_diffuse_requested": bool(getattr(adapter, "surface_diffuse_requested", False)),
            "workspace_policy": workspace_policy}


def _binding_metadata(metadata):
    """Only immutable workspace, solar and column-policy authorities.

    The current complete configuration is still compared across members on
    every call. Adaptive ``dt`` and sound-step counts are live clock outputs,
    so their common changes cannot freeze the first step's radiation binding.
    Fixed-clock values remain part of the immutable binding identity.
    """
    config = dict(metadata["config"])
    if config.get("use_adaptive_time_step") is True:
        from gpuwm.core.adaptive_clock import ADAPTIVE_DERIVED_RUN_FIELDS
        config = {name: value for name, value in config.items()
                  if name not in ADAPTIVE_DERIVED_RUN_FIELDS}
    return {"adapter": metadata["adapter"], "config": config,
            "p_top": _scalar(float(metadata["p_top"]))}


def validate_rrtmgp_binding_metadata(metadata, bound_metadata):
    """Reject a stale binding without pinning a synchronized adaptive step."""
    if _binding_metadata(metadata) != _binding_metadata(bound_metadata):
        raise ValueError(
            "RRTMGP immutable adapter/config/top authority changed after column "
            "binding; its workspace, solar geometry or column policy would be stale")


def validate_rrtmgp_member_metadata(adapters, states, configs, *, packed_state):
    """Refuse clock, cap, config and native-policy differences before writes."""
    adapters, states, configs = tuple(adapters), tuple(states), tuple(configs)
    if not adapters or len(states) != len(adapters) or len(configs) != len(adapters):
        raise ValueError("RRTMGP needs one adapter, clock/state and configuration per member")
    first_adapter, first_cfg = _adapter_metadata(adapters[0]), _config_identity(configs[0])
    first_clock = _scalar(float(states[0].elapsed_seconds))
    if not hasattr(states[0], "p_top"):
        raise ValueError("RRTMGP member columns need their declared scalar p_top before above-model packing")
    first_top = _scalar(float(states[0].p_top))
    first_updates = int(getattr(getattr(states[0], "physics", None), "microphysics_updates", 0))
    if not np.isfinite(float(states[0].elapsed_seconds)) or not np.isfinite(float(states[0].p_top)):
        raise ValueError("RRTMGP member clock and scalar top must be finite")
    for member, (adapter, state, cfg) in enumerate(zip(adapters, states, configs)):
        if _adapter_metadata(adapter) != first_adapter:
            raise ValueError(f"RRTMGP member {member} has different spectra, trace, validation or solar-date metadata; one native call would apply another member's policy")
        if _config_identity(cfg) != first_cfg:
            raise ValueError(f"RRTMGP member {member} configuration differs; one native call would change its cadence or cloud coupling")
        if _scalar(float(state.elapsed_seconds)) != first_clock:
            raise ValueError(f"RRTMGP member {member} clock differs; one solar geometry call would radiate it at the wrong time")
        if not hasattr(state, "p_top") or _scalar(float(state.p_top)) != first_top:
            raise ValueError(f"RRTMGP member {member} scalar p_top differs; one above-model cap would change its column")
        if int(getattr(getattr(state, "physics", None), "microphysics_updates", 0)) != first_updates:
            raise ValueError(f"RRTMGP member {member} microphysics update metadata differs; one effective-radius selection would use the wrong state")
    if (_scalar(float(packed_state.elapsed_seconds)) != first_clock
            or not hasattr(packed_state, "p_top") or _scalar(float(packed_state.p_top)) != first_top):
        raise ValueError("RRTMGP packed clock/top differ from the member authority before radiation")
    if int(getattr(getattr(packed_state, "physics", None), "microphysics_updates", 0)) != first_updates:
        raise ValueError("RRTMGP packed microphysics update metadata would select the wrong effective-radius branch")
    return {"adapter": first_adapter, "config": first_cfg,
            "elapsed_seconds": float(states[0].elapsed_seconds),
            "p_top": float(states[0].p_top), "microphysics_updates": first_updates}


def rrtmgp_geometry_plan(*, ny, nx, reserved_bytes=0):
    shape = (_positive(ny, "ny"), _positive(nx, "nx"))
    return BatchMemoryPlan(tuple(BatchArraySpec(f"rrtmgp:{name}", shape, "member")
        for name in ("latitude", "longitude")), reserved_bytes)


def _shape_bytes(shapes):
    return sum(prod(shape) * itemsize for shape, itemsize in shapes.values())


def _workspace_less_chunks(*, nz, column_chunk, p_top, metadata):
    """Named unshared chunk buffers retained by the ordinary native branch.

    The union deliberately reserves both sequential spectra. It is a call
    envelope, rather than a claim that all listed buffers coexist or persist.
    """
    from gpuwm.core.rrtmgp import rrtmgp_above_model_layer_counts
    lw_upper, sw_upper = rrtmgp_above_model_layer_counts(p_top)
    chunks = {}
    for kind, upper in (("lw", lw_upper), ("sw", sw_upper)):
        nlay, c = nz + upper, column_chunk
        gpt, bands, gases = (int(metadata[f"{key}_{kind}"]) for key in ("ngpt", "nband", "ngas"))
        layers, cube = (c, nlay), (c, nlay, gpt)
        chunks[f"{kind}/vmr"] = ((c, nlay, gases + 1), 4)
        chunks[f"{kind}/gas_col_dry"] = (layers, 4)
        chunks[f"{kind}/gas_tau"] = (cube, 4)
        chunks[f"{kind}/mcica_mask"] = (cube, 1)
        for name in ("tau", "ssa", "asy"):
            chunks[f"{kind}/cloud_{name}"] = ((c, nlay, bands), 4)
        chunks[f"{kind}/finalized_tau"] = (cube, 4)
        for name in ("flux_up", "flux_dn"):
            chunks[f"{kind}/{name}"] = ((c, nlay + 1), 4)
        if kind == "lw":
            chunks.update({"lw/planck_lay": (cube, 4),
                "lw/planck_lev": ((c, nlay + 1, gpt), 4),
                "lw/planck_sfc": ((c, gpt), 4), "lw/emiss_gpt": ((c, gpt), 4),
                "lw/incident": ((c, gpt), 4)})
        else:
            chunks.update({"sw/gas_ssa": (cube, 4), "sw/finalized_ssa": (cube, 4),
                "sw/finalized_asy": (cube, 4), "sw/albedo_gpt": ((c, gpt), 4),
                "sw/incidence_gpt": ((c, gpt), 4), "sw/mu0": (layers, 4),
                "sw/flux_dir": ((c, nlay + 1), 4)})
    return chunks


def rrtmgp_call_memory_inventory(cfg, *, members, ny, nx, p_top, column_chunk,
                               native_workspace=True):
    """Official native workspace and transient formulas for joined columns."""
    from gpuwm.core.preflight import rrtmgp_column_shapes, rrtmgp_workspace_shapes
    members, ny, nx, column_chunk = (_positive(value, name) for value, name in
        ((members, "members"), (ny, "ny"), (nx, "nx"), (column_chunk, "column_chunk")))
    # Only the estimator's column count changes. The native call retains the
    # real per-domain configuration and never receives this counting view.
    raw = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    count_cfg = SimpleNamespace(**{**raw, "ny": members * ny, "nx": nx})
    columns = rrtmgp_column_shapes(count_cfg, p_top, column_chunk=column_chunk)
    if not columns:
        raise ValueError("RRTMGP column plan resolved no modern radiation transients for the selected configuration")
    workspace = rrtmgp_workspace_shapes(count_cfg.nz, column_chunk, p_top) if native_workspace else {}
    unshared = {}
    if not native_workspace:
        from gpuwm.core.preflight import _gas_table_meta
        unshared = _workspace_less_chunks(nz=count_cfg.nz,
            column_chunk=min(column_chunk, members * ny * nx), p_top=p_top,
            metadata=_gas_table_meta())
    cells, nz = members * ny * nx, int(count_cfg.nz)
    # The native column inventory excludes the returned carrier and these
    # optional full-width branch buffers. Declare them separately rather
    # than hiding them in a per-member multiplier.
    extra = {"returned/rthratenlw": ((nz, cells), 4),
             "returned/rthratensw": ((nz, cells), 4),
             "returned/cldfra": ((nz, cells), 4),
             **{f"returned/{name}": ((cells,), 4) for name in
                ("swdown", "glw", "gsw", "coszen", "olr", "swddir", "swddif")},
             "shortwave/daylight_indices": ((cells,), 8),
             "shortwave/direct_surface": ((cells,), 4),
             "shortwave/scatter_up": ((min(cells, column_chunk), nz + 1), 4),
             "shortwave/scatter_down": ((min(cells, column_chunk), nz + 1), 4)}
    from gpuwm.core.mynn_radiation import mynn_bl_cloud_active
    if mynn_bl_cloud_active(getattr(cfg, "bl_pbl_physics", 0), getattr(cfg, "icloud_bl", 0)):
        extra.update({f"mynn/{name}": ((cells, nz), 4) for name in ("qc_bl", "qi_bl", "cldfra_bl")})
        extra.update({f"mynn/{name}": ((cells, nz), 1) for name in ("supplied_liquid", "supplied_ice")})
    return {"geometry": rrtmgp_geometry_plan(ny=ny, nx=nx).inventory(members),
            "column_transients": columns, "column_transient_bytes": _shape_bytes(columns),
            "additional_call_buffers": extra, "additional_call_buffer_bytes": _shape_bytes(extra),
            "solver_workspace": workspace, "solver_workspace_bytes": _shape_bytes(workspace),
            "workspace_less_chunk_envelope": unshared, "workspace_less_chunk_envelope_bytes": _shape_bytes(unshared),
            "ownership": ("one stream-owned chunk workspace; member-owned geometry; native call transients"
                          if native_workspace else
                          "one stream-owned workspace-less native call; member-owned geometry; named transient envelope"),
            "qualification": "native preflight basis plus named branch/result buffers; full forecast pool peak remains a graph gate"}


def _resident(array, xp, *, shape, device, name):
    if (not isinstance(array, xp.ndarray) or array.shape != shape
            or array.dtype != np.dtype("float32") or not array.flags.c_contiguous
            or int(array.device.id) != device):
        raise ValueError(f"RRTMGP {name} needs resident contiguous float32 {shape} on device {device}")


class PackedRRTMGPColumns:
    """A private-stream all-member adapter with explicit metadata authorities."""

    def __call__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("RRTMGP packed column owner is closed")
            xp = self._xp
            if int(xp.cuda.runtime.getDevice()) != self.device:
                raise RuntimeError("RRTMGP packed radiation uses a different current device")
            if self.reuse_scope is not None:
                self.reuse_scope.require("radiation", adapters=self.member_adapters)
            metadata = validate_rrtmgp_member_metadata(self.member_adapters, self.member_states,
                self.member_configs, packed_state=self.state)
            validate_rrtmgp_binding_metadata(metadata, self._metadata)
            self._validate_table_owners()
            current = xp.cuda.get_current_stream()
            self._input_ready.record(current)
            self.stream.wait_event(self._input_ready)
            self.adapter.update_count = self.member_adapters[0].update_count
            self.receipt["native_calls"] += 1
            with self.stream:
                result = self.adapter(atmosphere=self.atmosphere, fields=self.fields,
                                      state=self.state, cfg=self.member_configs[0])
                self._output_ready.record(self.stream)
            current.wait_event(self._output_ready)
            for adapter in self.member_adapters:
                adapter.update_count += 1
            self.receipt["successful_calls"] += 1
            self.receipt["member_update_counts"] = [int(adapter.update_count) for adapter in self.member_adapters]
            self.receipt["last_elapsed_seconds"] = metadata["elapsed_seconds"]
            self.receipt["last_config"] = metadata["config"]
            from gpuwm.core.rrtmgp import _CHUNK_SCRATCH
            self.receipt["stream_scratch_allocations"] = [
                {"name": key[2], "shape": list(array.shape), "dtype": array.dtype.str,
                 "payload_bytes": int(array.nbytes)}
                for key, array in _CHUNK_SCRATCH.items()
                if key[0] == ("cupy", self.device) and key[1] == int(self.stream.ptr)]
            from gpuwm.certify.kernel_manifest import kernel_manifest
            self.receipt["compiled_kernel_manifest"] = {
                key: value for key, value in kernel_manifest().items() if "rrtmgp" in key}
            return result

    def _validate_table_owners(self):
        for member, adapter in enumerate(self.member_adapters):
            for name, owners in self._table_owners.items():
                if getattr(adapter, name) is not owners[member]:
                    raise ValueError(f"RRTMGP member {member} {name} table owner changed after binding; the packed coefficient geometry would be stale")
            for name, owners in self._setup_owners.items():
                if getattr(adapter, name, None) is not owners[member]:
                    raise ValueError(f"RRTMGP member {member} {name} setup owner changed after binding; the packed solar or ozone geometry would be stale")

    def close(self):
        """Release only this binding's stream scratch after its last call."""
        with self._lock:
            if self._closed:
                return dict(self._released)
            from gpuwm.core.rrtmgp import release_rrtmgp_stream_scratch
            self._released = release_rrtmgp_stream_scratch(device_id=self.device, stream=self.stream)
            self._closed = True
            self.adapter.chunk_workspace = None
            self.adapter.latitude_deg = None
            self.adapter.longitude_deg = None
            self.workspace = None
            self.storage = None
            self.receipt["released"] = dict(self._released)
            return dict(self._released)


def prepare_rrtmgp_column_batch(adapters, *, member_states, member_configs,
                                atmosphere, fields, packed_state, members, ny, nx,
                                available_bytes, reserved_bytes=0, array_module=None,
                                borrow_workspace=False, reuse_scope=None):
    """Clone native metadata, bind private geometry, then call all members once.

    Original adapters provide clock/config/table authority and carried counters;
    no member adapter is called to calculate a forecast. The packed state owns
    all transported masses, scheme radii and optional number moments. Returned
    heating and surface fluxes retain the ordinary native carrier layout.
    """
    if array_module is None:
        import cupy as array_module
    from gpuwm.core.rrtmgp import RRTMGPRadiation, cloud_optics_scheme
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace
    from gpuwm.io.restart import _array_setup_identity, _resolved_object_setup_identity
    members, ny, nx = (_positive(value, name) for value, name in
                       ((members, "members"), (ny, "ny"), (nx, "nx")))
    adapters, member_states, member_configs = tuple(adapters), tuple(member_states), tuple(member_configs)
    if len(adapters) != members or any(type(adapter) is not RRTMGPRadiation for adapter in adapters):
        raise TypeError("RRTMGP batch needs each member's initialized native RRTMGPRadiation adapter")
    if len({id(adapter) for adapter in adapters}) != members:
        raise ValueError("RRTMGP members share a mutable update-count owner; one packed call would increment that member's counter twice")
    metadata = validate_rrtmgp_member_metadata(adapters, member_states, member_configs, packed_state=packed_state)
    first = adapters[0]
    table_names = ("lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables")
    identities = {}
    for name in table_names:
        owner = getattr(first, name)
        # Native coefficient loaders cache one immutable host table owner.
        # A different object may carry different coefficients or geometry;
        # compare its actual authoritative host content before accepting it.
        identity = _resolved_object_setup_identity(owner, name)
        identities[name] = identity["sha256"]
        for member, adapter in enumerate(adapters[1:], 1):
            other = getattr(adapter, name)
            if other is not owner and _resolved_object_setup_identity(other, name)["sha256"] != identity["sha256"]:
                raise ValueError(f"RRTMGP member {member} {name} coefficients or table geometry differ")
    ozone_identity = {name: _array_setup_identity(getattr(first, name))
                      for name in ("_ozone_logp", "_ozone_vmr")}
    for member, adapter in enumerate(adapters[1:], 1):
        for name, identity in ozone_identity.items():
            if _array_setup_identity(getattr(adapter, name)) != identity:
                raise ValueError(f"RRTMGP member {member} ozone climatology differs; the packed gas profile would use another member's values")
    device = int(array_module.cuda.runtime.getDevice())
    nz = atmosphere["pressure"].shape[0]
    if any(int(cfg.nz) != nz or int(cfg.ny) != ny or int(cfg.nx) != nx for cfg in member_configs):
        raise ValueError("RRTMGP member configuration shape differs from the declared packed column layout")
    for name in ("pressure", "temperature", "exner", "qv", "qc", "qi"):
        _resident(atmosphere[name], array_module, shape=(nz, members * ny, nx), device=device, name=f"atmosphere:{name}")
    _resident(atmosphere["p_interface"], array_module, shape=(nz + 1, members * ny, nx), device=device, name="atmosphere:p_interface")
    for name in ("tsk", "albedo"):
        _resident(fields[name], array_module, shape=(members * ny, nx), device=device, name=f"fields:{name}")
    # Band emissivity is also accepted by the ordinary adapter. Its trailing
    # axes are the packed surface, while its leading axis is table bands.
    emiss = fields["emiss"]
    shape = (members * ny, nx) if emiss.ndim == 2 else (first.lw_tables.nband, members * ny, nx)
    _resident(emiss, array_module, shape=shape, device=device, name="fields:emiss")
    if not first.longwave:
        _resident(fields["glw"], array_module, shape=(members * ny, nx), device=device, name="fields:glw")
    scheme = cloud_optics_scheme(int(member_configs[0].mp_physics))
    state_fields = {"qc", "qr", "qi", "qs", "effc", "effr", "effi", "effs", "nc", "nr", "ni", "ns", "qnc", "qnr", "qni", "qns"}
    for name in state_fields:
        value = getattr(packed_state, name, None)
        if value is not None:
            _resident(value, array_module, shape=(nz, members * ny, nx), device=device, name=f"state:{name}")
    from gpuwm.core.mynn_radiation import mynn_bl_cloud_active
    if mynn_bl_cloud_active(getattr(member_configs[0], "bl_pbl_physics", 0), getattr(member_configs[0], "icloud_bl", 0)):
        for name in ("qc_bl", "qi_bl", "cldfra_bl"):
            _resident(fields[name], array_module, shape=(nz, members * ny, nx), device=device, name=f"fields:{name}")
    if scheme in ("wsm6", "thompson", "nssl", "p3"):
        _resident(fields["xland"], array_module, shape=(members * ny, nx), device=device, name="fields:xland")
    for member, adapter in enumerate(adapters):
        for name in ("latitude_deg", "longitude_deg"):
            _resident(getattr(adapter, name), array_module, shape=(ny, nx), device=device, name=f"member {member}:{name}")
    chunk = _positive(first.column_chunk, "column_chunk")
    native_workspace = getattr(first, "chunk_workspace", None) is not None
    if type(borrow_workspace) is not bool:
        raise TypeError("RRTMGP workspace borrowing needs an explicit boolean ownership declaration")
    if borrow_workspace:
        # A complete original-driver rendezvous owns this reuse. It joins
        # every queued native call before an original fallback can enter the
        # same member's workspace; the ordinary adapter keeps its owner.
        from gpuwm.ensemble.packed_production_physics import OriginalWorkspaceReuseScope
        if type(reuse_scope) is not OriginalWorkspaceReuseScope:
            raise TypeError("radiation workspace reuse needs the original all-member driver rendezvous owner")
        reuse_scope.require("radiation", adapters=adapters)
        workspace = first.chunk_workspace
        if not native_workspace or type(workspace) is not SharedRRTMGPChunkWorkspace:
            raise TypeError("borrowed radiation workspace needs the actual original modern RRTMGP allocation")
        if (workspace.nz != nz or workspace.column_chunk != chunk
                or _scalar(workspace.p_top) != _scalar(metadata["p_top"])
                or int(workspace.storage.device.id) != device):
            raise ValueError("original RRTMGP workspace layout or device differs from the native column policy")
    memory = rrtmgp_call_memory_inventory(member_configs[0], members=members, ny=ny, nx=nx,
        p_top=metadata["p_top"], column_chunk=chunk, native_workspace=native_workspace)
    # Tables are immutable shared inputs. If their device mirror is absent,
    # binding accounts for and establishes that one upload before calls.
    upload_bytes = 0
    for name in table_names:
        table = getattr(first, name)
        if device not in table._device:
            for value in vars(table).values():
                if isinstance(value, np.ndarray):
                    itemsize = 1 if value.dtype == np.dtype("bool") else 4
                    upload_bytes += ((int(value.size) * itemsize + 511) // 512) * 512
    memory["new_shared_table_upload_bytes"] = upload_bytes
    if isinstance(reserved_bytes, (bool, np.bool_)) or index(reserved_bytes) < 0:
        raise ValueError("RRTMGP reserved bytes must be a nonnegative integer")
    reserve = (memory["column_transient_bytes"] + memory["additional_call_buffer_bytes"]
               + memory["workspace_less_chunk_envelope_bytes"] + upload_bytes
               + (0 if borrow_workspace else ((memory["solver_workspace_bytes"] + 511) // 512) * 512)
               + index(reserved_bytes))
    storage = BatchStorage(rrtmgp_geometry_plan(ny=ny, nx=nx, reserved_bytes=reserve),
                           members, array_module=array_module, available_bytes=available_bytes)
    for name, attribute in (("latitude", "latitude_deg"), ("longitude", "longitude_deg")):
        for member, adapter in enumerate(adapters):
            array_module.copyto(storage.arrays[f"rrtmgp:{name}"][member], getattr(adapter, attribute))
    for name in table_names:
        getattr(first, name).to_device()
    clone = copy(first)
    clone.latitude_deg = storage.arrays["rrtmgp:latitude"].reshape(members * ny, nx)
    clone.longitude_deg = storage.arrays["rrtmgp:longitude"].reshape(members * ny, nx)
    clone.trace_vmr = dict(first.trace_vmr)
    clone.trace_gas_overrides = None if first.trace_gas_overrides is None else dict(first.trace_gas_overrides)
    if not borrow_workspace:
        workspace = (SharedRRTMGPChunkWorkspace(nz, chunk, metadata["p_top"], _array_module=array_module)
                     if native_workspace else None)
    clone.chunk_workspace = workspace
    result = PackedRRTMGPColumns()
    result.adapter, result.workspace, result.storage = clone, workspace, storage
    result.member_adapters, result.member_states, result.member_configs = adapters, member_states, member_configs
    result.atmosphere, result.fields, result.state = dict(atmosphere), dict(fields), packed_state
    result._xp, result.device, result._metadata = array_module, device, metadata
    result._table_owners = {name: tuple(getattr(adapter, name) for adapter in adapters) for name in table_names}
    result._setup_owners = {name: tuple(getattr(adapter, name, None) for adapter in adapters)
                           for name in ("latitude_deg", "longitude_deg", "_ozone_logp", "_ozone_vmr", "chunk_workspace")}
    result.stream = array_module.cuda.Stream(non_blocking=True)
    result._input_ready, result._output_ready = array_module.cuda.Event(), array_module.cuda.Event()
    result._lock, result._closed, result._released = threading.Lock(), False, {}
    result.reuse_scope = reuse_scope if borrow_workspace else None
    from gpuwm.core import rrtmgp
    from gpuwm.core.kernels import module_source
    result.receipt = {"members": members, "nz": nz, "member_shape": (ny, nx),
        "native_calls": 0, "successful_calls": 0, "native_calls_per_invocation": 1,
        "algorithm": "ordinary all-member RTE+RRTMGP column call", "metadata": metadata,
        "source_sha256": sha256(inspect.getsource(RRTMGPRadiation.__call__).encode()).hexdigest(),
        "module_sha256": sha256(inspect.getsource(rrtmgp).encode()).hexdigest(),
        "kernel_source_sha256": {name: sha256(module_source(name).encode()).hexdigest()
                                 for name in ("rrtmgp_gas", "rrtmgp_cloud", "rrtmgp_mcica",
                                              "rrtmgp_rte", "rrtmgp_validation")},
        "table_sha256": identities, "memory": memory,
        "workspace_ownership": "borrowed original member workspace" if borrow_workspace else "native allocation",
        "borrowed_workspace_bytes": memory["solver_workspace_bytes"] if borrow_workspace else 0,
        "ozone_identity": ozone_identity,
        "required_binding_bytes": storage.plan.required_bytes(members),
        "geometry_payload_bytes": storage.payload_bytes, "workspace_payload_bytes": 0 if workspace is None else workspace.nbytes,
        "native_workspace_policy": "shared native workspace" if native_workspace else "ordinary workspace-less branch",
        "stream_scratch_policy": "private stream; native chunk cache released by close",
        "field_writeback": "native returned carrier; caller owns driver writeback and mass coupling"}
    return result
