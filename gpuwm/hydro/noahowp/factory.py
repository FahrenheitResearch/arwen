"""Realization table factories for resident Noah-OWP and parent LSM exchange."""
from __future__ import annotations


def _resolve(reference, components):
    """Resolve setup-only component:array references from a realization row."""
    component, separator, name = reference.partition(":")
    if not separator or component not in components:
        raise ValueError(f"unknown prepared component reference {reference!r}")
    source = components[component]
    if name in source.arrays:
        return source.arrays[name]
    fields = getattr(source, "fields", {})
    if name in fields:
        return fields[name]
    raise ValueError(f"prepared component has no array {reference!r}")


def prepare_slab_prebuild(*, settings, options, device_id):
    """Start the Noah-OWP physics compile before any stage prepares (gpuwm.hydro.assembly.prebuild_kernels)."""
    from .runtime import prebuild
    prebuild(settings.get("numerics", "fast"), device_id)


def prepare_slab(*, settings, options, state, physics, device_id, mode, components):
    """Bind complete Rust-prepared initial state and optional canonical forcing.

    ``slabs`` maps f32/i32/f64 to component:array references. Alternatively,
    ``fields`` supplies the complete schema by name. Parameter initialization
    and hydrofabric parsing finish in their Rust preparer before this factory.
    """
    from datetime import datetime, timezone
    import cupy as cp
    from .runtime import PreparedNoahOWP

    try:
        origin = datetime.fromisoformat(options["time_origin"])
    except (TypeError, ValueError) as error:
        raise ValueError("Noah-OWP time_origin must name the prepared start timestamp") from error
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=timezone.utc)
    epoch = origin.timestamp()

    initial = options.get("slabs")
    fields = options.get("fields")
    manifest_path = options.get("initial_manifest")
    if sum(x is not None for x in (initial, fields, manifest_path)) != 1:
        raise ValueError("Noah-OWP requires initial_manifest, prepared slabs, or complete fields")
    metadata = None
    parameter_hash = options.get("parameter_sha256")
    if manifest_path is not None:
        from .initial import load_initial_cuda
        initial, metadata = load_initial_cuda(manifest_path, device_id=device_id)
        if parameter_hash is not None and parameter_hash != metadata["parameter_sha256"]:
            raise ValueError("Noah-OWP initial parameter identity differs from realization")
        parameter_hash = metadata["parameter_sha256"]
        if "catchment_ids" in options and list(options["catchment_ids"]) != [c["id"] for c in metadata["columns"]]:
            raise ValueError("Noah-OWP initial roster differs from the realization catchment order")
    if parameter_hash is None:
        raise ValueError("Noah-OWP prepared slabs require parameter_sha256")
    forcing = None
    if "forcing_component" in options:
        source = components[options["forcing_component"]]
        forcing = source.fields
    component = PreparedNoahOWP(
        None if fields is None else {k: _resolve(v, components) for k, v in fields.items()},
        slabs=None if initial is None else initial if metadata is not None else {
            k: _resolve(v, components) for k, v in initial.items()},
        forcing=forcing, numerics=settings.get("numerics", "fast"),
        parameter_sha256=parameter_hash,
        fabric_sha256=options["fabric_sha256"], time_origin=options["time_origin"])
    dt = component.arrays["domain.dt"]
    interval = physics.bldt_seconds if mode == "coupled" else settings["interval_seconds"]
    if dt.device.id != device_id or bool(cp.any(dt != cp.float32(interval)).item()):
        raise ValueError("Noah-OWP resident dt/card differs from the prepared driver")
    if bool(cp.any(component.arrays["domain.start_datetime"] != cp.float64(epoch)).item()):
        raise ValueError("Noah-OWP resident start date differs from realization time_origin")
    if metadata is not None:
        component.restart_identity["initial_request_sha256"] = metadata["request_sha256"]
        component.restart_identity["initial_rules_sha256"] = metadata["rules_sha256"]
        component.restart_identity["catchment_ids"] = tuple(c["id"] for c in metadata["columns"])
    return component


def _parent_fields(physics, mode):
    if mode != "coupled" or physics.noahmp_params is None:
        raise ValueError("catchment land exchange requires the parent Noah-MP LSM")
    from gpuwm.core.noahmp_runtime import _slab_path_selected
    if not _slab_path_selected():
        raise ValueError("hydro bottom drainage requires the GPU Noah-MP slab path; host diagnostic leaves do not export it")
    import cupy as cp

    fields = physics.fields
    # Attach once, before the first parent LSM call and checkpoint inventory.
    # Ordinary and glacier publication both supply mm per LSM step.
    fields["infxsrt"] = fields["runsfxy"]
    for name in ("sfcheadrt", "soldrain"):
        if name not in fields:
            fields[name] = cp.zeros_like(fields["runsfxy"])
    return fields


def _parent_inputs(fields, physics, settings, options):
    return dict(runsrf=fields["infxsrt"], runsub=fields["runsbxy"],
                soldrain=fields["soldrain"], sfcheadrt=fields["sfcheadrt"],
                smois=fields["smois"], sh2o=fields["sh2o"], ecan=fields["ecanxy"],
                etran=fields["etranxy"], edir=fields["edirxy"],
                dt_seconds=physics.bldt_seconds,
                weights_sha256=options["weights_sha256"],
                fabric_sha256=options["fabric_sha256"], numerics=settings.get("numerics", "fast"))


def prepare_exchange(*, settings, options, state, physics, device_id, mode, components):
    """Bind parent step depths, ET rates, soil state and Rust catchment rows.

    sfcheadrt is persistent mm storage, never a runoff rate. It is exposed for
    a terrain stage to own; this downstream exchange does not reinfiltrate it.
    No additional atmospheric-forcing or PET stage is required.
    """
    from gpuwm.hydrofabric import load_cuda
    from .coupling import PreparedCatchmentExchange

    fields = _parent_fields(physics, mode)

    fabric = (components[options["network_component"]] if "network_component" in options
              else load_cuda(options["hydrofabric_manifest"], device_id=device_id))
    arrays = fabric.arrays
    if "network_component" in options:
        from gpuwm.hydro.parent_geometry import bind
        from gpuwm.hydrofabric import parent_grid_contract
        grid = getattr(physics, "hydro_parent_geometry", None)
        if grid is None:
            raise ValueError("parent runoff exchange needs the actual forecast ProjectedGrid")
        if fabric.manifest.get("parent_grid_contract") != parent_grid_contract(bind(grid)):
            raise ValueError("prepared hydro weights differ from the actual forecast parent grid")
    component = PreparedCatchmentExchange(
        offsets=arrays["exchange_offsets"], indices=arrays["exchange_indices"],
        weights=arrays["exchange_weights"], **_parent_inputs(fields, physics, settings, options))
    component.restart_identity["hydrofabric"] = fabric.restart_identity
    component.restart_identity["catchment_ids"] = list(fabric.catchment_ids)
    return component


def prepare_accumulated_exchange(*, settings, options, state, physics, device_id, mode, components):
    """Parent exchange for an LSM that accumulates runoff (Noah, RUC).

    Binds the parent's resident SFCRUNOFF/UDRUNOFF (mm accumulated), QFX and
    soil carriers to the same Rust-prepared catchment CSR the Noah-MP exchange
    uses; the step depth is differenced on the card at each LSM call.
    """
    if mode != "coupled":
        raise ValueError("parent runoff exchange requires coupled hydro mode")
    from gpuwm.hydro.parent_geometry import bind
    from gpuwm.hydrofabric import parent_grid_contract
    from .coupling import PreparedAccumulatedExchange

    fields = physics.fields
    missing = [name for name in ("sfcrunoff", "udrunoff", "qfx", "smois", "sh2o") if name not in fields]
    if missing:
        raise ValueError(f"parent LSM publishes no {missing}; the accumulated-runoff exchange needs them")
    fabric = components[options["network_component"]]
    grid = getattr(physics, "hydro_parent_geometry", None)
    if grid is None:
        raise ValueError("parent runoff exchange needs the actual forecast ProjectedGrid")
    if fabric.manifest.get("parent_grid_contract") != parent_grid_contract(bind(grid)):
        raise ValueError("prepared hydro weights differ from the actual forecast parent grid")
    arrays = fabric.arrays
    component = PreparedAccumulatedExchange(
        offsets=arrays["exchange_offsets"], indices=arrays["exchange_indices"],
        weights=arrays["exchange_weights"], surface_total=fields["sfcrunoff"],
        subsurface_total=fields["udrunoff"], qfx=fields["qfx"], smois=fields["smois"],
        sh2o=fields["sh2o"],
        # Launched once per routing interval when the realization says so: the
        # depth is everything accumulated since the previous launch.
        dt_seconds=float(options.get("interval_seconds", physics.bldt_seconds)),
        weights_sha256=options["weights_sha256"], fabric_sha256=options["fabric_sha256"],
        numerics=settings.get("numerics", "fast"))
    component.restart_identity["hydrofabric"] = fabric.restart_identity
    component.restart_identity["catchment_ids"] = list(fabric.catchment_ids)
    return component


def prepare_accumulated_exchange_memory(*, settings, options, context, components):
    from ..coupling_memory import accumulated_exchange_memory
    return accumulated_exchange_memory(settings=settings, options=options, context=context,
                                       components=components)


def prepare_terrain_exchange(*, settings, options, state, physics, device_id, mode, components):
    """Map parent depths to terrain rates using Rust-prepared destination CSR.

    mapping_arrays contains offsets/indices/weights component:array references.
    No identity mapping is inferred from matching dimensions.
    """
    from .coupling import PreparedRasterExchange

    fields = _parent_fields(physics, mode)
    mapping = {key: _resolve(options["mapping_arrays"][key], components)
               for key in ("offsets", "indices", "weights")}
    if fields["runsfxy"].device.id != device_id:
        raise ValueError("terrain exchange parent differs from the prepared driver card")
    return PreparedRasterExchange(
        **mapping, **_parent_inputs(fields, physics, settings, options),
        destination_shape=options["destination_shape"],
        parent_grid_sha256=options["parent_grid_sha256"],
        terrain_grid_sha256=options["terrain_grid_sha256"])


def prepare_slab_memory(*, settings, options, context, components):
    from ..runoff_memory import noahowp_memory
    return noahowp_memory(settings=settings, options=options, context=context, components=components)


def prepare_exchange_memory(*, settings, options, context, components):
    from ..coupling_memory import exchange_memory
    return exchange_memory(settings=settings, options=options, context=context, components=components)
