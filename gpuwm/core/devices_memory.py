"""CPU admission for permanently resident ranks, grouped by physical card."""
from __future__ import annotations

from dataclasses import replace
from math import prod
from types import SimpleNamespace

from gpuwm.core.devices import DEVICES_OFF, DevicesRefused, refuse_unrouted_devices

GIB = 2**30
# Per-rank performance storage. Requests above this cap use the direct,
# producer-fenced output copy, preserving capacity without changing fields.
FRAME_SNAPSHOT_LIMIT_BYTES = 4 * GIB


def inventory_shapes(cfg):
    """Conservative carrier census from the resident allocation inventories.

    Physics persistents include rebuilt operands, so this is an upper bound
    until a prepared bundle supplies its actual carrier inventory.
    """
    from gpuwm.core import preflight as pf
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS
    from gpuwm.io.restart import SERIALIZED_SCRATCH_SLOTS
    shapes = {"state/" + key: shape
              for key, shape in pf.state_array_shapes(cfg).items()
              if key in STATE_SERIALIZED_ATTRS}
    shapes.update({"driver/" + key: shape
                   for key, shape in pf.physics_array_shapes(cfg).items()})
    shapes.update({"scratch/" + key: shape
                   for key, shape in pf.scratch_slot_registry(cfg).items()
                   if key in SERIALIZED_SCRATCH_SLOTS or key == "refl_10cm"})
    return {key: shape for key, shape in shapes.items()
            if len(shape) >= 2 and shape[-2] in (cfg.ny, cfg.ny + 1)
            and shape[-1] in (cfg.nx, cfg.nx + 1)}


def _arrays(cfg):
    import numpy as np
    return {name: SimpleNamespace(shape=shape, ndim=len(shape),
                                 dtype=np.dtype("float32"),
                                 flags=SimpleNamespace(c_contiguous=True))
            for name, shape in inventory_shapes(cfg).items()}


def estimate_devices(exp, *, options=None, max_map_factor=1.0,
                     forcing_intervals=None, forcing_interval_seconds=3600.0,
                     vram_gib=None, profile=None, inventory=None,
                     geography=None, source=None, profiles=None):
    """Price rank resident envelopes, packed bands, template, and pinned host.

    The lazy rank APIs own halo and decomposition validation. No device is
    opened here. Repeated ids sum all resident ranks on that card.
    """
    from gpuwm.core import preflight as pf
    from gpuwm.core.streaming import ranked_halo, ranked_specs
    from gpuwm.ingest.prepared_store import default_slab_rows
    from tilestream.harness import tile_config
    from tilestream.multigpu import seam_plan, _bands
    if len(exp.domains) != 1:
        # A tree is priced grid by grid through estimate_devices_tree,
        # which calls this once per split grid as a one-grid experiment.
        refuse_unrouted_devices(exp, "rank memory admission for a nested tree "
                                "(estimate_devices_tree prices a tree)")
    options = exp.devices if options is None else options
    cfg = exp.root.run
    if options.enabled:
        from gpuwm.core.devices import validate_ranked_physics
        validate_ranked_physics(cfg)
    try:
        halo = ranked_halo(cfg, max_map_factor=max_map_factor)
        specs = ranked_specs(cfg, options, halo=halo)
    except (ValueError, RuntimeError) as error:
        raise DevicesRefused(f"[devices] split refused before allocation: {error}") from error
    active = max(int(cfg.spec_zone), int(cfg.relax_zone)) if cfg.specified else 0
    for rank, spec in enumerate(specs):
        if min(spec.interior_nx, spec.interior_ny) < max(halo + 1, 2 * active + 1):
            raise DevicesRefused(
                f"[devices] rank {rank} interior {spec.interior_ny}x{spec.interior_nx} "
                f"is narrower than halo + 1 = {halo + 1} or twice boundary "
                f"frame {active}; refused to prevent a seam reading a neighbor's "
                "halo or a boundary frame without a unique interior")
    ids = options.device_ids()
    cards = {dev: {"card": dev, "ranks": [], "resident_bytes": 0,
                   "seam_bytes": 0, "template_bytes": 0,
                   "frame_snapshot_bytes": 0} for dev in ids}
    def resident(run, dev=None):
        local = replace(exp, devices=DEVICES_OFF,
                        domains=(replace(exp.root, run=run),))
        from gpuwm.boundary_fields import source_boundary_species
        return pf.estimate_experiment(
            local, forcing_intervals=forcing_intervals,
            forcing_interval_seconds=forcing_interval_seconds,
            vram_gib=vram_gib, profile=(profile if profiles is None else profiles.get(dev)),
            boundary_species=source_boundary_species(source))
    arrays = []
    ranks = []
    for rank, (dev, spec) in enumerate(zip(ids, specs)):
        run = tile_config(cfg, spec.cnx, spec.cny)
        estimate = resident(run, dev)
        cards[dev]["ranks"].append(rank)
        cards[dev]["resident_bytes"] += estimate.peak_envelope_bytes
        if inventory is None:
            arrays.append(_arrays(run))
        else:
            import numpy as np
            arrays.append({key: SimpleNamespace(
                shape=tuple(array.shape[:-2]) +
                      (spec.cny + array.shape[-2] - cfg.ny,
                       spec.cnx + array.shape[-1] - cfg.nx),
                ndim=array.ndim, dtype=array.dtype,
                flags=SimpleNamespace(c_contiguous=True))
                for key, array in inventory.items() if array.ndim >= 2})
        ranks.append({"rank": rank, "card": dev, "compute_shape": [spec.cny, spec.cnx],
                      "interior_shape": [spec.interior_ny, spec.interior_nx],
                      "resident_bytes": estimate.peak_envelope_bytes})
        # A frame keeps output-time device copies alive while the slabs
        # step and their copies download to pinned host memory. Bound the
        # lazily allocated member buffers by the entire carrier inventory,
        # including halos and staggered faces, on each rank. Repeated ids
        # own every rank's buffers on the same physical card.
        snapshot = min(FRAME_SNAPSHOT_LIMIT_BYTES,
                       sum(prod(array.shape) * array.dtype.itemsize
                           for array in arrays[-1].values()))
        cards[dev]["frame_snapshot_bytes"] += snapshot
        ranks[-1]["frame_snapshot_bytes"] = snapshot
    staging = 0
    for seam in seam_plan(specs, halo, nx=cfg.nx, ny=cfg.ny):
        _, size = _bands(seam, arrays[seam.src_gpu], arrays[seam.dst_gpu], cfg.nz,
                         specs[seam.src_gpu], specs[seam.dst_gpu],
                         sorted(arrays[seam.src_gpu]))
        # A channel owns one source send buffer and one destination receive buffer.
        cards[ids[seam.src_gpu]]["seam_bytes"] += size
        cards[ids[seam.dst_gpu]]["seam_bytes"] += size
        if options.transport == "host" and ids[seam.src_gpu] != ids[seam.dst_gpu]:
            staging += size
    rows = default_slab_rows(cfg.nx, cfg.ny)
    last = cfg.ny % rows or rows
    template = resident(tile_config(cfg, cfg.nx, last), ids[0])
    # The retained template keeps resident arrays, not a second step envelope.
    cards[ids[0]]["template_bytes"] = template.domains[0].resident_bytes
    loader = resident(tile_config(cfg, cfg.nx, rows), ids[0]).peak_envelope_bytes
    for row in cards.values():
        row["total_bytes"] = sum(row[key] for key in
                                 ("resident_bytes", "seam_bytes", "template_bytes",
                                  "frame_snapshot_bytes"))
        if row["card"] == ids[0]:
            row["total_bytes"] = max(row["total_bytes"], loader)
    # Same shape products used by prepared_store's allocation guard. With a
    # bundle this is its own exact manifest; before preparation use the census.
    if inventory is None:
        # Use the host store's FieldSpec products, also used by prepared_store.
        from tilestream.hoststore import FieldSpec, domain_bytes
        import numpy as np
        def manifest(shapes):
            return tuple(FieldSpec(
                key, np.dtype("float32"), len(shape) == 3 and shape[0] in (cfg.nz, cfg.nz + 1),
                len(shape) == 3 and shape[0] == cfg.nz + 1,
                shape[-2] == cfg.ny + 1, shape[-1] == cfg.nx + 1,
                layers=shape[0] if len(shape) == 3 and shape[0] not in (cfg.nz, cfg.nz + 1) else None)
                for key, shape in shapes.items())
        store = domain_bytes(manifest(inventory_shapes(cfg)), cfg.nz, cfg.ny, cfg.nx)
        from gpuwm.state_serialization_contract import STATE_SETUP_ARRAYS, STATE_DERIVED_SETUP_ARRAYS
        setup = {key: shape for key, shape in pf.state_array_shapes(cfg).items()
                 if key in STATE_SETUP_ARRAYS + STATE_DERIVED_SETUP_ARRAYS and len(shape) >= 2}
        geo = domain_bytes(manifest(setup), cfg.nz, cfg.ny, cfg.nx)
        # The geography owners are driver._SCHEME_GEOGRAPHY, including
        # independently selected radiation components. Scheme coordinates
        # can be float64; bound those operands at eight bytes, not four.
        from gpuwm.config import radiation_scheme_ids
        lw, sw = radiation_scheme_ids(cfg)
        radiation_owners = int(bool(lw or sw)) + (2 if lw != sw and lw and sw else 0)
        coordinate_owners = (radiation_owners
                             + int(cfg.sf_surface_physics == 4)
                             + int(cfg.o3input == 2))
        geo += coordinate_owners * 2 * cfg.nx * cfg.ny * 8
        basis = "hoststore FieldSpec products over resident carrier/setup census upper bound"
    else:
        store = sum(int(array.nbytes) for array in inventory.values())
        geo = sum(int(array.nbytes) for array in (geography or {}).values()
                  if hasattr(array, "nbytes"))
        basis = "prepared store and geography allocation inventory"
    intervals = pf.lbc_intervals(exp.run_seconds, forcing_interval_seconds,
                                retained_intervals=forcing_intervals)
    host_boundary = pf.lbc_host_series_bytes(cfg, intervals, source=source)
    return {"options": options.to_json(), "grid": list(options.resolved_grid(cfg.nx, cfg.ny)),
            "halo": halo, "rank_shapes": ranks, "cards": list(cards.values()),
            "host_store_bytes": store + geo, "host_staging_bytes": staging,
            "host_boundary_bytes": host_boundary,
            "host_bytes": store + geo + staging + host_boundary, "host_basis": basis}


def tree_grid_as_root(dc):
    """One grid of a tree priced as a single-domain root (memory only).

    A nest's own forcing is its coupler's rolling tables, not a tabulated
    series; pricing it as a specified root prices a boundary series in their
    place, an upper bound of the same order.
    """
    run = replace(dc.run, nested=False, specified=True)
    step = dc.time_step if dc.time_step is not None else max(1, int(dc.run.dt))
    return replace(dc, parent_id=0, i_parent_start=1, j_parent_start=1,
                   parent_grid_ratio=1, parent_time_step_ratio=1, run=run,
                   time_step=step, time_step_fract_num=0,
                   time_step_fract_den=1, start_time=None)


def estimate_devices_tree(exp, *, split_ids=None, forcing_intervals=None,
                          forcing_interval_seconds=3600.0, source=None,
                          vram_gib=None, profile=None, profiles=None):
    """Price a split TREE per card: the one pricing the door and the runner share.

    Each split grid is priced the way a split single domain is
    (:func:`estimate_devices`: every slab's resident envelope, its packed
    seam bands and the store-building template), each resident grid as a
    resident domain on the first card of ``ids``, and the pinned host
    stores of the split grids together.  ``gpuwm check --devices``, the
    ``gpuwm go`` gate before the download and the tree runner before its
    first restore all read these numbers, so the three cannot disagree
    about one card.  Every card row carries ``grids``: the bytes each grid
    puts on it.  ``split_ids`` defaults to the tree's own by-name
    validation (:func:`gpuwm.core.devices.validate_tree_devices`), so a
    tree it refuses is refused here too, before anything is priced.
    """
    from gpuwm.boundary_fields import source_boundary_species
    from gpuwm.core import preflight as pf
    from gpuwm.core.devices import validate_tree_devices
    if split_ids is None:
        split_ids = validate_tree_devices(exp)
    split = tuple(int(gid) for gid in split_ids)
    options = exp.devices
    ids = list(dict.fromkeys(options.device_ids()))
    first = ids[0]
    cards = {dev: {"card": dev, "ranks": [], "resident_bytes": 0,
                   "seam_bytes": 0, "template_bytes": 0, "total_bytes": 0,
                   "frame_snapshot_bytes": 0,
                   "grids": {}} for dev in ids}
    host = {"host_store_bytes": 0, "host_staging_bytes": 0,
            "host_boundary_bytes": 0}
    # Each grid is priced as a one-grid experiment, so the split it carries
    # names no domains list: a list naming another grid of the tree would
    # be refused there as a grid that experiment does not have.
    split_options = replace(options, domains=None)
    grids, halos = [], {}
    for dc in exp.domains:
        gid = int(dc.grid_id)
        local = replace(exp, domains=(tree_grid_as_root(dc),),
                        devices=split_options if gid in split else DEVICES_OFF)
        if gid in split:
            estimate = estimate_devices(
                local, forcing_intervals=forcing_intervals,
                forcing_interval_seconds=forcing_interval_seconds,
                source=source, vram_gib=vram_gib, profile=profile,
                profiles=profiles)
            for row in estimate["cards"]:
                target = cards[row["card"]]
                for key in ("resident_bytes", "seam_bytes", "template_bytes",
                            "frame_snapshot_bytes", "total_bytes"):
                    target[key] += int(row[key])
                target["ranks"].extend(f"d{gid:02d}:{rank}" for rank in row["ranks"])
                target["grids"][gid] = target["grids"].get(gid, 0) + int(row["total_bytes"])
            for key in host:
                host[key] += int(estimate[key])
            halos[gid] = estimate["halo"]
            grids.append({"grid_id": gid, "road": "split", "halo": estimate["halo"],
                          "grid": estimate["grid"],
                          "cards": [{"card": r["card"], "total_bytes": int(r["total_bytes"])}
                                    for r in estimate["cards"]],
                          "host_bytes": int(estimate["host_bytes"])})
        else:
            resident = pf.estimate_experiment(
                local, forcing_intervals=forcing_intervals,
                forcing_interval_seconds=forcing_interval_seconds,
                vram_gib=vram_gib,
                profile=(profile if profiles is None else profiles.get(first)),
                boundary_species=source_boundary_species(source))
            size = int(resident.peak_envelope_bytes)
            cards[first]["resident_bytes"] += size
            cards[first]["total_bytes"] += size
            cards[first]["grids"][gid] = cards[first]["grids"].get(gid, 0) + size
            grids.append({"grid_id": gid, "road": "resident", "card": first,
                          "total_bytes": size})
    return {"options": options.to_json(), "split_grid_ids": list(split),
            "grids": grids, "halo": halos, "cards": list(cards.values()),
            **host, "host_bytes": sum(host.values()),
            "host_basis": "per split grid: " + ("hoststore FieldSpec products over "
                                                "resident carrier/setup census upper bound")}


def devices_gate(estimate, *, budgets=None, host_budget=None):
    lines = []
    refused = False
    for row in estimate["cards"]:
        budget = None if budgets is None else budgets.get(row["card"])
        over = budget is not None and row["total_bytes"] > budget
        refused |= over
        verdict = "UNMEASURED" if budget is None else "REFUSED" if over else "ADMITTED"
        if "grids" in row:
            # A split tree: what each grid puts on this card, then the sum.
            terms = " + ".join(f"d{gid:02d} {size/GIB:.2f} GiB"
                               for gid, size in sorted(row["grids"].items()))
            lines.append(
                f"card {row['card']}: {verdict}: {terms or 'nothing'} "
                f"= {row['total_bytes']/GIB:.2f} GiB; budget "
                + ("unknown" if budget is None else f"{budget/GIB:.2f} GiB"))
            continue
        lines.append(
            f"card {row['card']}: {verdict}: resident {row['resident_bytes']/GIB:.2f} GiB "
            f"+ seams {row['seam_bytes']/GIB:.2f} GiB "
            f"+ template {row['template_bytes']/GIB:.2f} GiB "
            f"+ frame snapshots {row['frame_snapshot_bytes']/GIB:.2f} GiB "
            f"= {row['total_bytes']/GIB:.2f} GiB; budget "
            + ("unknown" if budget is None else f"{budget/GIB:.2f} GiB"))
    host_over = host_budget is not None and estimate["host_bytes"] > host_budget
    refused |= host_over
    lines.append(f"host: {'REFUSED' if host_over else 'PRICED'}: pinned store "
                 f"{estimate['host_store_bytes']/GIB:.2f} GiB + staging "
                 f"{estimate['host_staging_bytes']/GIB:.2f} GiB + boundary "
                 f"{estimate['host_boundary_bytes']/GIB:.2f} GiB")
    return {"verdict": "\n".join(lines), "refuse": bool(refused), "warn": False,
            "devices": estimate}


def include_preparation(gate, exp, *, source, budgets=None, host_budget=None,
                        forcing_intervals=None, forcing_interval_seconds=3600.0,
                        ingest_forcing_interval_seconds=None, vram_gib=None,
                        profile=None):
    """Keep the existing preparation gate beside the per-card forecast gate."""
    from gpuwm.core import preflight as pf
    kwargs = dict(source=source, forcing_intervals=forcing_intervals,
                  forcing_interval_seconds=forcing_interval_seconds,
                  ingest_forcing_interval_seconds=ingest_forcing_interval_seconds,
                  vram_gib=vram_gib, profile=profile)
    phases = pf.estimate_phases(exp, **kwargs)
    first_budget = None if budgets is None else budgets.get(exp.devices.device_ids()[0])
    moved_to_cpu = False
    if (phases.ingest_priced and first_budget is not None
            and phases.ingest_envelope_bytes > first_budget):
        if getattr(phases, "preprocess_backend", "cuda") == "auto":
            phases = pf.estimate_phases(exp, preprocess_backend="cpu", **kwargs)
            moved_to_cpu = True
        else:
            gate["refuse"] = True
            gate["verdict"] += ("\npreparation: REFUSED: "
                                f"{phases.ingest_envelope_bytes/GIB:.2f} GiB exceeds "
                                f"first card budget {first_budget/GIB:.2f} GiB")
    host_refusal = phases.host_preparation_refusal(host_budget)
    if host_refusal:
        gate["refuse"] = True
        gate["verdict"] += "\npreparation: " + host_refusal
    gate["preparation"] = {"priced": phases.ingest_priced,
                           "backend": getattr(phases, "preprocess_backend", None),
                           "may_move_to_cpu": moved_to_cpu,
                           "device_bytes": phases.ingest_envelope_bytes,
                           "host_refusal": host_refusal}
    return gate
