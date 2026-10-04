"""Original-member forecast and ensemble-output reservations before packing."""
from __future__ import annotations

from fractions import Fraction
from dataclasses import replace

from gpuwm.ensemble.admission import (
    AllocatorMargin, EnsembleMemoryModel, MemoryComponent,
    ordinary_memory_model_from_estimate,
)


def ordinary_output_memory_components(experiment, collector, *, other_experiments=()):
    """Declared collector buffers for every possible original domain shape.

    The collector retains a workspace per card and horizontal shape. Upload
    and replay call peaks are bounded by the largest original domain, while
    retained diagnostic/QPF buffers are priced for every distinct shape.
    A domain's original full-state and step-health arrays remain in the
    ordinary preflight scratch inventory, rather than being counted twice.
    """
    from gpuwm.ensemble.batch_product_output import replay_memory_plan_for_shape
    from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
    domains = tuple(domain for exp in (experiment,) + tuple(other_experiments) for domain in exp.domains)
    if not domains:
        raise ValueError("ordinary ensemble output admission needs actual domains")
    by_shape = {}
    for domain in domains:
        shape = (int(domain.run.ny), int(domain.run.nx))
        by_shape[shape] = max(by_shape.get(shape, 0), int(domain.run.nz))
    retained, uploads, replays = [], [], []
    for number, (shape, levels) in enumerate(sorted(by_shape.items())):
        persistent = collector.memory_plan(shape)
        retained.extend(BatchArraySpec(f"shape{number}:" + spec.name,
                        spec.shape, spec.ownership, spec.dtype) for spec in persistent.arrays)
        full = collector.memory_plan(shape, refl_levels=levels, host_inputs=True)
        existing = {spec.name for spec in persistent.arrays}
        upload = BatchMemoryPlan(tuple(spec for spec in full.arrays if spec.name not in existing), reserved_bytes=0)
        uploads.append(upload.required_bytes(1))
        replay = replay_memory_plan_for_shape(collector.requests, shape,
            members=collector.members, tile_rows=collector.tile_rows)
        replays.append(replay.required_bytes(collector.members))
    return (
        MemoryComponent("ensemble_diagnostic_qpf_buffers", "products",
            plan=BatchMemoryPlan(tuple(retained), reserved_bytes=0),
            evidence="collector.memory_plan for every distinct domain shape, including initial/earlier and 1/3/6 h QPF surfaces"),
        MemoryComponent("ensemble_host_upload_peak", "products", fixed_bytes=max(uploads),
            evidence="largest collector host_inputs plan minus its retained workspace; one synchronous borrowed frame per card"),
        MemoryComponent("ensemble_full_roster_replay", "products", fixed_bytes=max(replays),
            evidence="largest replay_memory_plan_for_shape for the complete requested roster and bounded tile rows"),
    )


def ordinary_ensemble_memory_model(estimate, experiment, collector, *, inventory_id,
                                   stochastic_provider=None, fft_workspace_bytes=0,
                                   other_variants=()):
    """Add ensemble allocations to the original complete forecast envelope."""
    from gpuwm.core.preflight import FORECAST_POOL_HEADROOM
    variants = ((estimate, experiment),) + tuple(other_variants)
    originals = tuple(ordinary_memory_model_from_estimate(value, inventory_id=inventory_id)
                      for value, _ in variants)
    original = originals[0]
    if len(originals) > 1:
        original = EnsembleMemoryModel((MemoryComponent("ordinary_variant_forecast_peak", "forecast",
            fixed_bytes=max(value.required_bytes(1) for value in originals), basis="envelope",
            evidence="maximum original full forecast envelope over every bound member configuration; one live member per card"),),
            inventory_id=inventory_id)
    external = ordinary_output_memory_components(experiment, collector,
        other_experiments=tuple(exp for _, exp in variants[1:]))
    if stochastic_provider is not None:
        stochastic = [stochastic_provider.memory_components(exp, fft_workspace_bytes=fft_workspace_bytes)
                      for _, exp in variants]
        external += max(stochastic, key=lambda components: sum(component.inventory(1)["required_bytes"]
                        for component in components))
    # The forecast envelope already contains its original pool headroom,
    # context/kernel backing, health inventory and external minimum reserve.
    # Apply the same measured fractional headroom only to the new arrays.
    fraction = Fraction(str(FORECAST_POOL_HEADROOM)) - 1
    margin = AllocatorMargin(fraction.numerator, fraction.denominator,
        categories=("products", "stochastic"),
        evidence="original FORECAST_POOL_HEADROOM for additional ensemble buffers; forecast health and EXTERNAL_MARGIN_BYTES already in original peak envelope")
    model = EnsembleMemoryModel(original.components + external, margin, inventory_id)
    return model, external, margin


def ordinary_stochastic_streamed_inputs(inputs, *, batch, external_components,
                                       allocator_margin, stochastic_provider,
                                       fft_workspace_bytes, device_profile=None):
    """Reserve actual planned local stochastic windows before initialization.

    Search only the original execution-only tile budget. Original clocks,
    adaptive rules, physics choices, source/head and prepared arrays survive.
    A smaller budget is retained monotonically until its original tile plan
    and the stochastic windows that plan owns both fit the same card.
    """
    from gpuwm.core import streaming
    from gpuwm.ensemble.prepared_execution import ordinary_member_execution_inputs
    products = tuple(component for component in external_components if component.category != "stochastic")
    globals_only = products + stochastic_provider.memory_components(inputs.experiment,
        fft_workspace_bytes=fft_workspace_bytes, include_rates=False)
    base, global_receipt = ordinary_member_execution_inputs(inputs, batch=batch,
        external_components=globals_only, allocator_margin=allocator_margin)
    budget = int(global_receipt["ordinary_tile_budget_bytes"])
    iterations = 0
    def cap(value, budget):
        if value is None:
            return None
        return replace(value, vram_budget_bytes=min(budget, value.vram_budget_bytes)
                       if value.vram_budget_bytes is not None else budget)
    while True:
        iterations += 1
        exp = replace(base.experiment, tiles=cap(base.experiment.tiles, budget),
            domains=tuple(replace(domain, tiles=cap(domain.tiles, budget)) for domain in base.experiment.domains))
        candidate = replace(base, experiment=exp)
        machine = streaming.planner_machine(vram_bytes=budget, name="ensemble selected card",
                                             device_profile=device_profile)
        windows, decisions = {}, []
        for domain in exp.domains:
            options = streaming.options_for_domain(domain, exp.tiles)
            devices = getattr(exp, "devices", None)
            ranked = (int(getattr(devices, "count", 1)) > 1 and
                      (getattr(devices, "domains", None) is None or domain.grid_id in devices.domains))
            decision = (streaming.ranked_decision(domain.run, devices) if ranked else
                        streaming.decide(domain.run, options, machine=machine, allow_resident=False))
            if not decision.stream or decision.tile_nx is None or decision.tile_ny is None:
                raise RuntimeError("original streamed planner did not supply its stochastic compute windows")
            specs = (streaming.ranked_specs(domain.run, devices, halo=decision.halo) if ranked else
                     streaming.tile_specs(domain.run, decision))
            shapes = tuple(sorted({(int(spec.cny), int(spec.cnx)) for spec in specs}))
            windows[int(domain.grid_id)] = (shapes, int(decision.nbuffers or 1))
            decisions.append({"grid_id": int(domain.grid_id), "tile_nx": decision.tile_nx,
                "tile_ny": decision.tile_ny, "halo": decision.halo,
                "nbuffers": int(decision.nbuffers or 1), "window_shapes": shapes,
                "road": decision.road})
        external = products + stochastic_provider.memory_components(exp,
            fft_workspace_bytes=fft_workspace_bytes, window_shapes=windows)
        rows = tuple(component.inventory(1) for component in external)
        withheld = sum(row["required_bytes"] for row in rows) + allocator_margin.required_bytes(rows)
        remaining = int(batch.available_bytes) - withheld
        if remaining >= budget:
            admitted, receipt = ordinary_member_execution_inputs(candidate, batch=batch,
                external_components=external, allocator_margin=allocator_margin)
            receipt.update(stochastic_window_admission={"basis": "original planner and exact tile-spec shape classes",
                "iterations": iterations, "decisions": decisions,
                "global_pattern_storage": "whole domain", "rates": "original local tile/rank windows",
                "effective_tile_budget_bytes": budget})
            return admitted, receipt
        budget = remaining if remaining > 0 else budget // 2
        if budget <= 0:
            raise MemoryError("no original streamed tile budget remains beside its whole-domain stochastic patterns and ensemble products")


def require_bootstrap_device(node, *, array_module, device_id):
    """Read-only ownership check, with no device field transfer or cloning."""
    array_type = getattr(array_module, "ndarray", None)
    if array_type is None:
        return
    seen = set()
    def walk(value, path):
        if isinstance(value, array_type):
            if int(value.device.id) != device_id:
                raise ValueError(f"original bootstrap array {path} belongs to another physical card")
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for name, child in value.items():
                walk(child, f"{path}/{name}")
        elif isinstance(value, (tuple, list)):
            for number, child in enumerate(value):
                walk(child, f"{path}/{number}")
        elif type(value).__module__.startswith("gpuwm.") and hasattr(value, "__dict__"):
            for name, child in vars(value).items():
                walk(child, f"{path}/{name}")
    walk(node.state, "state")


__all__ = ["ordinary_output_memory_components", "ordinary_ensemble_memory_model",
           "ordinary_stochastic_streamed_inputs", "require_bootstrap_device"]
