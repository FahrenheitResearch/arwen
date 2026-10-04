"""Shared prepared inputs and exact ordinary-member ensemble execution.

Preparation belongs to the caller and is performed once. Qualified native
batch executors can replace one admitted pack through an explicit callback.
The ordinary fallback retains the original runner, clock and physics.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from collections.abc import Mapping
import json
import hashlib
import inspect
from pathlib import Path
import re
import time

from gpuwm.ensemble.execution import (
    MemberRunControl, execute_member_packing, failed_member_rows, member_run_scope,
    name_failing_members,
)
from gpuwm.ensemble.request import EnsembleRequest
from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
from gpuwm.ensemble.packing import CardBudget, MemberBatch, pack_members
from gpuwm.ensemble.radar_output import radar_enabled, member_history_required, finish_member_radar


def _atomic_json(path, record):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, default=_json_scalar) + "\n", encoding="utf-8")
    temporary.replace(path)


def release_finished_member(array_module=None):
    """Free a finished ordinary member's device state before the next starts.

    DomainState and PhysicsDriver reference each other, so a finished
    member's arrays are cyclic garbage that only a collector pass frees.
    Breakage this prevents: packing prices one live member per card and the
    ordinary runner's own admission counts only driver-free plus pool-free
    bytes, so each uncollected member read as unavailable memory. Measured
    on a 150x150x49 case, the default pool grew 591 MB per finished member;
    on a domain needing more than half the card the third member would be
    refused or dropped to tiles after two full forecasts.
    """
    import gc
    gc.collect()
    # The frees above are queued on this thread's stream. An array module
    # without a stream interface (a planning fixture) has nothing to wait for.
    current_stream = getattr(getattr(array_module, "cuda", None), "get_current_stream", None)
    if current_stream is not None:
        current_stream().synchronize()


@contextmanager
def finished_member_release(array_module=None):
    """Release on the way out of one ordinary member's forecast."""
    try:
        yield
    except BaseException:
        # The failing member's frames still hold its state. Collect what is
        # already free and keep the first error as the one that is raised.
        release_finished_member()
        raise
    release_finished_member(array_module)


def _json_scalar(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Path):
        return str(value)
    from datetime import datetime
    if isinstance(value, datetime):
        return value.isoformat()
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"ensemble receipt cannot serialize {type(value).__name__}")


class PreparedEnsembleSession:
    """The worker and source-provider interface, independent of a CLI route.

    Input providers receive immutable shared preparation and a global member
    index. They return independently bound prepared inputs, including boundary
    source authority. No source/perturbation descriptor is silently ignored.

    Every member of an N > 1 session runs its own inputs: a member roster,
    an input provider, a source execution or active stochastic physics.
    A session with none of them refuses at first use, because it would run
    N copies of one forecast and publish zero spread and 0 or 1
    probabilities.  ``identical_members`` is the one exception and no front
    door sets it: an engine gate that runs N copies on purpose, to compare
    each with the ordinary forecast word for word, states that purpose
    there, and the run's manifest records it.
    """
    def __init__(self, request, *, output_directory, collector=None,
                 input_provider=None, native_executor=None, memory_model=None,
                 cards=None, device_scope=None, renderer=None, member_roster=None,
                 stochastic_provider=None, array_module=None,
                 card_bootstrap_factory=None, source_execution=None,
                 identical_members=None):
        self.request = EnsembleRequest.from_mapping(request)
        if identical_members is not None and (
                not isinstance(identical_members, str) or not identical_members.strip()):
            raise ValueError("identical_members states, in words, why this engine gate "
                             "runs copies of one forecast")
        self.identical_members = identical_members
        self.output_directory = Path(output_directory)
        self.collector = collector
        self.input_provider = input_provider
        self.native_executor = native_executor
        self.memory_model = memory_model
        self.cards = cards
        self.device_scope = device_scope
        self.renderer = renderer
        self.member_roster = member_roster
        self.stochastic_provider = stochastic_provider
        self.array_module = array_module
        self.card_bootstrap_factory = card_bootstrap_factory
        self.source_execution = source_execution
        if source_execution is not None:
            from gpuwm.ensemble.posted_execution import PostedRecipeExecution
            from gpuwm.ensemble.ordinary_execution import OrdinaryRecipeExecution
            if not isinstance(source_execution, (PostedRecipeExecution, OrdinaryRecipeExecution)):
                raise TypeError("recipe source execution needs its concrete ordinary or changed-field preparation owner")
            if input_provider is not None or member_roster is not None:
                raise ValueError("a posted source owner cannot also replace inputs through another member provider")
        self._stochastic_authorities = {}
        self._stochastic_input_bindings = {}
        if self.stochastic_provider is None and self.request.stochastic is not None:
            from gpuwm.ensemble.stochastic_model import StochasticModelProvider
            self.stochastic_provider = StochasticModelProvider.from_mapping(self.request.stochastic)
        self.last_manifest = None
        self.last_output_directory = None
        self.member_order = (tuple(source_execution.member_order) if source_execution is not None else
            tuple(member.member_id for member in member_roster.members)
            if member_roster is not None else tuple(range(self.request.members)))
        if len(self.member_order) != self.request.members:
            raise ValueError("requested member count differs from the verified source roster")
        if ((self.request.sources or self.request.perturbation is not None)
                and input_provider is None and member_roster is None and source_execution is None):
            # Breakage it prevents: the request names a source (or a
            # perturbation) per member, and this session would run every
            # member on the one prepared input under those names.
            from gpuwm.ensemble_admission import unbound_sources_refusal
            raise ValueError(unbound_sources_refusal())

    def _random_members(self, experiment):
        """Does stochastic physics make this session's members differ?"""
        selected = any(getattr(domain.run, f"spp_{name}", 0)
                       for domain in getattr(experiment, "domains", ()) or ()
                       for name in ("conv", "pbl", "lsm"))
        provider = self.stochastic_provider
        if provider is None:
            # Native selectors in the prepared configuration bind their own
            # provider in run_prepared.
            return selected
        declared = getattr(provider, "spp", False)
        return bool(getattr(provider, "sppt", None) is not None
                    or getattr(provider, "skebs_psi", None) is not None or selected
                    or isinstance(declared, dict) and any(declared.values()))

    def _require_member_inputs(self, experiment=None):
        """Every member of an N > 1 session runs its own inputs.

        Breakage it prevents: a session with no member source hands every
        member the same prepared inputs.  The run is N copies of one
        forecast, published with zero spread and probabilities of 0 or 1.
        A recipe names real source trajectories, so it is refused in its
        own words when no door supplied them.
        """
        if (self.input_provider is not None or self.member_roster is not None
                or self.source_execution is not None):
            return
        if self.request.recipe is not None:
            raise ValueError(
                f"the {self.request.recipe} recipe needs each member's own prepared "
                "source, and this door prepared one trajectory. Next: gpuwm ensemble "
                f"CONFIG --recipe {self.request.recipe} (or gpuwm go / gpuwm run)")
        if (self.request.members == 1 or self.identical_members is not None
                or self._random_members(experiment)):
            return
        from gpuwm.ensemble_admission import one_input_refusal
        raise ValueError(one_input_refusal(
            self.request.members,
            "This ensemble session was given no member sources, so every member "
            "would run the same prepared inputs."))

    def run_prepared(self, runner, inputs, *, output_directory=None, **runner_options):
        self._require_member_inputs(getattr(inputs, "experiment", None))
        started = time.perf_counter()
        out = Path(output_directory or self.output_directory)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "ensemble-run.json").exists():
            raise FileExistsError("ensemble output already exists; use a new run directory")
        exp = inputs.experiment
        if self.stochastic_provider is not None:
            inputs = self._configured_member_inputs(inputs)
            exp = inputs.experiment
            if self.member_roster is not None:
                from gpuwm.ensemble.member_preparation import PreparedMemberRoster
                original_roster = self.member_roster
                self.member_roster = PreparedMemberRoster(original_roster.recipe,
                    tuple(replace(member, inputs=self._configured_member_inputs(member.inputs))
                          for member in original_roster.members),
                    shared_geometry_sha256=original_roster.geometry_sha256)
        # Bind each provider exactly once before sizing. Its own configuration,
        # initial state and complete boundary trajectory remain its authority.
        member_inputs_by_id = {}
        for member in self.member_order:
            prepared = (self.source_execution.planning_inputs(member) if self.source_execution is not None else
                self.member_roster.select((member,))[0].inputs
                if self.member_roster is not None else inputs if self.input_provider is None else
                self.input_provider(shared_inputs=inputs, member_id=member, request=self.request))
            member_inputs_by_id[member] = self._configured_member_inputs(prepared)
        # Native selectors belong to each original member configuration. An
        # automatic owner preserves off variants while owning active variants.
        if self.stochastic_provider is None and any(
                getattr(domain.run, f"spp_{name}", 0)
                for prepared in (inputs, *member_inputs_by_id.values())
                for domain in getattr(prepared.experiment, "domains", ())
                for name in ("conv", "pbl", "lsm")):
            from gpuwm.ensemble.stochastic_model import StochasticModelProvider
            self.stochastic_provider = StochasticModelProvider()
            inputs = self._configured_member_inputs(inputs)
            exp = inputs.experiment
            member_inputs_by_id = {member: self._configured_member_inputs(prepared)
                                  for member, prepared in member_inputs_by_id.items()}
        variants = []
        seen_variants = set()
        for prepared in member_inputs_by_id.values():
            key = (id(prepared.experiment), getattr(prepared, "boundary_interval_seconds", None))
            if key not in seen_variants:
                seen_variants.add(key)
                variants.append(prepared)
        if self.collector is None:
            from gpuwm import rustwx
            from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
            renderer = self.renderer or rustwx.find_renderer()
            if renderer is None:
                raise RuntimeError(rustwx.renderer_remedy())
            if self.source_execution is not None:
                metadata = tuple(self.source_execution.member_metadata.values())
            elif self.member_roster is not None:
                metadata = self.member_roster.receipts
            elif self.request.retain_member_diagnostics:
                from gpuwm.ensemble.seeds import member_seed
                metadata = tuple({"member_id": member,
                    "seed": member_seed(self.request.base_seed, member),
                    "source": getattr(prepared, "source", None),
                    "prepared_authority_sha256": dict(getattr(prepared, "authority_sha256", {}) or {}),
                    "prepared_head_sha256": getattr(prepared, "prepared_head_sha256", None)}
                    for member, prepared in member_inputs_by_id.items())
            else:
                metadata = ()
            self.collector = HeadlineDiagnosticCollector(out, members=self.request.members,
                renderer=renderer, start_time=exp.start_time,
                thresholds=self.request.thresholds,
                keep_member_files=self.request.keep_member_files,
                member_order=self.member_order,
                member_metadata=metadata,
                retain_member_diagnostics=self.request.retain_member_diagnostics)
        member_archive = getattr(self.collector, "member_archive", None)
        if member_archive is not None:
            for member, prepared in member_inputs_by_id.items():
                member_archive.expect_member_forecast(member, prepared.experiment)
        cards, model, scope = self.cards, self.memory_model, self.device_scope
        cp = self.array_module
        profiles, ordinary_admissions, ordinary_sampling = {}, {}, []
        fft_workspaces = {}
        if cards is None or model is None or scope is None:
            if cp is None:
                import cupy as cp
            from gpuwm.core.preflight import estimate_experiment, local_memory_profile_from_device
            from gpuwm.ensemble.prepared_execution import member_device_ids_for_request
            from gpuwm.ensemble.production_memory import ordinary_ensemble_memory_model
            scope = scope or (lambda device: cp.cuda.Device(device))
            if cards is None:
                ids = member_device_ids_for_request(inputs, self.request,
                    visible_count=int(cp.cuda.runtime.getDeviceCount()),
                    current_device=int(cp.cuda.runtime.getDevice()))
                readings = []
                for device in ids:
                    with scope(device):
                        if self.stochastic_provider is not None:
                            fft_workspaces[device] = self.stochastic_provider.sample_fft_workspace_bytes(
                                exp, array_module=cp,
                                other_experiments=tuple(prepared.experiment for prepared in variants
                                                        if prepared.experiment is not exp))
                        free, total = cp.cuda.runtime.memGetInfo()
                        reusable = int(cp.get_default_memory_pool().free_bytes())
                        props = cp.cuda.runtime.getDeviceProperties(device)
                        name = props["name"]
                        if isinstance(name, bytes):
                            name = name.decode()
                        readings.append(CardBudget(device, min(int(total), int(free) + reusable), int(total), str(name)))
                        ordinary_sampling.append({"device_id": device, "driver_free_bytes": int(free),
                            "own_pool_reusable_bytes": reusable, "sample_timing": "before_member_initialization",
                            "stochastic_fft_workspace_bytes": fft_workspaces.get(device, 0)})
                cards = tuple(readings)
            if model is None:
                model = {}
                for card in cards:
                    with scope(card.device_id):
                        profile = local_memory_profile_from_device(cp, device_id=card.device_id)
                        profiles[card.device_id] = profile
                        if card.device_id not in fft_workspaces and self.stochastic_provider is not None:
                            fft_workspaces[card.device_id] = self.stochastic_provider.sample_fft_workspace_bytes(exp,
                                array_module=cp, other_experiments=tuple(prepared.experiment for prepared in variants
                                                                        if prepared.experiment is not exp))
                        estimates = tuple((estimate_experiment(prepared.experiment,
                            forcing_interval_seconds=prepared.boundary_interval_seconds,
                            profile=profile, vram_gib=card.total_bytes / (1024 ** 3)), prepared.experiment)
                            for prepared in variants)
                        estimate, forecast_exp = estimates[0]
                        priced, external, margin = ordinary_ensemble_memory_model(estimate, forecast_exp, self.collector,
                            inventory_id=f"ordinary-ensemble-device-{card.device_id}",
                            stochastic_provider=self.stochastic_provider,
                            fft_workspace_bytes=fft_workspaces.get(card.device_id, 0), other_variants=estimates[1:])
                        model[card.device_id] = priced
                        ordinary_admissions[card.device_id] = (external, margin)
        cards = tuple(cards)
        resolved_request = replace(self.request, member_device_ids=tuple(card.device_id for card in cards))
        from gpuwm.ensemble.progress import (
            EnsembleProgressAdapter, MemberTerminalProgress, progress_host)
        parameters = inspect.signature(runner).parameters
        progress_key = ("observer" if "observer" in runner_options or
                        ("observer" in parameters and "progress_callback" not in parameters)
                        else "progress_callback")
        # One stop request for the whole run. Members observe it at each
        # step through this adapter; the waves own raising and joining.
        run_control = MemberRunControl()
        host = progress_host(runner_options.get(progress_key))
        progress_adapter = EnsembleProgressAdapter(host,
            member_ids=self.member_order, run_seconds=float(exp.run_seconds),
            control=run_control,
            # A run with no progress host of its own (gpuwm go, gpuwm
            # ensemble, gpuwm sim) still says where each member is.
            terminal=(None if host is not None else
                      MemberTerminalProgress(self.member_order, float(exp.run_seconds))))
        has_radar = any(radar_enabled(prepared) for prepared in member_inputs_by_id.values())
        native = self.native_executor is not None and self.source_execution is None and not has_radar
        packing = pack_members(self.request.members, tuple(cards), model, batched=native,
            reason=("qualified native member executor" if native else
                    "posted member sources retain original native initialization and streaming seals"
                    if self.source_execution is not None else
                    "simulated radar retains original member volume writers and live native consumers"
                    if has_radar else
                    "ordinary runner preserves this configuration and each member clock"))
        manifest = {"schema": "gpuwm-ensemble-run.v1", "status": "running",
                    "request": self.request.receipt(), "packing": packing.receipt(),
                    # Stated only by an engine gate that runs copies on purpose.
                    **({} if self.identical_members is None else
                       {"identical_members": self.identical_members}),
                    "members_completed": [], "member_history_files": [],
                    "member_radar_products": [],
                    "member_order": list(self.member_order),
                    "source_roster": (None if self.member_roster is None else self.member_roster.receipt()),
                    **({"posted_source_execution": self.source_execution.receipt()}
                       if self.source_execution is not None else {}),
                    "runtime_stochastic_authorities": [dict(self._stochastic_authorities[id(prepared)], member_id=member)
                        for member, prepared in member_inputs_by_id.items() if id(prepared) in self._stochastic_authorities],
                    "ordinary_memory_sampling": ordinary_sampling,
                    "ordinary_memory_inventory": [
                        (model[card.device_id] if isinstance(model, Mapping) else model).inventory(1)
                        for card in cards],
                    "shared_preparation": True,
                    "time_step_policy": "ordinary per-member clock; qualified fixed-clock native packs",
                    "streaming_input_policy": "unchanged from prepared input authority"}
        _atomic_json(out / "ensemble-run.json", manifest)
        # Probability publication belongs to the roster collector. This lock
        # protects durable run progress when independent cards complete together.
        import threading
        manifest_lock = threading.Lock()
        bootstrap_owners, bootstrap_receipts = [], []

        def ordinary_inputs(member_inputs, batch):
            """Admission changes only the original execution-only tile door."""
            from gpuwm.ensemble.prepared_execution import ordinary_member_execution_inputs
            external_margin = ordinary_admissions.get(batch.device_id)
            if external_margin is None:
                if batch.execution_mode != "ordinary_streamed_member":
                    return member_inputs, None
                supplied = model[batch.device_id] if isinstance(model, Mapping) else model
                external = tuple(component for component in supplied.components
                                 if component.category in ("products", "health", "stochastic"))
                if external:
                    external_margin = (external, supplied.allocator_margin)
            if external_margin is None:
                from gpuwm.ensemble.production_memory import ordinary_output_memory_components
                from gpuwm.ensemble.admission import AllocatorMargin
                from gpuwm.core.preflight import FORECAST_POOL_HEADROOM
                external_margin = (ordinary_output_memory_components(member_inputs.experiment, self.collector),
                    AllocatorMargin.from_fraction(str(FORECAST_POOL_HEADROOM - 1),
                        evidence="original pool headroom for additional ensemble output reservations"))
            if (batch.execution_mode == "ordinary_streamed_member" and self.stochastic_provider is not None
                    and self.stochastic_provider.enabled_for_experiment(member_inputs.experiment)):
                from gpuwm.ensemble.production_memory import ordinary_stochastic_streamed_inputs
                return ordinary_stochastic_streamed_inputs(member_inputs, batch=batch,
                    external_components=external_margin[0], allocator_margin=external_margin[1],
                    stochastic_provider=self.stochastic_provider,
                    fft_workspace_bytes=fft_workspaces.get(batch.device_id, 0),
                    device_profile=profiles.get(batch.device_id))
            return ordinary_member_execution_inputs(member_inputs, batch=batch,
                external_components=external_margin[0], allocator_margin=external_margin[1])

        # The automatic factory's first original initializer also needs the
        # cold zero-fit overlay. This wrapper is reused for its ordinary tails.
        def admitted_runner(member_inputs, *, output_directory, **options):
            from gpuwm.ensemble.runtime_context import current_capture
            capture = current_capture()
            member = self.member_order[0] if capture is None else capture.member_id
            device = cards[0].device_id if cp is None else int(cp.cuda.runtime.getDevice())
            batches = [batch for batch in packing.batches if batch.device_id == device
                       and self.member_order.index(member) in batch.member_indices]
            if not batches:
                batches = [batch for batch in packing.batches if batch.device_id == device]
            if batches:
                batch = batches[0]
            else:
                card = next(card for card in cards if card.device_id == device)
                card_model = model[device] if isinstance(model, Mapping) else model
                need = card_model.required_bytes(1)
                mode = "ordinary_member" if need <= card.available_bytes else "ordinary_streamed_member"
                batch = MemberBatch(0, device, (0,), mode,
                    need if mode == "ordinary_member" else None, card.available_bytes,
                    "cold admission for an unused original per-card bootstrap")
            member_inputs, overlay = ordinary_inputs(member_inputs, batch)
            options[progress_key] = (None if capture is None else progress_adapter.callback_for_member(member))
            result = runner(member_inputs, output_directory=output_directory, **options)
            if overlay is not None and overlay.get("changed"):
                result = asdict(result) if is_dataclass(result) else dict(result)
                result["ensemble_execution_overlay"] = overlay
            return result

        def original_card_bootstrap(*, device_id, shared_inputs, request):
            """Restore the original prepared source on its own card, no step."""
            from gpuwm.ensemble.prepared_execution import InitializedCardBootstrap, sample_initialized_card_evidence
            from gpuwm.ensemble.runtime_context import member_output_scope
            holder = {}
            path = out / ".ensemble-initialization" / f"card-{device_id:02d}"
            path.mkdir(parents=True, exist_ok=False)
            def capture(**handoff):
                from gpuwm.ensemble.production_memory import require_bootstrap_device
                require_bootstrap_device(handoff["node"], array_module=cp, device_id=device_id)
                sample = sample_initialized_card_evidence(handoff["inputs"], array_module=cp)
                if sample.card.device_id != device_id:
                    raise RuntimeError("original bootstrap initialized a different physical card")
                holder.update(handoff, evidence=sample)
                return {"status": "PASS", "ensemble_bootstrap_only": True,
                        "completed_seconds": 0.0, "forecast_steps": 0}
            options = dict(runner_options, first_products=None, ensemble_bootstrap=capture)
            options[progress_key] = None
            with scope(device_id), member_output_scope(None):
                report = admitted_runner(shared_inputs, output_directory=path, **options)
            if (not holder or not isinstance(report, dict) or report.get("status") != "PASS"
                    or report.get("forecast_steps") != 0):
                raise RuntimeError("original per-card bootstrap did not stop before its first forecast step")
            bootstrap_owners.append(holder)
            bootstrap_receipts.append({"device_id": device_id, "path": str(path.relative_to(out)),
                "forecast_steps": 0, "evidence": holder["evidence"].receipt(),
                "source": "same original prepared input; fresh original model and clock on this card"})
            return InitializedCardBootstrap(holder["inputs"], holder["node"], holder["evidence"])

        automatic = None
        if (not has_radar and self.source_execution is None and self.native_executor is None and hasattr(inputs, "execution_plan")
                and self.cards is None and self.memory_model is None):
            from gpuwm.ensemble.prepared_execution import make_automatic_prepared_executor
            automatic = make_automatic_prepared_executor(admitted_runner, request=resolved_request,
                collector=self.collector, output_directory=out, member_roster=self.member_roster,
                input_provider=(None if self.input_provider is None else
                    lambda *, member_id, **kwargs: member_inputs_by_id[member_id]), device_scope=scope,
                initialize_callback_factory=self._initialization_callback,
                card_bootstrap_factory=self.card_bootstrap_factory or original_card_bootstrap,
                array_module=cp,
                stochastic_enabled=(self.stochastic_provider is not None and
                    self.stochastic_provider.enabled_for_experiment(exp)))
        first_result = None

        def execute(batch):
            # Packing owns local slots. All source, RNG and output interfaces
            # retain the original global member identifiers.
            batch = replace(batch, member_indices=tuple(self.member_order[slot] for slot in batch.member_indices))
            try:
                return execute_admitted(batch)
            except Exception as error:
                # The error says which member it belongs to, in its own
                # sentence and in the manifest, by the original member IDs.
                name_failing_members(error, batch.member_indices, device_id=batch.device_id,
                    wave=batch.wave, execution_mode=batch.execution_mode)
                raise

        def execute_admitted(batch):
            with scope(batch.device_id):
                if batch.execution_mode == "member_batched":
                    native_options = dict(runner_options)
                    native_options[progress_key] = progress_adapter.callback_for_member(batch.member_indices[0])
                    result = self.native_executor(inputs=inputs, batch=batch,
                        request=self.request, collector=self.collector,
                        output_directory=out, input_provider=self.input_provider,
                        runner_options=native_options)
                    if result is None:
                        raise RuntimeError("admitted native pack returned no forecast receipt")
                else:
                    member = batch.member_indices[0]
                    member_inputs = member_inputs_by_id[member]
                    member_inputs, overlay = ordinary_inputs(member_inputs, batch)
                    capture = MemberOutputCapture(self.collector.submit, member,
                        member_history_required(member_inputs, self.request.keep_member_files),
                        initialize_callback=self._initialization_callback(member))
                    options = dict(runner_options)
                    options["first_products"] = None
                    options[progress_key] = progress_adapter.callback_for_member(member)
                    member_out = out / "members" / f"member-{member:04d}"
                    if first_result is not None and member == first_result.first_member_id:
                        result = first_result.report
                    else:
                        member_out.mkdir(parents=True, exist_ok=False)
                        # Every time-lagged and multi-model member ends here:
                        # its device state goes before the next member starts.
                        with finished_member_release(cp), member_output_scope(capture):
                            if self.source_execution is None:
                                result = runner(member_inputs, output_directory=member_out, **options)
                            else:
                                def forecast(actual_inputs):
                                    nonlocal overlay
                                    actual_inputs = self._configured_member_inputs(actual_inputs)
                                    authority = self._stochastic_authorities.get(id(actual_inputs))
                                    actual_inputs, overlay = ordinary_inputs(actual_inputs, batch)
                                    with manifest_lock:
                                        if authority is not None:
                                            rows = {row["member_id"]: row for row in
                                                manifest["runtime_stochastic_authorities"]}
                                            rows[member] = dict(authority, member_id=member)
                                            manifest["runtime_stochastic_authorities"] = [
                                                rows[index] for index in self.member_order if index in rows]
                                        manifest["posted_source_execution"] = self.source_execution.receipt()
                                        _atomic_json(out / "ensemble-run.json", manifest)
                                    return runner(actual_inputs, output_directory=member_out, **options)
                                result = self.source_execution.run_member(member, forecast=forecast,
                                    observer=options.get(progress_key))
                    if self.request.keep_member_files:
                        for path in sorted(member_out.glob("**/wrfout_*")):
                            match = re.match(r"wrfout_d([0-9]+)_", path.name)
                            if match is None or not path.is_file() or path.suffix == ".json":
                                continue
                            episodes = [part for part in path.parts if part.startswith("episode-")]
                            episode = int(episodes[-1].split("-", 1)[1]) if episodes else 0
                            self.collector.add_member_file(path, member_id=member,
                                grid_id=int(match.group(1)), episode=episode)
                record = asdict(result) if is_dataclass(result) else result
                if batch.execution_mode != "member_batched" and overlay is not None and overlay.get("changed"):
                    record = dict(record, ensemble_execution_overlay=overlay)
                if batch.execution_mode != "member_batched" and capture.counter_calendars:
                    record = dict(record)
                    record["ensemble_counter_observations"] = [
                        calendar.receipt(member_id=member, episode=episode)
                        for calendar in capture.counter_calendars.values()
                        for observed_member, episode in calendar._progress
                        if observed_member == member]
                if isinstance(record, dict) and record.get("status") not in (None, "PASS"):
                    raise RuntimeError("ordinary member returned a failing forecast receipt")
                radar_products = (None if batch.execution_mode == "member_batched" else
                    finish_member_radar(member_inputs, member_out, result,
                                        keep_member_files=self.request.keep_member_files))
                with manifest_lock:
                    if radar_products is not None:
                        manifest["member_radar_products"].append(dict(radar_products,
                            member_id=member, directory=str(member_out.relative_to(out))))
                        manifest["member_radar_products"].sort(key=lambda row: self.member_order.index(row["member_id"]))
                    manifest["members_completed"].extend(batch.member_indices)
                    manifest["members_completed"].sort()
                    if self.source_execution is not None:
                        manifest["posted_source_execution"] = self.source_execution.receipt()
                    _atomic_json(out / "ensemble-run.json", manifest)
                return record

        try:
            if automatic is not None:
                try:
                    # A native pack inside this call runs its own waves; they
                    # find the run's stop request through this scope.
                    with member_run_scope(run_control):
                        first_result = automatic(inputs, runner_options=runner_options)
                except Exception as error:
                    # The first member runs here on the calling thread. A
                    # native pack that failed inside it already named itself.
                    name_failing_members(error, (self.member_order[0],))
                    raise
                finally:
                    # Retained original card owners cover every native wave.
                    # Drop them before any fallback tail is initialized.
                    bootstrap_owners.clear()
                    import gc
                    gc.collect()
                manifest["automatic_admission"] = first_result.admission
                manifest["additional_card_bootstraps"] = bootstrap_receipts
            if first_result is not None and first_result.native_complete:
                manifest["members_completed"] = list(self.member_order)
                manifest["packing"] = first_result.report["admission"]["packing"]
                results = ()
                completed_records = first_result.report["member_results"]
            else:
                results = execute_member_packing(packing, execute, control=run_control)
                completed_records = [{"packing": batch.receipt(), "result": result}
                                     for batch, result in results]
            products = self.collector.finish_run()
            self.collector.require_complete()
            if self.source_execution is not None:
                manifest["posted_source_execution"] = self.source_execution.require_complete()
            files = sorted(out.glob("members/**/wrfout_*"))
            files = [path for path in files if path.is_file() and path.suffix != ".json"]
            if files and not self.request.keep_member_files:
                raise RuntimeError("aggregate-only execution unexpectedly wrote member histories")
            manifest.update(status="PASS", wall_seconds=time.perf_counter() - started,
                            completed_seconds=float(exp.run_seconds), products=products,
                            ensemble_progress=progress_adapter.receipt(),
                            member_history_files=[str(path.relative_to(out)) for path in files],
                            member_results=completed_records)
            _atomic_json(out / "ensemble-run.json", manifest)
            self.last_manifest = manifest
            self.last_output_directory = out
            _atomic_json(out / "report.json", manifest)
            _atomic_json(out / "progress.json", {"schema": "gpuwm-ensemble-progress.v1",
                "status": "PASS", "model_elapsed_seconds": manifest["completed_seconds"],
                "requested_run_seconds": float(exp.run_seconds),
                "members_completed": list(manifest["members_completed"]),
                "ensemble_manifest": "ensemble-run.json"})
            return manifest
        except BaseException as error:
            if self.source_execution is not None:
                manifest["posted_source_execution"] = self.source_execution.receipt()
            # A stop is not a member failure: the run was interrupted and
            # every member that had not finished is simply incomplete.
            stopped = isinstance(error, (KeyboardInterrupt, SystemExit))
            completed = set(manifest["members_completed"])
            manifest.update(status="interrupted" if stopped else "failed",
                            wall_seconds=time.perf_counter() - started,
                            ensemble_progress=progress_adapter.receipt(),
                            error_type=type(error).__name__, error=str(error),
                            failed_members=[] if stopped else failed_member_rows(error),
                            members_not_completed=[member for member in self.member_order
                                                   if member not in completed])
            _atomic_json(out / "ensemble-run.json", manifest)
            raise

    def completed_products(self):
        """Verify durable aggregate manifests before a door closes rendering."""
        if self.last_manifest is None or self.last_manifest["status"] != "PASS":
            raise RuntimeError("ensemble rendering has no completed forecast roster")
        from gpuwm.ensemble.radar_output import verified_radar_products
        for member in self.last_manifest.get("member_radar_products", ()):
            actual = verified_radar_products(self.last_output_directory / member["directory"])
            if actual["manifest_sha256"] != member["manifest_sha256"]:
                raise RuntimeError("completed member radar manifest changed after publication")
        for domain in self.last_manifest["products"].get("domain_manifests", ()):
            path = self.last_output_directory / domain["manifest"]
            document = json.loads(path.read_text(encoding="utf-8"))
            if any(frame["status"] not in ("complete", "unavailable") for frame in document["frames"]):
                raise RuntimeError("ensemble product manifest still contains an incomplete roster")
            for frame in document["frames"]:
                for relative in (*frame.get("products", ()), *frame.get("maps", ())):
                    if not (self.last_output_directory / relative).is_file():
                        raise RuntimeError("ensemble product manifest names a missing completed artifact")
        return self.last_manifest["products"]

    def _initialization_callback(self, member_id, *, seed=None, prepared_member=None):
        if self.stochastic_provider is None:
            return None
        from gpuwm.ensemble.seeds import member_seed
        prepared = (prepared_member if prepared_member is not None else None
                    if self.member_roster is None else self.member_roster.select((member_id,))[0])
        if prepared is None and self.source_execution is not None:
            prepared = self.source_execution.stochastic_member_binding(member_id)
        if seed is None:
            seed = (self.source_execution.member_metadata[member_id]["seed"]
                    if self.source_execution is not None else
                    member_seed(self.request.base_seed, member_id) if prepared is None else prepared.seed)
        def initialize(*, model=None, prepared_case=None, state=None, cfg=None, grid=None, clock=None):
            binding = (self.source_execution.stochastic_member_binding(member_id)
                       if self.source_execution is not None else prepared)
            if model is not None:
                self.stochastic_provider.bind_model(model=model, member_id=member_id,
                    seed=seed, prepared_member=binding)
            else:
                self.stochastic_provider.bind_state(state=state, cfg=cfg, clock=clock,
                    member_id=member_id, seed=seed, prepared_member=binding)
        return initialize

    def _configured_member_inputs(self, inputs):
        if self.stochastic_provider is None:
            return inputs
        previous = self._stochastic_input_bindings.get(id(inputs))
        if previous is not None:
            original, bound = previous
            if original is not inputs:
                raise AssertionError("stochastic input owner identity changed inside the active session")
            return bound
        from gpuwm.ensemble.stochastic_authority import configure_member_inputs
        bound, authority = configure_member_inputs(self.stochastic_provider, inputs)
        self._stochastic_input_bindings[id(inputs)] = (inputs, bound)
        self._stochastic_input_bindings[id(bound)] = (bound, bound)
        if authority is not None:
            self._stochastic_authorities[id(bound)] = authority
        return bound

    def _provide_configured_input(self, **kwargs):
        return self._configured_member_inputs(self.input_provider(**kwargs))

    def run_experiment(self, runner, experiment, case_data, output_directory, **runner_options):
        """Reuse native source preparation around the original runtime door."""
        self._require_member_inputs(experiment)
        from gpuwm.ensemble.runtime_preparation import RuntimeMemberInputs, RuntimePreparationSource
        from gpuwm.runtime import ExperimentRunSummary
        out = Path(output_directory)
        source = RuntimePreparationSource(out / ".ensemble-preparation")
        shared = RuntimeMemberInputs(experiment, case_data)
        def member_runner(member_inputs, *, output_directory, first_products=None, **options):
            if not isinstance(member_inputs, RuntimeMemberInputs):
                raise TypeError("config-driven source members require their native RuntimeMemberInputs binding")
            with source.scope(member_inputs.member_input):
                return runner(member_inputs.experiment, member_inputs.case_data,
                              output_directory, **options)
        try:
            manifest = self.run_prepared(member_runner, shared,
                output_directory=out, **runner_options)
            preparation = source.receipt()
        finally:
            source.close()
        preparation["deleted_files"] = source.receipt()["deleted_files"]
        preparation["closed"] = True
        manifest["shared_native_preparation"] = preparation
        _atomic_json(out / "ensemble-run.json", manifest)
        _atomic_json(out / "report.json", manifest)
        ensemble_manifest = out / "ensemble-run.json"
        return ExperimentRunSummary(
            wrfout_paths=tuple(out / relative for relative in manifest["member_history_files"]),
            completed_seconds=manifest["completed_seconds"], nan_free=True,
            ensemble_manifest=ensemble_manifest,
            ensemble_manifest_sha256=hashlib.sha256(ensemble_manifest.read_bytes()).hexdigest())


def run_prepared_ensemble(inputs, *, runner, request, output_directory,
                          member_roster=None, input_provider=None, stochastic_provider=None,
                          source_execution=None, session_options=None, **options):
    """Run a prepared ensemble through an original prepared forecast door."""
    return PreparedEnsembleSession(request, output_directory=output_directory,
        member_roster=member_roster, input_provider=input_provider,
        stochastic_provider=stochastic_provider, source_execution=source_execution,
        **(session_options or {})).run_prepared(
        runner, inputs, **options)
