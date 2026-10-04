"""Native mapped member initialization from a checked shared source head."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import uuid


def shared_input_plan(context, posted_source, *, mapping, composition, primary,
                      supplements, provenance, contributing):
    """Hold requested source metadata to the already decoded ordinary plan.

    The source's decoder digest remains in that plan. A consumer never
    resolves or opens a decoder executable because it decodes no raw input.
    """
    from gpuwm.ensemble.physical_store import digest_file
    from gpuwm.ingest.boundary_stream import as_posted_placeholder, input_plan_sha256
    from gpuwm.mapped_authoring import manifest_row_path
    context.verify()
    plan = context.source_plan
    manifest = plan["manifest"]
    posted = context.prepared_head["basis"]["as_posted"]
    if (posted_source.physical_trajectory() != context.trajectory
            or list(posted_source.leads) != posted["forcing_leads"]
            or tuple(posted_source.valid_times) != tuple(value.replace(tzinfo=None)
                                                       for value in context.physical_stream.times)
            or posted_source.posted.route_table_sha256() != plan["route_table_sha256"]):
        raise ValueError("mapped member posting schedule differs from its pinned ordinary source")
    if (digest_file(mapping) != manifest["mapping_sha256"]
            or digest_file(composition) != manifest["composition_sha256"]):
        raise ValueError("mapped member mapping or composition differs from the captured source plan")
    def paths(values):
        return [manifest_row_path(path, posted_source.input_manifest) for path in values]
    def rows(value):
        return [value] if isinstance(value, dict) else value
    if paths(primary) != [row["path"] for row in manifest["primary_files"]]:
        raise ValueError("mapped member primary input inventory differs from its source plan")
    if set(supplements) != set(manifest["supplements"]) or any(
            paths(values) != [row["path"] for row in rows(manifest["supplements"][role])]
            for role, values in supplements.items()):
        raise ValueError("mapped member supplement inventory differs from its source plan")
    if set(provenance) != set(manifest["provenance"]) or any(
            digest_file(path) != manifest["provenance"][role]["sha256"]
            or Path(path).stat().st_size != manifest["provenance"][role]["bytes"]
            for role, path in provenance.items()):
        raise ValueError("mapped member provenance bytes differ from its source plan")
    contract = json.loads(Path(composition).read_bytes())
    bindings = contract.get("field_sources") or {}
    expected_roles = {row["mapping_role"] for row in bindings.values()}
    if set(contributing) - expected_roles:
        raise ValueError("mapped member supplied an unbound contributing mapping")
    for row in bindings.values():
        if row["mapping_role"] in contributing and digest_file(contributing[row["mapping_role"]]) != row["mapping_sha256"]:
            raise ValueError("mapped member contributing mapping differs from its captured composition")
    fixed = sorted({manifest_row_path(path, posted_source.input_manifest) for path in posted_source.fixed})
    if fixed != posted["fixed_rows"]:
        raise ValueError("mapped member fixed source rows differ from its captured input plan")
    return {"plan": deepcopy(plan), "fixed_rows": fixed,
        "route_table_sha256": plan["route_table_sha256"],
        "placeholder": as_posted_placeholder(input_plan_sha256(plan))}


def prepare_posted_mapped_member(provider, member_index, *, posted_plan,
        mapping_contract, mapping, composition, experiment_config,
        wps_namelist, output_root, preprocess, preprocess_workers,
        run_control_before, stock_wrf_export, physics_selection,
        source_adapter, case_policy, water_overlay_binding,
        static_input=None, static_receipt=None):
    """Reuse verified geometry and surface while rebuilding changed atmosphere.

    The ordinary producer owns raw decoding, horizontal mapping, geographic
    fields and their posted failure/seal lifecycle. Every member still uses
    the existing real initializer and boundary writer at every forcing knot.
    """
    from gpuwm.ensemble.posted_native import (
        checked_source_inputs, shared_surface, copy_common_artifacts, writer_source_wait)
    from gpuwm.ensemble.physical_store import digest_file, physical_static_identity
    from gpuwm.ensemble.posted_physical import bind_posted_physical_input, _link_or_copy
    from gpuwm.ensemble.mapped_physical_contract import (
        mapped_physical_field_contract, require_mapped_physical_field_contract)
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.boundary_fields import mapping_boundary_species
    from gpuwm.ingest.boundary_stream import (
        PreparedTreeWriter, prepared_head_urban_columns, producer_device_bytes)
    from gpuwm.ingest.lateral_bc import StateBoundaryFrames
    from gpuwm.ingest.preprocess_backend import (
        admit_preparation, preprocess_identity, release_backend_memory)
    from gpuwm.ingest.preparation_price import price_forcing_preparation
    from gpuwm.ingest.real import initialize_real
    from gpuwm.moisture_floor_receipt import moisture_floor_proof_entry
    from gpuwm.native_wrf_contract import native_geometry_contract, native_static_export_fields
    from gpuwm.wrf_direct import (export_prepared_wrf, StockWrfExportUnsupported,
        stock_wrf_export_not_requested, stock_wrf_export_refused)
    from gpuwm.mapped_direct import _file_receipt, _canonical, _AS_POSTED_SEAL_KEYS

    started = time.perf_counter()
    context = provider.source_context(member_index)
    output_root = Path(output_root).resolve()
    for protected in (context.prepared_root, context.physical_stream.root, provider.root):
        if (output_root == protected or output_root.is_relative_to(protected)
                or protected.is_relative_to(output_root)):
            raise ValueError("mapped member output must be separate from its pinned source and provider")
    if context.source_plan != posted_plan["plan"]:
        raise ValueError("mapped member acquisition differs from its pinned ordinary source plan")
    checked = checked_source_inputs(context, source=context.trajectory.source,
        experiment_config=experiment_config, wps_namelist=wps_namelist)
    if static_input is not None:
        requested = json.loads(Path(static_receipt).read_bytes())
        captured = checked.proof["execution_inputs"].get("root_static_receipt")
        # High-resolution overlays retain their baseline receipt. An exact
        # final source cache is also a valid way to name its shared statics.
        candidates = [dict(checked.geometry_receipt)]
        while isinstance(captured, dict):
            candidates.append(captured)
            captured = captured.get("baseline")
        if (requested not in candidates
                or digest_file(static_input) != requested.get("cache", {}).get("sha256")):
            raise ValueError("mapped member static request differs from its captured source authority")
    exp, grid, static = checked.experiment, checked.grid, checked.static
    cfg = exp.root.run
    head = context.prepared_head
    ordinary_source = deepcopy(head["basis"]["cache"]["identity"]["source_identity"])
    if (ordinary_source.get("adapter") != source_adapter
            or ordinary_source.get("mapping_sha256") != digest_file(mapping)
            or ordinary_source.get("composition_sha256") != digest_file(composition)
            or ordinary_source.get("preparation_case_policy") != case_policy
            or ordinary_source.get("water_temperature_overlay") != water_overlay_binding):
        raise ValueError("mapped member controls differ from the checked ordinary source")
    recorded_decoders = checked.proof["execution_inputs"]["decoders"]
    times = tuple(datetime.fromisoformat(value).replace(tzinfo=None)
                  for value in head["basis"]["proof_head"]["forcing_times"])
    if tuple(value.replace(tzinfo=None) for value in context.physical_stream.times) != times:
        raise ValueError("mapped member physical knots differ from the complete ordinary boundary clock")
    geometry = native_geometry_contract(grid, cfg)
    static_identity = physical_static_identity(
        native_static_export_fields(static, grid), checked.landuse_identity)
    expected_fields = mapped_physical_field_contract(geometry,
        mapping_path=mapping, composition_path=composition,
        source_identity=ordinary_source,
        extra_evidence={"raw_input_plan": head["basis"]["as_posted"]["input_plan_sha256"],
            **{"native_decoder_" + name: row["sha256"] for name, row in recorded_decoders.items()}})

    class PhysicalTimes:
        valid_times = times

        def __len__(self):
            return len(times)

        def __getitem__(self, index):
            return context.physical_stream.require(context.physical_stream.times[index])[0].read(0)

    preprocess = admit_preparation(preprocess,
        lambda: price_forcing_preparation("mapped", exp, PhysicalTimes(),
            boundary_species=mapping_boundary_species(mapping_contract)),
        workers=preprocess_workers)
    source_identity = deepcopy(ordinary_source)
    source_identity["preprocessing"] = preprocess_identity(preprocess.receipt())
    bindings = {}
    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
        etac=cfg.etac, eta_levels=exp.vertical.eta_levels)
    map_factors = (grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v())
    coriolis = grid.coriolis_m()
    rotation = grid.rotation_m()
    forcing = StateBoundaryFrames(spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)

    def build(index):
        frame, receipt = provider.resolve(member_index, times[index])
        require_mapped_physical_field_contract(frame, expected_fields)
        met = frame.read(0)
        bindings[index] = bind_posted_physical_input(frame, receipt,
            provider_plan=provider.plan, member_index=member_index,
            valid_time=times[index], grid=grid, cfg=cfg,
            source_identity=ordinary_source, input_plan=context.source_plan,
            static_identity=static_identity)
        initialized = initialize_real(met, cfg, coord, static["HGT_M"], grid=grid,
            landmask=static["LANDMASK"], p_top=exp.vertical.p_top,
            sfcp_to_sfcp=case_policy["sfcp_to_sfcp"], preprocess_backend=preprocess,
            state_backend="preprocess", boundary_only=index != 0,
            boundary_species=mapping_boundary_species(mapping_contract))
        initialized.state.set_map_coriolis(*map_factors, *coriolis,
            sina=rotation[0], cosa=rotation[1])
        return met, initialized

    output_root = Path(output_root)
    staging = output_root.with_name(".tmp-" + uuid.uuid4().hex[:8])
    staging.mkdir(exist_ok=False)
    writer = None
    try:
        copy_common_artifacts(context, checked, staging)
        evidence = staging / "source-evidence"
        evidence.mkdir(exist_ok=True)
        # Preflight has checked these portable authority documents. Pin each
        # copy again to keep a concurrent file change from being consumed.
        source_evidence = context.prepared_root / "source-evidence"
        for role, value in sorted(checked.authority_paths.items()):
            path = Path(value)
            if path.is_relative_to(source_evidence) and path.name != "input-manifest.json":
                before = checked.file_sha256[role]
                if digest_file(path) != before:
                    raise ValueError("shared mapped authority changed after source preflight")
                target = evidence / path.name
                if target.exists():
                    if digest_file(target) != before:
                        raise ValueError("shared mapped evidence copy differs from source")
                else:
                    _link_or_copy(path, target)
                if digest_file(path) != before or digest_file(target) != before:
                    raise ValueError("shared mapped evidence changed while it was copied")
        context.verify()
        initial_met, initial_result = build(0)
        base_met = context.physical_stream.require(context.physical_stream.times[0])[0].read(0)
        surface = shared_surface(checked, base_met=base_met, member_met=initial_met)
        del base_met
        source_identity["ensemble_posted_physical_input"] = bindings[0]
        identity = deepcopy(head["basis"]["cache"]["identity"])
        identity["source_identity"] = source_identity
        proof_head = deepcopy(head["basis"]["proof_head"])
        proof_head["stock_wrf_export"] = stock_wrf_export
        proof_head["preprocessing"] = preprocess.receipt()
        proof_head.update(moisture_floor_proof_entry(initial_result,
            when_unrecorded="native member initializer returned no recorded moisture-floor fields"))
        proof_head["execution_inputs"].update(run_control_before)
        # Source static and soil receipts stay unchanged and remain bound to
        # the original arrays. Changed atmosphere owns its new real receipt.
        writer = PreparedTreeWriter(staging=staging, output_root=output_root, identity=identity)
        provider.set_wait_observer(writer_source_wait(writer))
        writer.bind_physical_receipts(bindings)
        writer.admit(experiment=exp, backend=str(preprocess.receipt()["backend"]),
            device_bytes=producer_device_bytes(str(preprocess.receipt()["backend"])),
            source=mapping_contract, urban_columns=prepared_head_urban_columns(exp, static))
        forcing.add_state(initial_result.state, index=0)
        writer.write_head(initial_result=initial_result, met=initial_met, surface=surface.fields,
            metadata=deepcopy(head["basis"]["cache"]["metadata"]["user"]),
            lbc=deepcopy(head["basis"]["cache"]["lbc"]), proof_head=proof_head,
            forcing=forcing, as_posted=deepcopy(head["basis"]["as_posted"]),
            ensemble_physical={"provider_plan": provider.plan, "member_index": member_index,
                               "initial_receipt": bindings[0]})
        del initial_met, initial_result, surface
        release_backend_memory(preprocess)
        for index in range(1, len(times)):
            built = time.perf_counter()
            met, initialized = build(index)
            forcing.add_state(initialized.state, index=index)
            del met, initialized
            release_backend_memory(preprocess)
            writer.note_build_seconds(time.perf_counter() - built)
            marker = context.require_interval(index - 1)
            interval = forcing.interval(index - 1, times)
            writer.write_segment(index - 1, interval, relay_marker=marker)
            del interval
            forcing.release(index - 1)
        capsule = context.capture_seal()
        posted = head["basis"]["as_posted"]
        manifest_bytes = capsule["artifacts"][posted["manifest_path"]].encode("utf-8")
        (writer.root / posted["manifest_path"]).write_bytes(manifest_bytes)
        sealed_proof = json.loads(capsule["artifacts"]["proof.json"])
        sealed_header = json.loads(capsule["artifacts"]["prepared-cache/header.json"])
        sealed_identity = deepcopy(sealed_header["identity"])
        sealed_identity["source_identity"]["preprocessing"] = source_identity["preprocessing"]
        sealed_identity["source_identity"]["ensemble_posted_physical_input"] = bindings[0]
        records = json.loads(capsule["artifacts"]["boundary-stream/posted-leads.json"])
        writer.write_posted_leads({int(key): row["marker"] for key, row in records["leads"].items()},
            route_table_sha256=records["route_table_sha256"])
        cache_receipt = dict(writer.seal_cache(identity=sealed_identity,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            physical_provider_seal=provider.seal()))
        cache_receipt["path"] = "prepared-cache"
        export_schema = "gpuwm-native-direct-wrf-export-v3"
        if stock_wrf_export == "off":
            export = stock_wrf_export_not_requested(schema=export_schema)
        else:
            try:
                export = export_prepared_wrf(writer.cache_path,
                    writer.root / "native-static.npz", writer.root / "geometry-receipt.json",
                    writer.root / "wrf-native-input", valid_time=times[0],
                    boundary_interval_seconds=checked.boundary_interval_seconds,
                    experiment_config_suite=True,
                    expert_acknowledgements=tuple(physics_selection["acknowledgements"]),
                    acknowledgement_provenance=physics_selection["acknowledgement_provenance"])
            except StockWrfExportUnsupported as error:
                if stock_wrf_export == "required":
                    raise
                export = stock_wrf_export_refused(error, schema=export_schema)
        for key, row in run_control_before.items():
            if _file_receipt(Path(row["path"])) != row:
                raise ValueError("mapped member run-control bytes changed during preparation")
        context.verify()
        proof = {**proof_head,
            **{key: sealed_proof[key] for key in _AS_POSTED_SEAL_KEYS if key in sealed_proof},
            "posting": {"as_posted": True, "waits": [], "leads_late": []},
            "prepared_cache": cache_receipt, "export": export,
            "boundary_stream": writer.boundary_stream_proof(),
            "timing_seconds": {"total": time.perf_counter() - started}}
        proof["proof_content_sha256"] = hashlib.sha256(_canonical(proof).encode()).hexdigest()
        writer.publish(proof)
        return proof
    except BaseException as error:
        if writer is not None:
            writer.fail(error)
        raise
