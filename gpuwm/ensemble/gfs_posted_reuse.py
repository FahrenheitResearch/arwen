"""Native GFS member initialization using the checked common source head."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import uuid


def prepare_posted_gfs_member(provider, member_index, *, input_plan,
        experiment_config, wps_namelist, output_root, preprocess,
        preprocess_workers, physics_profile, expert_acknowledgements,
        physics_selection, case_policy, water_overlay_binding,
        stock_wrf_export, implementation_sha256, git_source_identity,
        input_manifest, progress):
    """Share raw decoding, statics and surface; initialize each atmosphere.

    All raw source and final manifest work belongs to the ordinary producer.
    A member's changed meteorology still uses the existing real initializer
    and boundary writer at every initial and boundary valid time.
    """
    from gpuwm.ensemble.posted_native import (
        checked_source_inputs, copy_common_artifacts, relay_source_segment, shared_surface,
        writer_source_wait,
    )
    from gpuwm.ensemble.posted_physical import bind_posted_physical_input, _link_or_copy
    from gpuwm.ensemble.physical_store import digest_file, physical_static_identity
    from gpuwm.ensemble.gfs_physical_contract import require_native_gfs_field_contract
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.ingest.boundary_stream import (
        PreparedTreeWriter, prepared_head_urban_columns, producer_device_bytes,
    )
    from gpuwm.ingest.preparation_price import price_forcing_preparation
    from gpuwm.ingest.preprocess_backend import (
        admit_preparation, preprocess_identity, release_backend_memory,
    )
    from gpuwm.moisture_floor_receipt import moisture_floor_proof_entry
    from gpuwm.native_wrf_contract import (
        NATIVE_LANDUSE_IDENTITY, native_geometry_contract,
    )
    from gpuwm import gfs_direct

    started = time.perf_counter()
    context = provider.source_context(member_index)
    output_root = Path(output_root).resolve()
    for protected in (context.prepared_root, context.physical_stream.root, provider.root):
        if (output_root == protected or output_root.is_relative_to(protected)
                or protected.is_relative_to(output_root)):
            raise ValueError("GFS member output must be separate from its checked source and provider")
    if input_plan != context.source_plan:
        raise ValueError("GFS member acquisition differs from its pinned ordinary source plan")
    checked = checked_source_inputs(context, source=context.trajectory.source,
        experiment_config=experiment_config, wps_namelist=wps_namelist,
        physics_profile=physics_profile, expert_acknowledgements=expert_acknowledgements)
    exp, grid, static = checked.experiment, checked.grid, checked.static
    cfg = exp.root.run
    head = context.prepared_head
    ordinary_source = deepcopy(head["basis"]["cache"]["identity"]["source_identity"])
    if (ordinary_source.get("preparation_case_policy") != case_policy
            or ordinary_source.get("water_temperature_overlay") != water_overlay_binding):
        raise ValueError("GFS member preparation controls differ from its ordinary source head")
    times = tuple(datetime.fromisoformat(value).replace(tzinfo=None)
                  for value in head["basis"]["proof_head"]["forcing_times"])
    if tuple(value.replace(tzinfo=None) for value in context.physical_stream.times) != times:
        raise ValueError("GFS physical knots differ from the original native forcing clock")
    geometry = native_geometry_contract(grid, cfg)
    static_identity = physical_static_identity(gfs_direct.native_static_export_fields(static, grid), NATIVE_LANDUSE_IDENTITY)
    control_files = {Path(experiment_config): digest_file(experiment_config), Path(wps_namelist): digest_file(wps_namelist)}

    class PhysicalTimes:
        valid_times = times

        def __len__(self):
            return len(times)

        def __getitem__(self, index):
            return context.physical_stream.require(context.physical_stream.times[index])[0].read(0)

    preprocess = admit_preparation(preprocess,
        lambda: price_forcing_preparation("gfs", exp, PhysicalTimes()), workers=preprocess_workers)
    preprocessing = preprocess.receipt()
    source_identity = {**ordinary_source, "preprocessing": preprocess_identity(preprocessing),
                       "implementation_sha256": implementation_sha256, "git_source_identity": git_source_identity}
    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
                                eta_levels=exp.vertical.eta_levels)
    bindings = {}
    forcing = gfs_direct.StateBoundaryFrames(spec_bdy_width=cfg.spec_bdy_width,
                                   spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)

    def build(index):
        frame, receipt = provider.resolve(member_index, times[index])
        require_native_gfs_field_contract(frame.require_field_contract(), geometry)
        met = frame.read(0)
        bindings[index] = bind_posted_physical_input(frame, receipt,
            provider_plan=provider.plan, member_index=member_index, valid_time=times[index],
            grid=grid, cfg=cfg, source_identity=ordinary_source, input_plan=context.source_plan,
            static_identity=static_identity)
        initialized = gfs_direct.initialize_real(met, cfg, coord, static["HGT_M"], grid=grid,
            landmask=static["LANDMASK"], p_top=exp.vertical.p_top,
            sfcp_to_sfcp=case_policy["sfcp_to_sfcp"], preprocess_backend=preprocess,
            state_backend="preprocess", boundary_only=index != 0)
        initialized.state.set_map_coriolis(static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
            static["F"], static["E"], sina=static["SINALPHA"], cosa=static["COSALPHA"])
        return met, initialized

    staging = output_root.with_name(".tmp-"+uuid.uuid4().hex[:8])
    staging.mkdir(exist_ok=False)
    writer = None
    try:
        common = copy_common_artifacts(context, checked, staging)
        progress.enter("initialize_all_times", forcing_times=len(times))
        initialize_started = time.perf_counter()
        initial_met, initial_result = build(0)
        base_met = context.physical_stream.require(context.physical_stream.times[0])[0].read(0)
        surface = shared_surface(checked, base_met=base_met, member_met=initial_met)
        del base_met
        source_identity["ensemble_posted_physical_input"] = bindings[0]
        identity = deepcopy(head["basis"]["cache"]["identity"])
        identity["source_identity"] = source_identity
        proof_head = deepcopy(head["basis"]["proof_head"])
        proof_head.update(implementation_sha256=implementation_sha256, git_source_identity=git_source_identity,
            preprocessing=preprocessing, preprocessing_receipt_sha256=hashlib.sha256(json.dumps(
                preprocessing, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            stock_wrf_export="optional" if stock_wrf_export else "off", physics=physics_selection)
        proof_head.update(moisture_floor_proof_entry(initial_result,
            when_unrecorded="native member initializer returned no moisture-floor record"))
        writer = PreparedTreeWriter(staging=staging, output_root=output_root, identity=identity)
        provider.set_wait_observer(writer_source_wait(writer))
        writer.bind_physical_receipts(bindings)
        writer.admit(experiment=exp, backend=str(preprocessing["backend"]),
            device_bytes=producer_device_bytes(str(preprocessing["backend"])),
            urban_columns=prepared_head_urban_columns(exp, static))
        forcing.add_state(initial_result.state, index=0)
        metadata = deepcopy(head["basis"]["cache"]["metadata"]["user"])
        metadata["preprocessing"] = preprocess_identity(preprocessing)
        progress.enter("write_prepared_cache")
        writer.write_head(initial_result=initial_result, met=initial_met, surface=surface.fields,
            metadata=metadata, lbc=deepcopy(head["basis"]["cache"]["lbc"]), proof_head=proof_head,
            forcing=forcing, as_posted=deepcopy(head["basis"]["as_posted"]),
            ensemble_physical={"provider_plan": provider.plan, "member_index": member_index,
                               "initial_receipt": bindings[0]})
        del initial_met, initial_result, surface
        release_backend_memory(preprocess)
        progress.enter("initialize_all_times", forcing_times=len(times))
        for index in range(1, len(times)):
            built = time.perf_counter()
            met, initialized = build(index)
            forcing.add_state(initialized.state, index=index)
            del met, initialized
            release_backend_memory(preprocess)
            writer.note_build_seconds(time.perf_counter()-built)
            interval = forcing.interval(index-1, times)
            relay_source_segment(context, index-1, writer, interval)
            del interval
            forcing.release(index-1)
        initialize_seconds = time.perf_counter()-initialize_started
        progress.enter("write_prepared_cache")
        capsule = context.capture_seal()
        posted = head["basis"]["as_posted"]
        manifest_bytes = capsule["artifacts"][posted["manifest_path"]].encode("utf-8")
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        (writer.root/posted["manifest_path"]).write_bytes(manifest_bytes)
        # Preserve the normal caller-facing manifest location without ever
        # rewriting a source authority the ordinary producer already owns.
        manifest_path = Path(input_manifest)
        if manifest_path.exists():
            if manifest_path.read_bytes() != manifest_bytes:
                raise ValueError("GFS member manifest destination holds another source authority")
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("xb") as output:
                output.write(manifest_bytes)
        sealed_proof = json.loads(capsule["artifacts"]["proof.json"])
        sealed_header = json.loads(capsule["artifacts"]["prepared-cache/header.json"])
        records = json.loads(capsule["artifacts"]["boundary-stream/posted-leads.json"])
        sealed_identity = deepcopy(sealed_header["identity"])
        sealed_identity["source_identity"].update(
            preprocessing=source_identity["preprocessing"], implementation_sha256=implementation_sha256,
            git_source_identity=git_source_identity, ensemble_posted_physical_input=bindings[0])
        writer.write_posted_leads({int(key): row["marker"] for key, row in records["leads"].items()},
                                 route_table_sha256=records["route_table_sha256"])
        for name in ("decoder-gate.tsv", "decoder-inventory.tsv", "decoder-sha256.tsv"):
            path = context.prepared_root/name
            observed = digest_file(path)
            _link_or_copy(path, writer.root/name)
            if digest_file(writer.root/name) != observed:
                raise ValueError("GFS member decoder receipt changed while sharing source evidence")
        cache_receipt = dict(writer.seal_cache(identity=sealed_identity, manifest_sha256=manifest_digest,
                                               physical_provider_seal=provider.seal()))
        cache_receipt["path"] = "prepared-cache"
        progress.enter("direct_wrf_export")
        export, _refusal = gfs_direct._single_domain_stock_export(writer.cache_path,
            writer.root/"native-static.npz", writer.root/"geometry-receipt.json", writer.root/"wrf-native-input",
            valid_time=times[0], boundary_interval_seconds=checked.boundary_interval_seconds,
            physics_selection=physics_selection, stock_wrf_export=stock_wrf_export)
        for path, expected in control_files.items():
            if digest_file(path) != expected:
                raise ValueError("GFS member experiment or WPS authority changed during preparation")
        context.verify()
        artifacts = {
            "source_manifest": {"path": posted["manifest_path"], "bytes": len(manifest_bytes), "sha256": manifest_digest},
            "static_cache": {"path": "native-static.npz", "bytes": (writer.root/"native-static.npz").stat().st_size,
                             "sha256": common["static"]["sha256"]},
            "geometry_receipt": {"path": "geometry-receipt.json", "bytes": (writer.root/"geometry-receipt.json").stat().st_size,
                                 "sha256": common["geometry"]["sha256"]},
            "prepared_cache": {"path": "prepared-cache", "content_sha256": cache_receipt["content_sha256"],
                               "payload_bytes": cache_receipt["payload_bytes"]},
            "wrf_files": {name: {"path": "wrf-native-input/"+name, **details}
                          for name, details in (export.get("files") or {}).items()},
        }
        proof = {**proof_head,
            **{key: sealed_proof[key] for key in gfs_direct._AS_POSTED_SEAL_KEYS if key in sealed_proof},
            "posting": {"as_posted": True, "waits": [], "leads_late": []},
            "initialization_artifacts": artifacts, "prepared_cache": cache_receipt,
            "export": export, "boundary_stream": writer.boundary_stream_proof(),
            "timing_seconds": {"initialize_all_times": initialize_seconds,
                               "total": time.perf_counter()-started}}
        progress.enter("publish")
        writer.publish(proof)
        return proof
    except BaseException as error:
        if writer is not None:
            writer.fail(error)
        raise
