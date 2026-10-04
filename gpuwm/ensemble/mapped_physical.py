"""Shared native horizontal preparation for a roster of mapped sources."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from gpuwm.ensemble.physical_store import NativePhysicalStore, digest_file, physical_static_identity


class MappedPhysicalPreparation:
    """Load target geography once and map each verified source trajectory.

    This prepares physical fields only. A member's real initialization and
    boundary extraction still run through the ordinary native preparation
    door after any declared source anomaly operation.
    """
    def __init__(self, *, experiment_config, wps_namelist, static_input,
                 static_receipt, cpu_bridge=None, workers=1):
        from gpuwm.case_data import optional_case_data_from_config, preparation_case_policy
        from gpuwm.experiment import load_experiment
        from gpuwm.ingest.preparation_setup import PreparationSetup
        from gpuwm.ingest.preprocess_backend import ParallelCpuPreprocessBackend
        from gpuwm.ingest.water_overlay import load_bound_water_overlay
        from gpuwm.ingest.water_temperature import WaterTemperatureStatics
        from gpuwm.native_wrf_contract import (NATIVE_LANDUSE_IDENTITY, native_geometry_contract,
            validate_native_lambert_contract, verify_native_static_receipt, load_native_static_cache)
        from gpuwm.static.highres_production import (
            apply_prepared_highres, load_static_highres, static_highres_identity)
        from gpuwm.vertical_adaptation import adapt_experiment_for_statics
        self.config_path = Path(experiment_config).resolve()
        self.wps_path = Path(wps_namelist).resolve()
        static_input = Path(static_input).resolve()
        static_receipt = Path(static_receipt).resolve()
        self.static_input_path, self.static_receipt_path = static_input, static_receipt
        self.cpu_bridge = cpu_bridge
        self.experiment = load_experiment(self.config_path)
        if len(self.experiment.domains) != 1:
            raise ValueError("a shared physical-source context needs one target domain; prepare each domain's own context")
        self.cfg = self.experiment.root.run
        self.grid = validate_native_lambert_contract(self.experiment, self.wps_path, source_name="mapped")
        receipt = verify_native_static_receipt(static_receipt, static_input, self.grid, self.cfg)
        self.static = load_native_static_cache(static_input, self.grid, self.cfg.ny, self.cfg.nx)
        self.controls = {str(path):digest_file(path) for path in
                         (self.config_path,self.wps_path,static_input,static_receipt)}
        case_data = optional_case_data_from_config(self.config_path)
        self.case_policy = preparation_case_policy(case_data)
        self.overlay, self.overlay_binding = load_bound_water_overlay(
            None if case_data is None else case_data.water_temperature_overlay)
        highres = load_static_highres(self.config_path)
        self.highres_identity = None if highres is None else static_highres_identity(highres)
        self.static, self.static_receipt = apply_prepared_highres(
            self.static, self.grid, config=highres, domain_id=1,
            case_date=self.experiment.start_time.date(),
            landuse_attrs=NATIVE_LANDUSE_IDENTITY, baseline_receipt=receipt)
        self.experiment, self.vertical_adaptation = adapt_experiment_for_statics(
            self.experiment, (self.grid,), root_terrain=self.static["HGT_M"],
            static_catalog=None, static_highres=highres)
        self.cfg = self.experiment.root.run
        self.geometry = native_geometry_contract(self.grid, self.cfg)
        self.static_identity = physical_static_identity(self.static, NATIVE_LANDUSE_IDENTITY)
        self.water_statics = WaterTemperatureStatics.for_route(
            route="mapped", policy=self.case_policy["water_temperature_policy"],
            landmask=self.static["LANDMASK"], lu_index=self.static["LU_INDEX"],
            landuse_attrs=NATIVE_LANDUSE_IDENTITY)
        self.workers = workers
        self.backend = ParallelCpuPreprocessBackend(bridge=cpu_bridge, workers=workers)
        self.setup = PreparationSetup()
        self.setup.activate()

    def close(self):
        self.setup.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def capture_posted(self, fetched_root, output_root, *, prepared_root, geog_root,
                       stock_wrf_export=False):
        """Publish physical knots through the ordinary posted native producer.

        The caller may run this method in its existing preparation worker.
        Its ordinary prepared head and each physical ready marker become
        visible while future raw leads are still pending. The ordinary
        producer owns all waits, failures, static validation and final seal.
        """
        from gpuwm.prep_handoff import posted_preparation_arguments_from_directory
        from gpuwm.source_cli import _parser, main, preparation_runners
        from gpuwm.source_adapters import get_source_adapter
        from gpuwm.ensemble.posted_physical import PostedPhysicalStream
        argv = posted_preparation_arguments_from_directory(fetched_root)
        args = _parser().parse_args(argv)
        # Whether a source decodes through mapping and composition inputs
        # is its preparation row, never the row's name.
        runner = preparation_runners().get(get_source_adapter(args.source).runner)
        if runner is None or not runner.composition_inputs:
            raise ValueError("posted mapped capture requires a mapped composition acquisition")
        if args.as_posted is None:
            raise ValueError("posted capture requires the acquisition's ordinary as-posted handoff")
        if any(digest_file(path) != expected for path, expected in self.controls.items()):
            raise ValueError("shared physical preparation controls changed before posted capture")
        paths = {"--experiment-config": self.config_path, "--wps-namelist": self.wps_path,
                 "--static-input": self.static_input_path, "--static-receipt": self.static_receipt_path,
                 "--geog-root": Path(geog_root), "--output-root": Path(prepared_root),
                 "--physical-output-store": Path(output_root)}
        if self.cpu_bridge is not None:
            paths["--cpu-preprocess-bridge"] = Path(self.cpu_bridge)
        for flag, path in paths.items():
            if flag in argv:
                raise ValueError(f"posted capture handoff already binds caller control {flag}")
            argv.extend((flag, str(path)))
        argv.extend(("--preprocess-backend", "cpu", "--preprocess-workers", str(self.workers)))
        if not stock_wrf_export:
            argv.append("--no-stock-wrf-export")
        if main(argv) != 0:
            raise RuntimeError("ordinary mapped posted capture failed; inspect its retained preparation log")
        if any(digest_file(path) != expected for path, expected in self.controls.items()):
            raise ValueError("shared physical preparation controls changed during posted capture")
        stream = PostedPhysicalStream(output_root)
        from gpuwm.ensemble.posted_physical import capture_source_seal
        return {"physical_head_sha256": stream.head_sha256,
                "source_seal": capture_source_seal(prepared_root, stream)}

    def capture(self, fetched_root, output_root):
        """Consume the same verified acquisition handoff as ordinary prep."""
        from gpuwm.ingest.horiz import interpolate_era5_to_lambert
        from gpuwm.ingest.cg_topo import RootTerrainBlend
        from gpuwm.ingest.water_overlay import overlay_snapshot_sequence, verify_overlay_sequence
        from gpuwm.mapped_authoring import author_input_manifest
        from gpuwm.mapped_composition import (decode_composed_source, mapped_composition_receipt,
            composition_receipt_identity_sha256)
        from gpuwm.mapped_direct import _forcing_series, _forcing_valid_times
        from gpuwm.mapped_source import read_input_list
        from gpuwm.prep_handoff import preparation_arguments
        from gpuwm.source_adapters import get_source_adapter
        from gpuwm.source_cli import (_parser, _apply_packaged_profile, _role_bindings,
                                      preparation_runners)
        started = time.perf_counter()
        # A source owns its one-time root blend. Never let one donor's terrain
        # become the next donor's input to that operation in this shared context.
        static = dict(self.static)
        terrain_blend = RootTerrainBlend(self.experiment, static, route="mapped")
        root = Path(fetched_root).resolve()
        handoff = json.loads((root/"prep-arguments.json").read_text(encoding="utf-8"))
        args = _parser().parse_args(preparation_arguments(handoff))
        adapter = get_source_adapter(args.source)
        runner = preparation_runners().get(adapter.runner)
        if runner is None or not runner.composition_inputs:
            raise ValueError("this source uses a native preparation runner; capture its physical fields through that runner")
        if adapter.packaged_profile:
            errors = _apply_packaged_profile(args, adapter, "ensemble physical preparation")
            if errors:
                raise ValueError("; ".join(errors))
        primary = tuple(read_input_list(args.input_list))
        supplements = _role_bindings(args.supplement or (), multiple=True)
        provenance = _role_bindings(args.provenance or (), multiple=False)
        contributing = _role_bindings(args.contributing_mapping or (), multiple=False)
        manifest = Path(args.source_sha256s or args.author_input_manifest).resolve()
        if not manifest.exists():
            author_input_manifest(manifest, mapping_path=args.mapping, composition_path=args.composition,
                primary_files=primary, supplement_files=supplements, provenance_files=provenance,
                contributing_mappings=contributing, expected_format=args.source_format)
        manifest_digest = digest_file(manifest)
        if args.source_sha256s_sha256 is not None and args.source_sha256s_sha256 != manifest_digest:
            raise ValueError("physical source input manifest differs from the verified acquisition handoff")
        # A second installation may provide identical packaged provenance at
        # another path. Retain the manifest's saved authority after proving
        # both copies equal its declared content. Data/member paths are never
        # rebound by this definition-only relocation.
        manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
        for role, requested in tuple(provenance.items()):
            recorded = manifest_document.get("provenance", {}).get(role, {})
            saved = (manifest.parent / recorded.get("path", "")).resolve()
            if saved != Path(requested).resolve():
                expected = recorded.get("sha256")
                if (not saved.is_file() or digest_file(saved) != expected
                        or digest_file(requested) != expected):
                    raise ValueError("saved native provenance differs from the installed source definition")
                provenance[role] = saved
        bundle = decode_composed_source(
            args.composition, args.mapping, primary, supplements, provenance,
            input_manifest=manifest, input_manifest_sha256=manifest_digest,
            contributing_mappings=contributing, scratch_destination=output_root,
            atmospheric_grids=(self.grid,), workers=self.workers)
        try:
            snapshots = _forcing_series(bundle)
            for_grids = getattr(snapshots, "for_grids", None)
            if for_grids is not None:
                snapshots = for_grids((self.grid,))
            snapshots = overlay_snapshot_sequence(snapshots, self.overlay,
                            binding=self.overlay_binding, workers=self.workers)
            times = _forcing_valid_times(snapshots)
            if (not times or times[0] != self.experiment.start_time
                    or (times[-1]-times[0]).total_seconds() < self.experiment.run_seconds):
                raise ValueError("physical source does not cover the complete model initial/boundary window")
            source_identity = {
                "adapter":"rw-wps-mapped-composition-v2",
                "mapping_sha256":bundle.mapping_sha256,
                "composition_sha256":bundle.composition_sha256,
                "input_manifest_sha256":manifest_digest,
                "composition_receipt_sha256":composition_receipt_identity_sha256(mapped_composition_receipt(bundle)),
                "preparation_case_policy":self.case_policy,
                "water_temperature_overlay":self.overlay_binding,
                "static_identity":self.static_identity,
                **({"static_highres":self.highres_identity} if self.highres_identity is not None else {})}
            from gpuwm.ensemble.mapped_physical_contract import (
                mapped_physical_field_contract, mapped_physical_evidence_files)
            field_evidence_files = {**mapped_physical_evidence_files(args.mapping, args.composition),
                                    "raw_source_manifest": manifest,
                                    **{"native_decoder_"+key: path for key, path in bundle.decoder_paths.items()}}
            field_contract = mapped_physical_field_contract(
                self.geometry, mapping_path=args.mapping, composition_path=args.composition,
                source_identity=source_identity,
                extra_evidence={"raw_source_manifest": manifest_digest,
                                **{"native_decoder_"+key: value for key, value in bundle.decoder_sha256.items()}})
            store = NativePhysicalStore(output_root, grid_identity=self.geometry, source_identity=source_identity,
                                        field_contract=field_contract)
            for snapshot in snapshots:
                mapped = interpolate_era5_to_lambert(snapshot, self.grid, backend=self.backend,
                    target_landmask=np.asarray(static["LANDMASK"]) >= 0.5,
                    water_temperature_statics=self.water_statics)
                terrain_blend.before_initialize(mapped.fields.get("SOURCE_OROGRAPHY"))
                store.write(mapped)
            verify_overlay_sequence(snapshots)
            if any(digest_file(path) != expected for path,expected in self.controls.items()):
                raise ValueError("shared native physical preparation controls changed during mapping")
            preprocessing = self.backend.receipt()
            store.document["source"]["preprocessing"] = preprocessing
            store.document["source"]["static_identity"] = physical_static_identity(
                static, self.static_identity["attributes"])
            result = store.seal()
            result["seconds"] = time.perf_counter()-started
            result["preprocessing"] = preprocessing
            result["field_contract_evidence_files"] = {key: str(path) for key, path in field_evidence_files.items()}
            return result
        finally:
            bundle.close()
