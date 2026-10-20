"""Compose a packaged source and export its canonical state to WPS in Rust.

The output is consumed by the existing hex init door. No weather arrays enter
Python and no model-specific hex adapter is selected here.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess


def register_cli(sub):
    parser = sub.add_parser("source-intermediate", help="compose a global packaged source into WPS intermediate for init")
    parser.add_argument("--source", help="source registry name, including earth2-<table key>")
    parser.add_argument("--profile", help="packaged composition profile (overrides the source default)")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--supplement", action="append", default=[], metavar="ROLE=FILE")
    parser.add_argument("--valid-time", required=True, help="frame time YYYY-MM-DDTHH:MM:SS")
    parser.add_argument("--member")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.set_defaults(func=main)


def main(args):
    from gpuwm.mapped_authoring import author_input_manifest
    from gpuwm.mapped_engine_bridge import require_engine, run_engine
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.source_authorities import (packaged_authorities,
        packaged_contributing_mappings, packaged_provenance_files)

    profile = args.profile
    if not profile and args.source:
        adapter = get_source_adapter(args.source)
        profile = adapter.packaged_profile
        # The family's registry aliases share an adapter, while the model table
        # owns each model's profile. Resolve the alias through that same table.
        if adapter.source_id == "earth2":
            from gpuwm.earth2_run import resolve_model
            _, row = resolve_model(args.source.removeprefix("earth2-"))
            profile = row["profile"]
    if not profile:
        raise ValueError("source-intermediate requires --profile or a source with a packaged profile")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    authorities = packaged_authorities(profile)
    contributors = packaged_contributing_mappings(profile)
    provenance = packaged_provenance_files(profile)
    supplements = {}
    for binding in args.supplement:
        role, sep, path = binding.partition("=")
        if not sep or not role or not path:
            raise ValueError("--supplement takes ROLE=FILE")
        supplements.setdefault(role, []).append(Path(path).resolve())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = author_input_manifest(
        args.work_dir / "inputs.json", mapping_path=authorities["mapping"],
        composition_path=authorities["composition"], primary_files=args.input,
        supplement_files=supplements, provenance_files=provenance,
        contributing_mappings=contributors, member=args.member)
    engine = require_engine()
    capabilities = subprocess.run([str(engine), "capabilities"], check=True,
                                  capture_output=True, text=True)
    if not json.loads(capabilities.stdout.splitlines()[-1]).get("wps_intermediate_v5"):
        raise ValueError("source-intermediate needs this release's mapped engine with WPS intermediate v5 export")
    composed = run_engine(
        "compose", mapping=authorities["mapping"], files=args.input,
        composition=authorities["composition"], supplements=supplements,
        provenance=provenance, contributing_mappings=contributors,
        input_manifest=manifest["manifest"]["path"],
        input_manifest_sha256=manifest["manifest"]["sha256"],
        output=args.work_dir / "composed", engine=engine, threads=args.workers,
        lead_batch=True, on_progress=lambda event: print(json.dumps(event), flush=True))
    contract = Path(__file__).with_name("authorities") / "wps-intermediate-fields.v1.json"
    command = [str(engine), "wps", "--frameset", str(args.work_dir / "composed"),
               "--composition", str(authorities["composition"]), "--contract", str(contract),
               "--valid-time", args.valid_time, "--out", str(args.out)]
    if args.member:
        command += ["--member", args.member]
    from gpuwm.ingest.preparation_workers import worker_environment
    result = subprocess.run(command, check=False, text=True, capture_output=True,
                            env=worker_environment(args.workers))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Rust WPS export failed without a diagnostic")
    receipt = json.loads(result.stdout.splitlines()[-1])
    receipt.update(source=args.source, profile=profile, compose=composed["receipt"],
                   input_manifest=manifest["manifest"], command=command)
    receipt_path = args.out.with_name(args.out.name + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "output": receipt["output"],
                      "nfglevels": receipt["nfglevels"], "nfgsoillevels": receipt["nfgsoillevels"]}))
    return 0
