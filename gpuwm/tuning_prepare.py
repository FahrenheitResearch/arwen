"""Bounded CPU preparation through the ordinary source acquisition handoff.

The acquisition's table owns primary files and supplemental roles. The
native preparer owns decode, interpolation, soil initialization and the
complete domain tree. This helper adds explicit CPU scheduling and receipts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


SCHEMA = "gpuwm-tuning-preparation.v1"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "RAYON_NUM_THREADS", "GPUWM_PREPROCESS_THREADS", "GPUWM_MAPPED_ENGINE_THREADS")


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def disk_inventory(paths):
    """Record real capacity of every declared large-data destination."""
    result = []
    for name, path in paths.items():
        target = Path(path).resolve()
        measured = target
        while not measured.exists() and measured != measured.parent:
            measured = measured.parent
        usage = shutil.disk_usage(measured)
        result.append({"role": name, "path": str(target), "measured_path": str(measured),
                       "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free})
    return result


def cpu_limit(requested=None):
    """Cap native threads at the actual cgroup CPU quota, never host nproc."""
    from gpuwm.ingest.preparation_workers import cpu_budget
    limits = [4, cpu_budget()["available_cpus"]]
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
            raise ValueError("preparation workers must be a positive integer")
        limits.append(requested)
    return min(limits)


@contextmanager
def cpu_environment(temporary, workers):
    temporary = Path(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    values = {key: str(workers) for key in THREAD_VARIABLES}
    values.update(TMPDIR=str(temporary.resolve()), CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
    keys = {*values, "GPUWM_PHYSICS_PARAMS"}
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(values)
    os.environ.pop("GPUWM_PHYSICS_PARAMS", None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def materialize_config(source_config, output, *, cache_root):
    """Write the sole allowed location edit before the native input hashes."""
    from gpuwm.toml_document import emit_experiment_toml
    source_config, output = Path(source_config).resolve(), Path(output).resolve()
    raw = tomllib.loads(source_config.read_text(encoding="utf-8-sig"))
    if "ensemble" in raw or "physics_params" in raw:
        raise ValueError("prepare a selector group without a forecast-time ensemble or parameter set")
    if not isinstance(raw.get("fetch"), dict) or not raw["fetch"].get("source"):
        raise ValueError("selector group has no explicit ordinary source fetch recipe")
    if len(raw.get("domain", [])) < 2:
        raise ValueError("tuning preparation requires the complete nested product configuration")
    static = raw.get("static")
    if not isinstance(static, dict) or not isinstance(static.get("highres"), dict):
        raise ValueError("selector group must explicitly declare product high-resolution statics before materialization")
    original = json.loads(json.dumps(raw, default=str))
    old_cache = static["highres"].get("cache_root")
    static["highres"]["cache_root"] = str(Path(cache_root).resolve())
    output.mkdir(parents=True, exist_ok=True)
    target = output / "experiment.toml"
    text = emit_experiment_toml(raw)
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise ValueError("materialized selector group already contains different configuration bytes")
    if not target.exists():
        target.write_text(text, encoding="utf-8")
    after = json.loads(json.dumps(tomllib.loads(text), default=str))
    after["static"]["highres"].pop("cache_root", None)
    original["static"]["highres"].pop("cache_root", None)
    if after != original:
        raise ValueError("TOML materialization changed a field beyond the shared cache path")
    return target, {"source_config_sha256": _sha(source_config),
                    "experiment_config_sha256": _sha(target),
                    "only_change": "static.highres.cache_root absolute shared cache path",
                    "source_cache_root": old_cache, "runtime_cache_root": str(Path(cache_root).resolve())}


def preparation_arguments(handoff, *, wps, config, geog_root, bundle_root, manifest_path, workers):
    """Append only caller controls to the acquisition's actual argument door."""
    from gpuwm.prep_handoff import preparation_arguments as acquisition_arguments
    arguments = acquisition_arguments(handoff)
    caller_flags = {"--wps-namelist", "--experiment-config", "--geog-root", "--output-root",
                    "--preprocess-backend", "--preprocess-workers", "--hierarchy-workers"}
    if any(flag in arguments for flag in caller_flags):
        raise ValueError("fetch handoff already supplies a caller preparation control; two authorities cannot select one input")
    if arguments.count("--author-input-manifest") == 1:
        arguments[arguments.index("--author-input-manifest") + 1] = str(manifest_path)
    arguments += ["--wps-namelist", str(wps), "--experiment-config", str(config),
                  "--geog-root", str(Path(geog_root).resolve()), "--output-root", str(bundle_root),
                  "--preprocess-backend", "cpu", "--preprocess-workers", str(workers),
                  "--hierarchy-workers", "1"]
    return arguments


def prepare_group(config, *, shared_raw, geog_root, output, cache_root,
                  workers=None, timeout_seconds=14400):
    """Acquire one shared source window, prepare all domains, then preflight."""
    if os.name == "nt":
        raise ValueError("native preparation runs on the assigned remote box; the desktop is not a preparation host")
    if timeout_seconds <= 0:
        raise ValueError("preparation must have a positive hard time bound")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "tuning-preparation.json"
    if receipt_path.exists():
        raise ValueError("preparation receipt already exists; verify its actual bundle instead of overwriting a prior attempt")
    workers = cpu_limit(workers)
    from gpuwm.ingest.preparation_workers import cpu_budget
    report = {"schema": SCHEMA, "status": "RUNNING", "workers": workers,
              "hierarchy_workers": 1, "preprocess_backend": "cpu", "cpu_budget": cpu_budget(),
              "disk_before": disk_inventory({"shared_source": shared_raw, "shared_statics": cache_root,
                                               "prepared_output": output})}
    _write(receipt_path, report)
    try:
        temporary = Path.home() / ".gpuwm" / "tuning-harness" / "tmp" / ("prepare-" + output.name)
        with cpu_environment(temporary, workers):
            from gpuwm import fetch_routes, runplan, stage_cli
            from gpuwm.experiment import load_experiment
            from gpuwm.go_cli import config_fetch_request, managed_download_dir, pin_request, pinned_transport
            from gpuwm.hrrr_route_inputs import run_route_inputs
            from gpuwm.prepared_domain_tree_forecast import preflight_prepared_tree

            config, materialization = materialize_config(config, output / "authority", cache_root=cache_root)
            raw = tomllib.loads(config.read_text(encoding="utf-8"))
            experiment = load_experiment(config)
            hints = config_fetch_request(raw)
            pinned, _ = pinned_transport(hints)
            hints = pin_request(hints, pinned)
            hints["as_posted"] = False
            source = str(hints["source"])
            if runplan.prepared_chain_for_source(source, source_root=None) != "prepared:staged":
                raise ValueError("this source has no table acquisition handoff for native nested preparation")
            shared = managed_download_dir(Path(shared_raw).resolve(), hints)
            fetch_arguments = runplan._fetch_arguments_from_hints(hints, out=shared)
            fetch_command = [sys.executable, "-B", "-m", "gpuwm.cli", "fetch", *fetch_arguments]
            report.update(materialization=materialization, source=source, shared_source_root=str(shared),
                          fetch_command=fetch_command)
            _write(receipt_path, report)
            with (output / "fetch.log").open("w", encoding="utf-8") as log:
                acquired = subprocess.run(fetch_command, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=timeout_seconds, check=False)
            if acquired.returncode:
                raise ValueError(f"ordinary source fetch exited {acquired.returncode}; no tuning inputs were prepared")
            handoff_path = shared / fetch_routes.PREP_ARGUMENTS_NAME
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            if handoff.get("schema") != fetch_routes.PREP_ARGUMENTS_SCHEMA or handoff.get("source") != source:
                raise ValueError("ordinary fetch did not publish the requested source's preparation handoff")
            if handoff.get("cycle") != str(hints["cycle"]):
                raise ValueError("ordinary fetch handoff cycle differs from the requested selector group's cycle")
            if handoff.get("unbound_supplement_roles"):
                raise ValueError(f"ordinary fetch left unresolved supplemental roles {handoff['unbound_supplement_roles']}; supply the required source data before preparation")
            route_inputs = run_route_inputs(config, experiment, raw=raw, into=output / "authority" / "route-inputs")
            wps = route_inputs["wps_namelist"]
            bundle_root = output / "prepared"
            arguments = preparation_arguments(handoff, wps=wps, config=config, geog_root=geog_root,
                bundle_root=bundle_root, manifest_path=output / "authority" / "inputs.json", workers=workers)
            command = ["nice", "-n", "10", sys.executable, "-B", "-m", "gpuwm.cli", "prep", *arguments]
            report.update(preparation_command=command, fetch_handoff_sha256=_sha(handoff_path),
                          wps_namelist_sha256=_sha(wps), configured_domains=len(experiment.domains))
            _write(receipt_path, report)
            with (output / "prepare.log").open("w", encoding="utf-8") as log:
                prepared = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=timeout_seconds, check=False)
            if prepared.returncode:
                raise ValueError(f"ordinary native preparation exited {prepared.returncode}; no runnable tuning bundle is claimed")
            bundle = stage_cli.resolve_bundle(bundle_root)
            if bundle["layout"] != "tree":
                raise ValueError("the ordinary preparation returned only a root domain; the tuning recipe requires every nest")
            hashes = stage_cli.tree_digests(bundle, config)
            prepared_inputs = {"prepared_root": str(bundle_root), "experiment_config": str(config),
                               "preparation_receipt_sha256": hashes["preparation_receipt"],
                               "experiment_config_sha256": hashes["experiment_config"]}
            verified = preflight_prepared_tree(**{**prepared_inputs, "prepared_root": bundle_root,
                                                  "experiment_config": config})
            if len(verified.domains) != len(experiment.domains):
                raise ValueError("native preparation domain count differs from the complete selector group")
            report.update(status="PASS", **prepared_inputs, packet_config_sha256=materialization["source_config_sha256"],
                          domain_count=len(verified.domains),
                          forcing_hours=list(verified.forcing_hours),
                          preparation_document=str(bundle["document"]), preparation_document_sha256=_sha(bundle["document"]),
                          disk_after=disk_inventory({"shared_source": shared_raw, "shared_statics": cache_root,
                                                      "prepared_output": output}))
            _write(receipt_path, report)
            (output / "DONE").write_text("0\n", encoding="ascii")
            return report
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        _write(receipt_path, report)
        (output / "DONE").write_text("1\n", encoding="ascii")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shared-raw", type=Path, required=True)
    parser.add_argument("--geog-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args(argv)
    try:
        result = prepare_group(args.config, shared_raw=args.shared_raw, geog_root=args.geog_root,
                               output=args.out, cache_root=args.cache_root, workers=args.workers,
                               timeout_seconds=args.timeout_seconds)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except Exception as error:
        print(f"tuning preparation: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
