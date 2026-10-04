"""Run a pinned native source roster through the ordinary ensemble session.

The request binds the recipe, native preparation arguments and acquisition
verification records. Preparation and GPU ownership remain with the caller.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def plain(value):
    return json.loads(json.dumps(value, default=str))


def sha(path):
    from gpuwm.ensemble.physical_store import digest_file
    return digest_file(path)


def moment(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def trajectory(row):
    from gpuwm.ensemble.recipes import SourceTrajectory
    return SourceTrajectory(row["source"], moment(row["cycle"]), row.get("member"))


def recipe_from_record(row):
    from gpuwm.ensemble.recipes import SourceRecipe, RecipeMember
    recipe = SourceRecipe(row["kind"], trajectory(row["base"]), moment(row["start"]), moment(row["end"]),
        tuple(RecipeMember(item["index"], item["seed"], trajectory(item["trajectory"])) for item in row["members"]),
        tuple(trajectory(item) for item in row.get("donor_population", ())),
        row.get("calibration", "not calibrated"))
    if recipe.sha256 != row["sha256"]:
        raise ValueError("campaign recipe changed from its frozen scientific roster")
    return recipe


def execution_window(request, recipe):
    full = (recipe.end - recipe.start).total_seconds()
    purpose = request.get("execution_purpose", "calibration")
    if purpose not in {"calibration", "identity"}:
        raise ValueError("campaign execution purpose must identify calibration or identity qualification")
    duration = request.get("identity_run_seconds", full)
    if (type(duration) not in (int, float) or not math.isfinite(duration)
            or duration <= 0 or duration > full
            or (purpose != "identity" and duration != full)):
        raise ValueError("a shorter native run must be explicitly scoped as identity qualification")
    return {"purpose": purpose, "source_window_seconds": full,
            "execution_run_seconds": duration,
            "calibration_complete": purpose == "calibration" and "member_indices" not in request}


def bounded_identity_inputs(inputs, execution):
    """Change only the stop time after the complete preparation is admitted."""
    original = inputs.experiment
    full, duration = execution["source_window_seconds"], execution["execution_run_seconds"]
    if float(original.run_seconds) != full:
        raise ValueError("identity input was not admitted for its complete source window")
    if duration == full:
        return inputs, None
    if execution["purpose"] != "identity":
        raise ValueError("only an explicit identity qualification may shorten runtime duration")
    if len(original.domains) != 1:
        raise ValueError("campaign identity qualification requires its original single-domain experiment")
    steps = Fraction(str(duration)) / original.dt_exact(original.root.grid_id)
    if steps.denominator != 1:
        raise ValueError("identity stop must lie on the original model step lattice")
    domains = tuple(replace(domain, run=replace(domain.run, run_seconds=duration))
                    for domain in original.domains)
    bounded = replace(original, run_seconds=duration, domains=domains)
    before, after = asdict(original), asdict(bounded)
    expected = dict(before, run_seconds=duration,
                    domains=tuple(dict(domain, run=dict(domain["run"], run_seconds=duration))
                                  for domain in before["domains"]))
    if after != expected:
        raise ValueError("identity runtime changed configuration beyond the declared stop time")
    def digest(value):
        return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    receipt = {"source_experiment_sha256": digest(before), "execution_experiment_sha256": digest(after),
               "source_window_seconds": full, "execution_run_seconds": duration,
               "changed_fields": ["run_seconds", "domains[0].run.run_seconds"],
               "execution_model_steps": int(steps), "purpose": "identity"}
    return replace(inputs, experiment=bounded), receipt


def verify_runtime(request):
    """Pin the qualified installed implementation, beyond version labels."""
    from gpuwm.ensemble.acquisition_binding import verified_artifact
    reference = request.get("required_runtime")
    if not reference:
        raise ValueError("campaign request lacks its qualified joint runtime artifact receipt")
    document = json.loads(verified_artifact(reference))
    binding = document.get("campaign_binding", {})
    if binding.get("status") != "PASS":
        raise ValueError("joint runtime artifact has not passed its qualification gates")
    if Path(sys.executable).resolve() != Path(binding["python_executable"]).resolve():
        raise ValueError("campaign interpreter differs from the qualified joint runtime")
    modules = binding.get("module_files", {})
    if not {"gpuwm", "gpuwm_data"} <= set(modules):
        raise ValueError("joint runtime must pin both loaded engine and data packages")
    for name, artifact in modules.items():
        module = sys.modules.get(name)
        specification = importlib.util.find_spec(name) if module is None else None
        location = (getattr(module, "__file__", None) if module is not None else
                    None if specification is None else specification.origin)
        if location is None or Path(location).resolve() != Path(artifact["path"]).resolve():
            raise ValueError("campaign loaded another engine or data package location")
        verified_artifact(artifact)
    artifacts = binding.get("artifacts", ())
    if not artifacts:
        raise ValueError("joint runtime lacks its installed source and native artifact hashes")
    for artifact in artifacts:
        if sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("qualified joint runtime source or native artifact changed")
    for name, value in binding.get("environment", {}).items():
        if name == "TMPDIR":
            continue
        if os.environ.get(name) != value:
            raise ValueError("campaign native bridge environment differs from its qualified artifact")
    return {"receipt": dict(reference), "python_executable": sys.executable,
            "module_files": modules, "verified_artifacts": len(artifacts),
            "environment": dict(binding.get("environment", {})),
            "execution_temporary_directory": os.environ.get("TMPDIR"), "status": "PASS"}


def source_binding(row):
    from gpuwm.ensemble.acquisition_binding import AcquiredSourceBinding
    identity = row.get("trajectory", row.get("identity"))
    result = AcquiredSourceBinding(trajectory(identity), Path(row["path"]), row["sha256"], row["verification"])
    result.verify()
    return result


def indexed_input(row, recipe, member, indexes):
    """Keep source assignment attached to the exact admitted native row."""
    from gpuwm.ensemble.acquisition_binding import verified_artifact
    reference = row["input_index"]
    key = (reference["path"], reference["sha256"])
    if key not in indexes:
        indexes[key] = json.loads(verified_artifact(reference))
    ordinal = reference["row"]
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("native input index needs its exact nonnegative row number")
    admitted = indexes[key]["rows"][ordinal]
    assignments = admitted.get("frozen_members", [admitted])
    if not any(value.get("recipe_sha256") == recipe.sha256
               and value.get("original_member_index") == member.index
               and value.get("original_member_seed") == member.seed for value in assignments):
        raise ValueError("native input index assigns another frozen member or seed")
    if admitted["preflight_arguments"] != row["preflight_arguments"]:
        raise ValueError("campaign native preparation differs from its pinned input index row")
    expected_trajectory = recipe.base if recipe.kind == "recentered" else member.trajectory
    acquisition = admitted.get("base_acquisition_manifest") if recipe.kind == "recentered" else admitted.get("acquisition_manifest")
    if acquisition is None:
        raise ValueError("indexed native input has no original source acquisition")
    identity = acquisition.get("trajectory", acquisition.get("identity"))
    if trajectory(identity) != expected_trajectory:
        raise ValueError("indexed native input names another source trajectory")
    if len(row["source_bindings"]) != 1 or row["source_bindings"][0]["sha256"] != acquisition["sha256"]:
        raise ValueError("native input source binding differs from its indexed acquisition")
    metadata = admitted["current_metadata"]
    if "cache_header" in metadata:
        references = metadata.values()
    else:
        prepared_root = Path(admitted["prepared_root"])
        references = [
            {"path": str(prepared_root / "native/prepared-cache/header.json"), "sha256": metadata["cache_header_sha256"]},
            {"path": str(prepared_root / "proof.json"), "sha256": metadata["proof_sha256"]},
            {"path": admitted["experiment_config"], "sha256": metadata["experiment_config_sha256"]},
            {"path": admitted["wps_namelist"], "sha256": metadata["wps_namelist_sha256"]}]
    for value in references:
        verified_artifact(value)
    return admitted


def physical_assignment(inputs, admitted, recipe, member, donors, request):
    """Close recentering identity against all frozen physical donor manifests."""
    from gpuwm.ensemble.acquisition_binding import verified_artifact
    if recipe.kind != "recentered":
        return None
    reference = admitted["physical_manifest"]
    document = json.loads(verified_artifact(reference))
    native = inputs.cache_identity["source_identity"].get("ensemble_physical_input")
    if (not native or native["manifest_sha256"] != reference["sha256"]
            or native["manifest"] != document):
        raise ValueError("prepared member consumed another physical input manifest")
    source = document["source"]
    if (source.get("schema") != "gpuwm-ensemble-recentered-preparation.v1"
            or source.get("selected_member") != member.trajectory.member):
        raise ValueError("prepared recentered input selected another original donor member")
    amplitude = admitted["amplitude"]
    if (amplitude not in (0.5, 1.0, 1.5) or request.get("icbc_amplitude", amplitude) != amplitude
            or any(value.get("amplitude") != amplitude for value in source["bounds"].values())):
        raise ValueError("prepared recentering differs from its frozen amplitude candidate")
    bound = {binding.trajectory.member: binding.verification["artifacts"]["physical_manifest"]["sha256"]
             for binding in donors}
    if bound != {name: value["sha256"] for name, value in source["donors"].items()}:
        raise ValueError("prepared recentering used another physical donor population")
    return {"manifest": reference, "selected_member": source["selected_member"],
            "amplitude": amplitude, "donor_manifests": bound}


def prepare_roster(request):
    """Validate the whole scientific roster before any GPU allocation."""
    from gpuwm.ensemble.member_preparation import PreparedMemberInput, PreparedMemberRoster
    from gpuwm.native_wrf_contract import native_geometry_contract
    from gpuwm.prepared_single_domain_forecast import preflight_prepared_forecast
    frozen_recipe = recipe_from_record(request["recipe"])
    recipe = (frozen_recipe.select_members(request["member_indices"])
              if "member_indices" in request else frozen_recipe)
    execution = execution_window(request, frozen_recipe)
    expected = tuple(item.index for item in recipe.members)
    if tuple(row["member_index"] for row in request["prepared_members"]) != expected:
        raise ValueError("campaign native inputs must retain the exact selected original member order")
    prepared, records, indexes = [], [], {}
    geometry_sha = None
    for member, row in zip(recipe.members, request["prepared_members"]):
        admitted = indexed_input(row, frozen_recipe, member, indexes)
        arguments = dict(row["preflight_arguments"])
        for name in ("prepared_root", "experiment_config", "wps_namelist", "domain_bundle"):
            if arguments.get(name) is not None:
                arguments[name] = Path(arguments[name])
        inputs = preflight_prepared_forecast(**arguments)
        if (moment(inputs.experiment.start_time) != recipe.start
                or float(inputs.experiment.run_seconds) != execution["source_window_seconds"]):
            raise ValueError("native campaign preparation differs from its complete source window")
        geometry = native_geometry_contract(inputs.grid, inputs.experiment.root.run)
        digest = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if geometry_sha is None:
            geometry_sha = digest
        if digest != geometry_sha:
            raise ValueError("campaign members have different native target geometries")
        lbc = inputs.cache_reader.header["metadata"]["lbc"]
        intervals = lbc["intervals"]
        times = tuple(recipe.start + timedelta(seconds=float(item["start_seconds"])) for item in intervals)
        times += (recipe.start + timedelta(seconds=float(intervals[-1]["end_seconds"])),)
        sources = tuple(source_binding(value) for value in row["source_bindings"])
        donors = tuple(source_binding(value) for value in row.get("donor_bindings", ()))
        physical = physical_assignment(inputs, admitted, recipe, member, donors, request)
        inputs, identity_override = bounded_identity_inputs(inputs, execution)
        receipt = {"preflight_arguments": plain(arguments), "authorities": dict(inputs.file_sha256),
            "source": inputs.source, "member": inputs.source_member,
            "frozen_recipe_sha256": frozen_recipe.sha256, "execution_recipe_sha256": recipe.sha256,
            "execution_window": execution,
            "identity_duration_override": identity_override,
            "native_input_index": row["input_index"], "physical_input": physical,
            "status": "PASS"}
        prepared.append(PreparedMemberInput(member.index, member.seed, inputs,
            {"cache_identity": dict(inputs.cache_identity), "stage": "initial native state"},
            {"cache_identity": dict(inputs.cache_identity), "boundary_valid_times": [value.isoformat() for value in times]},
            member.trajectory, sources, times, digest, recipe.sha256, donors, receipt))
        records.append(dict(receipt, member_index=member.index, member_seed=member.seed))
    return PreparedMemberRoster(recipe, prepared, shared_geometry_sha256=geometry_sha), records


def stochastic_controls(amplitude):
    from gpuwm.ensemble.stochastic import StochasticConfig
    if type(amplitude) not in (int, float):
        raise ValueError("campaign stochastic amplitude must be a frozen numerical candidate")
    if amplitude == 0:
        return None
    if type(amplitude) not in (int, float) or amplitude not in (0.5, 1.0):
        raise ValueError("campaign stochastic amplitude must be one of its frozen candidates")
    def standard(kind):
        return StochasticConfig.wrf_reference(kind)
    return {"sppt": {"stddev": standard("sppt").stddev * amplitude},
        "skebs": {name: {"backscatter": standard("skebs_" + name).backscatter * amplitude * amplitude}
                  for name in ("psi", "theta")},
        "spp": {"conv": 0, "pbl": 1, "lsm": 1},
        "spp_configs": {name: {"stddev": standard("spp_" + name).stddev * amplitude}
                        for name in ("pbl", "lsm")}}


def campaign_session(request, roster, output_directory):
    """The ensemble session one campaign arm runs in.

    The frozen stochastic candidates are this campaign's own arms: it is
    the measurement that calibrates them.  A public request may not carry
    stochastic controls (``EnsembleRequest`` refuses them while they are
    uncalibrated), so the campaign binds its provider to the session
    itself, the way the calibrated policy does.  Passing them in the
    request refused every non-zero arm, and a refusal no measurement can
    reach could never be retired.  The amplitude stays in
    ``campaign-run-request.json`` and the bound process configurations in
    the manifest's ``runtime_stochastic_authorities``.
    """
    from gpuwm.ensemble.production import PreparedEnsembleSession
    options = {"members": len(roster.members), "keep_member_files": request.get("keep_member_files", False),
        "retain_member_diagnostics": True, "thresholds": request["thresholds"],
        "member_device_ids": request.get("member_device_ids", [0]),
        "base_seed": request["base_seed"]}
    arguments = {"member_roster": roster}
    controls = stochastic_controls(request["stochastic_amplitude"])
    if controls is not None:
        from gpuwm.ensemble.stochastic_model import StochasticModelProvider
        arguments["stochastic_provider"] = StochasticModelProvider.from_mapping(controls)
    return PreparedEnsembleSession(options, output_directory=output_directory, **arguments)


def run(request, out, *, preflight_only=False, bare_identity=False,
        ready_file=None, start_file=None):
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError("campaign output already exists; each qualified run needs a new directory")
    out.mkdir(parents=True)
    def save(name, record):
        (out / name).write_text(json.dumps(record, indent=2, default=str, allow_nan=False) + "\n")
    save("campaign-run-request.json", request)
    record = {"schema": "gpuwm-ensemble-campaign-run.v1", "status": "failed",
        "route": "bare ordinary identity" if bare_identity else "ensemble campaign",
        "request_sha256": sha(out / "campaign-run-request.json"), "runner_sha256": sha(Path(__file__)),
        "versions": {name: importlib.metadata.version(name) for name in ("gpuwm", "gpuwm-data")}}
    try:
        if record["versions"] != request["required_versions"]:
            raise ValueError("campaign runtime distribution versions differ from its qualified joint artifact")
        record["runtime_authority"] = verify_runtime(request)
        roster, admissions = prepare_roster(request)
        record["execution_window"] = execution_window(request, roster.recipe)
        record["native_admissions"] = admissions
        record["roster"] = roster.receipt()
        if bare_identity and (record["execution_window"]["purpose"] != "identity"
                              or len(roster.members) != 1 or request["stochastic_amplitude"] != 0
                              or roster.recipe.kind != "control"):
            raise ValueError("bare identity comparison needs the unperturbed original singleton control")
        if preflight_only:
            if "cupy" in sys.modules:
                raise ValueError("campaign preflight unexpectedly imported the GPU runtime")
            record["status"] = "PREFLIGHT_PASS"
            return record
        if (ready_file is None) != (start_file is None):
            raise ValueError("prepared worker needs both its ready receipt and owned start marker")
        if ready_file is not None:
            if "cupy" in sys.modules:
                raise ValueError("prepared worker imported the GPU runtime before its lease")
            ready_file, start_file = Path(ready_file), Path(start_file)
            if ready_file.exists() or start_file.exists():
                raise FileExistsError("prepared worker markers already exist; use a new identity attempt")
            record["status"] = "PREFLIGHT_READY"
            save("campaign-run-receipt.json", record)
            ready = {"status": "PREFLIGHT_READY", "pid": os.getpid(),
                     "request_sha256": record["request_sha256"], "start_file": str(start_file)}
            temporary = ready_file.with_name(ready_file.name + ".tmp")
            temporary.write_text(json.dumps(ready) + "\n")
            temporary.replace(ready_file)
            deadline = time.monotonic() + 1800
            while not start_file.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("prepared worker did not receive its owned GPU start marker")
                time.sleep(0.25)
            start = json.loads(start_file.read_bytes())
            if start.get("pid") != os.getpid() or start.get("request_sha256") != record["request_sha256"]:
                raise ValueError("GPU start marker belongs to another prepared worker")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ.pop("GPUWM_NO_LOCAL_GPU", None)
            record["gpu_start"] = {"marker": str(start_file), "pid": os.getpid()}
        from gpuwm.prepared_single_domain_forecast import run_prepared_forecast
        if bare_identity:
            result = run_prepared_forecast(roster.members[0].inputs, output_directory=out / "forecast")
            if (result["status"] != "PASS" or result["run_seconds"]
                    != record["execution_window"]["execution_run_seconds"]):
                raise ValueError("ordinary identity forecast did not complete its explicit short window")
            record.update(status="BARE_IDENTITY_PASS", ordinary_report={
                "path": str(out / "forecast/report.json"), "sha256": sha(out / "forecast/report.json")})
            return record
        session = campaign_session(request, roster, out / "forecast")
        result = session.run_prepared(run_prepared_forecast, roster.members[0].inputs)
        if result["status"] != "PASS" or result["members_completed"] != list(session.member_order):
            raise ValueError("ensemble session did not complete the full original campaign roster")
        status = ("IDENTITY_PASS" if record["execution_window"]["purpose"] == "identity" else
                  "PARTITION_PASS" if "member_indices" in request else "PASS")
        record.update(status=status, ensemble_manifest={"path": str(out / "forecast/ensemble-run.json"),
            "sha256": sha(out / "forecast/ensemble-run.json")}, products=session.completed_products())
        return record
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["seconds"] = time.perf_counter() - started
        save("campaign-run-receipt.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--bare-identity", action="store_true",
                        help="Run the declared short singleton control through the bare ordinary forecast.")
    parser.add_argument("--ready-file", type=Path, help="Publish CPU admission before waiting for the owned GPU lease.")
    parser.add_argument("--start-file", type=Path, help="Owned start marker for an already-admitted worker.")
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    run(request, args.out, preflight_only=args.preflight_only, bare_identity=args.bare_identity,
        ready_file=args.ready_file, start_file=args.start_file)


if __name__ == "__main__":
    main()
