"""Compare complete identity-run histories and every checkpoint field word."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(path):
    from gpuwm.ensemble.physical_store import digest_file
    return digest_file(path)


def moment(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("_", "T", 1))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def selected_recipe_authority(request, roster):
    """Prove a session is an exact original-member selection of its recipe."""
    from gpuwm.ensemble.recipes import RecipeMember, SourceRecipe, SourceTrajectory
    def trajectory(value):
        return SourceTrajectory(value["source"], moment(value["cycle"]), value.get("member"))
    row = request["recipe"]
    frozen = SourceRecipe(row["kind"], trajectory(row["base"]), moment(row["start"]), moment(row["end"]),
        tuple(RecipeMember(value["index"], value["seed"], trajectory(value["trajectory"])) for value in row["members"]),
        tuple(trajectory(value) for value in row.get("donor_population", ())), row["calibration"])
    if (frozen.sha256 != row["sha256"]
            or _canonical(frozen.describe()) != _canonical({key: value for key, value in row.items() if key != "sha256"})):
        raise ValueError("identity session changed its canonical frozen source recipe")
    indices = request.get("member_indices", [member.index for member in frozen.members])
    selected = frozen.select_members(indices)
    if (roster.get("recipe_sha256") != selected.sha256
            or _canonical(roster.get("recipe")) != _canonical(selected.describe())
            or roster.get("member_order") != [member.index for member in selected.members]
            or [member.get("member_id") for member in roster["members"]] != list(indices)):
        raise ValueError("identity session roster is not the exact requested original-member selection")
    by_id = {member.index: member for member in selected.members}
    for member in roster["members"]:
        expected = by_id[member["member_id"]]
        preparation = member["preparation"]
        if (member["seed"] != expected.seed or member["trajectory_sha256"] != expected.trajectory.identity
                or member["recipe_sha256"] != selected.sha256
                or preparation["execution_recipe_sha256"] != selected.sha256
                or preparation["frozen_recipe_sha256"] != frozen.sha256):
            raise ValueError("identity session member lost its frozen recipe, seed or trajectory authority")
    return {"frozen_recipe_sha256": frozen.sha256, "execution_recipe_sha256": selected.sha256,
            "member_order": list(indices), "members": selected.describe()["members"],
            "donor_population": selected.describe()["donor_population"],
            "selection": "SourceRecipe.select_members with the original full donor population"}


def _header_differences(left, right, path=""):
    """Retain every exact scalar/type difference without numerical tolerance."""
    if type(left) is not type(right):
        return [{"path": path, "left": left, "right": right}]
    if isinstance(left, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                result.append({"path": path + "/" + key, "left": left.get(key), "right": right.get(key),
                               "left_present": key in left, "right_present": key in right})
            else:
                result.extend(_header_differences(left[key], right[key], path + "/" + key))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": path, "left": left, "right": right}]
        return [value for index, (a, b) in enumerate(zip(left, right))
                for value in _header_differences(a, b, path + "/" + str(index))]
    return [] if _canonical(left) == _canonical(right) else [{"path": path, "left": left, "right": right}]


def stochastic_header_comparison(left, right, relation):
    """Compare every RNG field; prove the two selected-roster descriptors."""
    group, independent = relation["group"], relation["independent"]
    if (group["frozen_recipe_sha256"] != independent["frozen_recipe_sha256"]
            or group["donor_population"] != independent["donor_population"]
            or len(independent["members"]) != 1
            or independent["members"][0] not in group["members"]):
        raise ValueError("stochastic identity has no exact grouped-to-singleton recipe relationship")
    member = independent["members"][0]
    if (member["index"] != relation["member_id"] or member["seed"] != relation["seed"]):
        raise ValueError("stochastic identity relationship names another original member or seed")
    if left is None and right is None:
        if relation["stochastic_enabled"]:
            raise ValueError("an enabled identity run omitted its stochastic checkpoint authority")
        return {"status": "PASS", "relationship": relation, "raw_headers": {"group": None, "independent": None},
                "raw_differences": [], "scientific_differences": []}
    if not relation["stochastic_enabled"] or not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("stochastic checkpoint enabling differs from the frozen identity request")
    from gpuwm.ensemble.stochastic_seeds import process_seed, seed_label_receipt
    for header, selection in ((left, group), (right, independent)):
        if (header.get("contract") != "gpuwm-ensemble-stochastic-binding.v1"
                or type(header.get("member_id")) is not int or header["member_id"] != member["index"]
                or header.get("recipe_sha256") != selection["execution_recipe_sha256"]):
            raise ValueError("stochastic checkpoint is not bound to its canonical selected recipe and original member")
        hook = header["hook"]
        receipt = hook.get("wrf_seed_labels")
        labels = None if receipt is None else receipt["labels"]
        if receipt != seed_label_receipt(labels):
            raise ValueError("stochastic checkpoint changed its WRF process seed derivation")
        processes = ([hook["sppt"]] if hook.get("sppt") is not None else [])
        if hook.get("skebs") is not None:
            processes.extend((hook["skebs"]["psi"], hook["skebs"]["theta"]))
        processes.extend(hook.get("spp", {}).values())
        if not processes or hook.get("enabled") is not True:
            raise ValueError("enabled stochastic checkpoint has no active process authority")
        for process in processes:
            metadata = process["metadata"]
            if metadata["member_seed"] != process_seed(member["seed"], metadata["config"]["kind"], labels):
                raise ValueError("stochastic process seed differs from its original recipe member")
    differences = _header_differences(left, right)
    scientific = [row for row in differences if row["path"] != "/recipe_sha256"]
    return {"status": "PASS" if not scientific else "FAIL", "relationship": relation,
            "raw_headers": {"group": deepcopy(left), "independent": deepcopy(right)},
            "raw_differences": differences, "scientific_differences": scientific,
            "comparison": "Every stochastic parameter, counter, seed and spectrum descriptor is exact; "
                          "recipe hashes are proved canonical selections of one frozen recipe."}


def inventory(root, pattern):
    result = {}
    for path in sorted(Path(root).rglob(pattern)):
        if not path.is_file():
            continue
        if pattern.startswith("wrfout_") and re.fullmatch(
                r"wrfout_d\d{2}_\d{4}-\d{2}-\d{2}[_T]\d{2}[_:]\d{2}[_:]\d{2}(?:\.nc)?", path.name) is None:
            continue
        key = path.name.split("__", 1)[0].removesuffix(".npz") if pattern.startswith("gpuwmrst_") else path.name
        if key in result:
            raise ValueError("identity comparison contains duplicate native output names")
        result[key] = path
    if not result:
        raise ValueError(f"identity comparison contains no {pattern} outputs")
    return result


def array_record(value):
    import numpy as np
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("identity arrays must contain native words, not Python objects")
    return {"dtype": str(array.dtype), "shape": list(array.shape), "bytes": array.nbytes,
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def word_comparison(left, right):
    import numpy as np
    a, b = array_record(left), array_record(right)
    same_layout = a["dtype"] == b["dtype"] and a["shape"] == b["shape"]
    changed = None
    if same_layout:
        left, right = np.ascontiguousarray(left), np.ascontiguousarray(right)
        size = left.dtype.itemsize
        first = left.view(np.uint8).reshape(-1, size)
        second = right.view(np.uint8).reshape(-1, size)
        changed = int(np.count_nonzero(np.any(first != second, axis=1)))
    return {"left": a, "right": b, "changed_words": changed,
            "equal": same_layout and changed == 0}


def native_history_words(variable):
    """Recover finite stored words only when the native reader is lossless."""
    import numpy as np
    values, stored = variable[:], variable.dtype
    if stored.kind == "S":
        return np.ascontiguousarray(values, dtype=stored)
    if stored.kind not in "fiu" or (stored.kind in "iu" and stored.itemsize > 4):
        raise ValueError("native history word comparison lacks an exact reader for this stored type")
    if not np.isfinite(values).all():
        raise ValueError("history NaN payload words require whole-container byte equality")
    restored = np.ascontiguousarray(values, dtype=stored)
    if restored.astype(values.dtype).tobytes() != values.tobytes():
        raise ValueError("native history reader did not preserve the declared stored words")
    return restored


def history_comparison(left, right):
    from gpuwm.netcdf_bridge import open_dataset
    a_sha, b_sha = digest(left), digest(right)
    fields, attributes = {}, {}
    with open_dataset(left) as a, open_dataset(right) as b:
        a.set_auto_maskandscale(False)
        b.set_auto_maskandscale(False)
        if set(a.variables) != set(b.variables):
            raise ValueError("identity history field inventories differ")
        if {key: len(value) for key, value in a.dimensions.items()} != {
                key: len(value) for key, value in b.dimensions.items()}:
            raise ValueError("identity history dimensions differ")
        for name in sorted(a.variables):
            left_var, right_var = a.variables[name], b.variables[name]
            if (left_var.dtype != right_var.dtype or left_var.shape != right_var.shape
                    or left_var.dimensions != right_var.dimensions):
                raise ValueError("identity history native field layouts differ")
            if a_sha == b_sha:
                fields[name] = {"equal": True, "changed_words": 0,
                                "evidence": "complete native container SHA256 equality",
                                "dtype": str(left_var.dtype), "shape": list(left_var.shape)}
            else:
                fields[name] = word_comparison(native_history_words(left_var), native_history_words(right_var))
                fields[name]["evidence"] = "native reader with exact stored-type roundtrip"
            names = set(left_var.ncattrs()) | set(right_var.ncattrs())
            changed = [key for key in names if left_var.attributes.get(key) != right_var.attributes.get(key)]
            if changed:
                attributes[name] = sorted(changed)
        valid = a.variables["Times"][:].tobytes().decode("ascii").strip("\x00 ")
        globals_changed = [key for key in set(a.ncattrs()) | set(b.ncattrs())
                           if a.global_attributes.get(key) != b.global_attributes.get(key)]
    if digest(left) != a_sha or digest(right) != b_sha:
        raise ValueError("identity history changed while its bytes were compared")
    return {"left": str(left), "right": str(right), "left_sha256": a_sha, "right_sha256": b_sha,
            "container_bytes_equal": a_sha == b_sha, "valid_time": moment(valid).isoformat(),
            "field_words_equal": all(row["equal"] for row in fields.values()), "fields": fields,
            "variable_attribute_differences": attributes, "global_attribute_differences": sorted(globals_changed)}


def checkpoint_comparison(left, right, seconds, *, stochastic_recipe_relation=None):
    from gpuwm.io.restart import _load_restart, require_readable_format_version
    from gpuwm.ensemble.state_sha import checkpoint_state_sha_receipt
    a_sha, b_sha = digest(left), digest(right)
    a, first = _load_restart(left, with_arrays=True)
    b, second = _load_restart(right, with_arrays=True)
    for path, header, arrays in ((left, a, first), (right, b, second)):
        require_readable_format_version(header["format_version"], path)
        if header["elapsed_seconds"] != seconds:
            raise ValueError("identity checkpoint belongs to another forecast duration")
        if set(arrays) != set(header["array_manifest"]):
            raise ValueError("identity checkpoint arrays differ from their own native manifest")
        for name, array in arrays.items():
            specification = header["array_manifest"][name]
            if list(array.shape) != specification["shape"] or str(array.dtype) != specification["dtype"]:
                raise ValueError("identity checkpoint field layout differs from its native manifest")
    if set(first) != set(second):
        raise ValueError("identity checkpoint field inventories differ")
    fields = {name: word_comparison(first[name], second[name]) for name in sorted(first)}
    required = ("format_version", "elapsed_seconds", "config", "setup_fingerprint",
                "physics_setup_fingerprint", "physics_setup", "driver", "ensemble_stochastic",
                "root_external_lbc_clock")
    header_differences = [key for key in required if a.get(key) != b.get(key)]
    stochastic = None
    if stochastic_recipe_relation is not None:
        stochastic = stochastic_header_comparison(a.get("ensemble_stochastic"), b.get("ensemble_stochastic"),
                                                  stochastic_recipe_relation)
        if stochastic["status"] == "PASS":
            header_differences = [key for key in header_differences if key != "ensemble_stochastic"]
        elif "ensemble_stochastic" not in header_differences:
            header_differences.append("ensemble_stochastic")
    state = {"left": checkpoint_state_sha_receipt(left), "right": checkpoint_state_sha_receipt(right)}
    if digest(left) != a_sha or digest(right) != b_sha:
        raise ValueError("identity checkpoint changed while its words were compared")
    return {"left": str(left), "right": str(right), "left_sha256": a_sha, "right_sha256": b_sha,
            "field_words_equal": all(row["equal"] for row in fields.values()), "fields": fields,
            "scientific_header_differences": header_differences, "state_inventory": state,
            **({"stochastic_header_relationship": stochastic} if stochastic is not None else {}),
            "other_header_differences": sorted(key for key in set(a) | set(b)
                                               if key not in required and a.get(key) != b.get(key))}


def compare(ensemble_root, ordinary_root):
    roots = (Path(ensemble_root), Path(ordinary_root))
    receipts = [json.loads((root / "campaign-run-receipt.json").read_bytes()) for root in roots]
    requests = [json.loads((root / "campaign-run-request.json").read_bytes()) for root in roots]
    if [row["status"] for row in receipts] != ["IDENTITY_PASS", "BARE_IDENTITY_PASS"]:
        raise ValueError("both complete short identity runs are required before comparison")
    for root, receipt in zip(roots, receipts):
        if digest(root / "campaign-run-request.json") != receipt["request_sha256"]:
            raise ValueError("identity request changed after its forecast")
    if receipts[0]["execution_window"] != receipts[1]["execution_window"]:
        raise ValueError("identity forecasts used different execution windows")
    if receipts[0]["runtime_authority"] != receipts[1]["runtime_authority"]:
        raise ValueError("identity forecasts used different qualified runtime artifacts")
    if receipts[0]["roster"] != receipts[1]["roster"]:
        raise ValueError("identity forecasts used different prepared inputs, identities or seeds")
    if requests[0] != requests[1] or len(receipts[0]["native_admissions"]) != 1:
        raise ValueError("bare and ensemble identity comparison must use the same singleton request")
    request = requests[0]
    duration = receipts[0]["execution_window"]["execution_run_seconds"]
    cadence = request["prepared_members"][0]["preflight_arguments"]["history_interval_seconds"]
    if duration % cadence:
        raise ValueError("identity comparison requires a complete final scheduled history frame")
    expected_times = [moment(request["recipe"]["start"]) + timedelta(seconds=index * cadence)
                      for index in range(int(duration / cadence) + 1)]
    history = [inventory(root, "wrfout_d*") for root in roots]
    if set(history[0]) != set(history[1]):
        raise ValueError("identity history frame inventories differ")
    frames = [history_comparison(history[0][name], history[1][name]) for name in sorted(history[0])]
    if sorted(moment(row["valid_time"]) for row in frames) != expected_times:
        raise ValueError("identity histories do not cover every declared valid time")
    checkpoints = [inventory(root, "gpuwmrst_d*.npz") for root in roots]
    if set(checkpoints[0]) != set(checkpoints[1]):
        raise ValueError("identity checkpoint output inventories differ")
    # The first qualification is one hour, equal to its original checkpoint cadence.
    if len(checkpoints[0]) != 1:
        raise ValueError("singleton identity comparison needs its one final native checkpoint")
    checkpoint = checkpoint_comparison(next(iter(checkpoints[0].values())),
                                       next(iter(checkpoints[1].values())), duration)
    words = all(row["field_words_equal"] for row in frames) and checkpoint["field_words_equal"]
    containers = all(row["container_bytes_equal"] for row in frames)
    scientific = not checkpoint["scientific_header_differences"]
    return {"schema": "ensemble-calibration.ordinary-singleton-identity.v1",
            "status": "PASS" if words and containers and scientific else "FAIL",
            "field_word_identity_status": "PASS" if words and scientific else "FAIL",
            "complete_history_container_identity_status": "PASS" if containers else "FAIL",
            "scope": "Every retained history variable word and native checkpoint array; whole history file bytes reported separately; no numerical tolerances.",
            "execution_window": receipts[0]["execution_window"], "histories": frames, "checkpoint": checkpoint,
            "receipts": [{"path": str(root / "campaign-run-receipt.json"),
                          "sha256": digest(root / "campaign-run-receipt.json")} for root in roots]}


def compare_independent_sessions(group_root, independent_roots):
    """Compare original members in ordinary waves with separate sessions."""
    def load(root):
        root = Path(root)
        receipt = json.loads((root / "campaign-run-receipt.json").read_bytes())
        request = json.loads((root / "campaign-run-request.json").read_bytes())
        if receipt["status"] != "IDENTITY_PASS" or digest(root / "campaign-run-request.json") != receipt["request_sha256"]:
            raise ValueError("grouped and independent forecasts need complete pinned identity receipts")
        reference = receipt["ensemble_manifest"]
        if digest(reference["path"]) != reference["sha256"]:
            raise ValueError("identity production session manifest changed")
        session = json.loads(Path(reference["path"]).read_bytes())
        if session["status"] != "PASS" or session["members_completed"] != receipt["roster"]["member_order"]:
            raise ValueError("identity session did not complete every original member")
        modes = {row["execution_mode"] for row in session["packing"]["batches"]}
        if not modes or any(not value.startswith("ordinary_") for value in modes):
            raise ValueError("this matrix qualifies ordinary member execution, not native packing")
        return root, receipt, request, session, sorted(modes)

    def member_authority(member):
        value = deepcopy(member)
        # The selected roster hash changes with grouping. The full frozen
        # recipe hash, original index/seed and every source artifact remain.
        value.pop("recipe_sha256")
        value["preparation"].pop("execution_recipe_sha256")
        return value

    def runtime_authority(receipt):
        value = dict(receipt["runtime_authority"])
        value.pop("execution_temporary_directory", None)
        return value

    grouped = load(group_root)
    group_selection = selected_recipe_authority(grouped[2], grouped[1]["roster"])
    members = {row["member_id"]: row for row in grouped[1]["roster"]["members"]}
    request = grouped[2]
    seconds = grouped[1]["execution_window"]["execution_run_seconds"]
    cadence = request["prepared_members"][0]["preflight_arguments"]["history_interval_seconds"]
    if seconds % cadence:
        raise ValueError("identity matrix needs its complete final scheduled history frame")
    expected_times = [moment(request["recipe"]["start"]) + timedelta(seconds=index * cadence)
                      for index in range(int(seconds / cadence) + 1)]
    results, selected = [], set()
    for independent_root in independent_roots:
        independent = load(independent_root)
        if len(independent[1]["roster"]["members"]) != 1:
            raise ValueError("an independent identity session must contain exactly one original member")
        member = independent[1]["roster"]["members"][0]
        independent_selection = selected_recipe_authority(independent[2], independent[1]["roster"])
        member_id = member["member_id"]
        if member_id not in members or member_id in selected:
            raise ValueError("independent identity repeats or changes the grouped original member roster")
        selected.add(member_id)
        for name in ("recipe", "base_seed", "stochastic_amplitude", "icbc_amplitude", "thresholds"):
            if request[name] != independent[2][name]:
                raise ValueError("independent identity changed its frozen source or stochastic setting")
        if (grouped[1]["execution_window"] != independent[1]["execution_window"]
                or runtime_authority(grouped[1]) != runtime_authority(independent[1])
                or member_authority(members[member_id]) != member_authority(member)):
            raise ValueError("independent identity changed the original member, seed or native authority")
        roots = [entry[0] / "forecast/members" / f"member-{member_id:04d}" for entry in (grouped, independent)]
        histories = [inventory(root, "wrfout_d*") for root in roots]
        if set(histories[0]) != set(histories[1]):
            raise ValueError("grouped and independent history inventories differ")
        frames = [history_comparison(histories[0][name], histories[1][name]) for name in sorted(histories[0])]
        if sorted(moment(row["valid_time"]) for row in frames) != expected_times:
            raise ValueError("grouped identity history omits a declared valid time")
        checkpoints = [inventory(root, "gpuwmrst_d*.npz") for root in roots]
        if set(checkpoints[0]) != set(checkpoints[1]) or len(checkpoints[0]) != 1:
            raise ValueError("grouped identity needs matching final native checkpoints")
        checkpoint = checkpoint_comparison(next(iter(checkpoints[0].values())),
            next(iter(checkpoints[1].values())), seconds,
            stochastic_recipe_relation={"group": group_selection, "independent": independent_selection,
                "member_id": member_id, "seed": member["seed"],
                "stochastic_enabled": request["stochastic_amplitude"] != 0})
        equal = (all(row["container_bytes_equal"] and row["field_words_equal"] for row in frames)
                 and checkpoint["field_words_equal"] and not checkpoint["scientific_header_differences"])
        results.append({"member_id": member_id, "seed": member["seed"], "status": "PASS" if equal else "FAIL",
                        "independent_root": str(independent[0]), "execution_modes": independent[4],
                        "histories": frames, "checkpoint": checkpoint})
    if selected != set(members):
        raise ValueError("identity matrix omits independent comparisons for grouped members")
    return {"schema": "ensemble-calibration.grouped-independent-identity.v1",
            "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
            "scope": "Ordinary member waves versus separate one-member sessions; no optimized native packing claim.",
            "execution_window": grouped[1]["execution_window"], "execution_modes": grouped[4],
            "frozen_recipe_sha256": request["recipe"]["sha256"], "original_member_ids": sorted(selected),
            "stochastic_amplitude": request["stochastic_amplitude"], "members": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, required=True)
    comparator = parser.add_mutually_exclusive_group(required=True)
    comparator.add_argument("--ordinary", type=Path)
    comparator.add_argument("--independent", type=Path, action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (compare(args.ensemble, args.ordinary) if args.ordinary is not None else
                  compare_independent_sessions(args.ensemble, args.independent))
    except BaseException as error:
        result = {"schema": "ensemble-calibration.ordinary-singleton-identity.v1",
                  "status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    result["comparator_sha256"] = digest(Path(__file__))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "field_word_identity_status",
                     "complete_history_container_identity_status", "error") if key in result}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
