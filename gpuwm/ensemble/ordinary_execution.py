"""Unchanged recipe trajectories through their original prepared tree runners."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from types import MappingProxyType

from gpuwm.ensemble.recipes import SourceRecipe, SourceTrajectory
from gpuwm.ensemble.source_preparation import PostedSourcePreparation
from gpuwm.ensemble.posted_execution import PostedRuntimeMemberBinding


@dataclass(frozen=True)
class OrdinaryMemberSource:
    """One member's original prepared configuration on its source trajectory."""
    specification: PostedSourcePreparation
    inputs: object


def _digest(path):
    from gpuwm.ensemble.physical_store import digest_file
    return digest_file(path)


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00").replace("_", "T"))
    if not isinstance(value, datetime):
        raise ValueError("ordinary source trajectory has no valid UTC clock")
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reader_signature(inputs):
    """Original loader digests, rather than caller-supplied source assertions."""
    from gpuwm.prepared_domain_tree_forecast import PreparedTreeInputs
    from gpuwm.prepared_single_domain_forecast import PreparedForecastInputs
    if isinstance(inputs, PreparedTreeInputs):
        return {"kind": "tree", "authority": dict(inputs.authority_sha256),
            "source_identity": dict(inputs.source_identity),
            "physics_profile_assertion": (None if inputs.physics_profile_assertion is None else
                                          dict(inputs.physics_profile_assertion)),
            "domains": [{"grid_id": row.grid_id, "parent_id": row.parent_id,
                "identity": dict(row.cache_identity), "authority": dict(row.authority_sha256)}
                for row in inputs.domains]}
    if isinstance(inputs, PreparedForecastInputs):
        return {"kind": "scalar", "authority": dict(inputs.file_sha256),
            "identity": dict(inputs.cache_identity), "source_member": inputs.source_member}
    raise TypeError("ordinary recipes need original preflighted scalar or tree inputs")


def _ordinary_preflight(inputs, *, head_sha256):
    """Reapply the original head reader without source preparation or future waits."""
    from gpuwm.prepared_domain_tree_forecast import PreparedTreeInputs, preflight_prepared_tree
    from gpuwm.ensemble.posted_execution import preflight_member_inputs
    if isinstance(inputs, PreparedTreeInputs):
        return preflight_prepared_tree(prepared_root=inputs.prepared_root,
            prepared_head_sha256=head_sha256, experiment_config=inputs.experiment_config,
            experiment_config_sha256=_digest(inputs.experiment_config),
            physics_profile=(inputs.physics_profile_assertion or {}).get("profile"),
            devices_options=inputs.experiment.devices)
    return preflight_member_inputs(inputs, prepared_root=inputs.prepared_root,
                                  head_sha256=head_sha256)


def _planned_trajectory(manifest):
    """Normalize only source fields the original manifest actually declares."""
    record = manifest.get("request", manifest)
    if not isinstance(record, dict):
        raise ValueError("ordinary input plan has no original source request")
    if "source" not in record:
        return None
    source = record["source"]
    if isinstance(source, dict):
        record = source
        source = record.get("source", record.get("source_id", record.get("model")))
    cycle = record.get("cycle", record.get("source_cycle"))
    if not source or cycle is None:
        raise ValueError("ordinary input plan has no bound source and cycle")
    return SourceTrajectory(source, _utc(cycle), record.get("member", manifest.get("member")))


def _planned_source_binding(specification, inputs, head):
    """Bind declarative raw inventories without opening future payloads."""
    from gpuwm.prep_handoff import _posted_handoff
    from gpuwm.mapped_authoring import INPUT_MANIFEST_SCHEMA
    from gpuwm.source_adapters import get_source_adapter
    manifest = head["basis"]["as_posted"]["input_plan"]["manifest"]
    _, arguments, schedule, selected, _ = _posted_handoff(
        specification.acquisition_root, trajectory=specification.trajectory)
    if get_source_adapter(inputs.source).source_id != selected.source:
        raise ValueError("ordinary head reader belongs to another native source selector")
    planned = _planned_trajectory(manifest)
    if planned is not None:
        if planned != selected:
            raise ValueError("ordinary prepared input plan belongs to another source, cycle or member")
        return {"basis": "original manifest source request and verified acquisition", "trajectory_sha256": planned.identity}
    # The original mapped author binds roles and paths, rather than duplicating
    # source/cycle labels. Its native source reader checks mapping/decoder/member
    # authority; the acquisition helper above checks the actual route and cycle.
    if manifest.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise ValueError("ordinary input plan has no original source request or recognized declarative inventory")
    if arguments.count("--input-list") != 1 or arguments.count("--author-input-manifest") != 1:
        raise ValueError("ordinary declarative source needs its original input-list and manifest authority")
    input_list = Path(arguments[arguments.index("--input-list") + 1]).resolve()
    manifest_path = Path(arguments[arguments.index("--author-input-manifest") + 1]).resolve()
    actual = tuple(Path(line).resolve() for line in input_list.read_text(encoding="utf-8").splitlines() if line.strip())
    rows = manifest.get("primary_files")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or not row.get("path") for row in rows):
        raise ValueError("ordinary declarative input plan has no original primary-file inventory")
    expected = tuple((manifest_path.parent / row["path"]).resolve() for row in rows)
    if actual != expected:
        raise ValueError("ordinary declarative primary-file inventory differs from the verified acquisition input list")
    plan_table = head["basis"]["as_posted"]["input_plan"].get("route_table_sha256")
    if schedule.get("table_sha256") is not None and schedule["table_sha256"] != plan_table:
        raise ValueError("ordinary input plan and acquisition use different source route tables")
    return {"basis": "original declarative inventory, head reader and verified acquisition",
        "trajectory_sha256": selected.identity, "input_list_sha256": _digest(input_list),
        "original_primary_paths": [str(path) for path in expected]}


_ROLE_WORDS = {
    "--experiment-config": ("experiment_config", "namelist"),
    "--namelist-input": ("namelist",), "--stock-wrf-namelist-input": ("namelist", "stock"),
    "--wps-namelist": ("wps",), "--domain-spec": ("domain_spec", "domain"),
    "--static-input": ("static", "geog"), "--static-cache": ("static", "geog"),
    "--static-receipt": ("static", "geometry"), "--cpu-preprocess-bridge": ("bridge", "decoder", "preprocessing"),
    "--bridge": ("bridge", "decoder"), "--grib2-inventory": ("inventory", "decoder"),
    "--grib2-dump": ("dump", "decoder"),
}


def _configured_file_bindings(specification, inputs, head, reader):
    """Bind authored file controls to actual ordinary named role digests."""
    roles = {}
    def remember(role, digest):
        if isinstance(digest, str) and len(digest) == 64:
            roles[str(role)] = digest
    def visit(value, path=()):
        if isinstance(value, dict):
            if "sha256" in value:
                remember("/".join(path), value["sha256"])
            for key, item in value.items():
                if key == "source_sha256" and isinstance(item, dict):
                    for role, digest in item.items():
                        remember("/".join((*path, key, role)), digest)
                elif key.endswith("_sha256") and isinstance(item, str):
                    remember("/".join((*path, key)), item)
                elif isinstance(item, (dict, list)):
                    visit(item, (*path, str(key)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))
    for role, digest in reader["authority"].items():
        remember("loader/" + role, digest)
    for domain in reader.get("domains", ()):
        for role, digest in domain["authority"].items():
            remember(f"loader/d{domain['grid_id']:02d}/" + role, digest)
    remember("loader/experiment_config", _digest(inputs.experiment_config))
    wps = getattr(inputs, "wps_namelist", None)
    if wps is not None:
        remember("loader/wps_namelist", _digest(wps))
    remember("cache/identity/namelist", head["basis"]["cache"]["identity"].get("namelist_sha256"))
    visit(head["basis"], ("head", "basis"))
    visit(reader, ("reader",))
    bindings = {}
    for flag, record in specification.configuration_files.items():
        tokens = _ROLE_WORDS.get(flag, ())
        matched = sorted(role for role, digest in roles.items() if digest == record["sha256"] and
            any(token in role.lower().replace("-", "_") for token in tokens))
        if not matched:
            raise ValueError(f"ordinary recipe {flag}={record['path']} has no captured ordinary role digest; "
                             "this override cannot be proved to have prepared the bound source head")
        bindings[flag] = {"sha256": record["sha256"], "ordinary_roles": matched}
    return bindings


def _native_control_bindings(specification, inputs, head):
    """Classify each remaining authored option; no initialization knob drops."""
    file_flags = set(specification.configuration_files)
    execution_values = {"--preprocess-workers", "--pipeline-workers", "--prepare-workers"}
    execution_switches = {"--no-stock-wrf-export", "--skip-stock-wrf-export"}
    records = {}
    arguments = specification.native_arguments
    index = 0
    exp = inputs.experiment
    def backend_values():
        from gpuwm.prepared_domain_tree_forecast import PreparedTreeInputs
        readers = (tuple(domain.cache_reader for domain in inputs.domains)
                   if isinstance(inputs, PreparedTreeInputs) else (inputs.cache_reader,))
        values = []
        def visit(value):
            if isinstance(value, dict):
                if "preprocess_backend" in value:
                    values.append(str(value["preprocess_backend"]))
                preprocessing = value.get("preprocessing")
                if isinstance(preprocessing, dict) and preprocessing.get("backend") is not None:
                    values.append(str(preprocessing["backend"]))
                for item in value.values():
                    if isinstance(item, (dict, list)):
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        for reader in readers:
            visit(reader.header.get("metadata", {}))
            visit(reader.header.get("identity", {}))
        visit(head["basis"]["cache"]["identity"])
        return values
    while index < len(arguments):
        option = arguments[index]
        index += 1
        flag, equals, attached = option.partition("=")
        if flag in execution_switches:
            if equals:
                raise ValueError(f"ordinary recipe {flag} is an execution switch, not a valued initialization override")
            records[flag] = {"basis": "original export bookkeeping only", "value": True}
            continue
        if not flag.startswith("--"):
            raise ValueError("ordinary recipe native controls must use their original named options")
        if equals:
            value = attached
        elif index < len(arguments) and not arguments[index].startswith("--"):
            value = arguments[index]
            index += 1
        else:
            raise ValueError(f"ordinary recipe {flag} has no captured original value")
        if flag in file_flags:
            continue
        if flag in execution_values:
            if not value.isdecimal() or int(value) < 1:
                raise ValueError(f"ordinary recipe {flag} must retain a positive original worker count")
            records[flag] = {"basis": "original independent-column/source scheduling only", "value": int(value)}
            continue
        captured = None
        if flag == "--history-interval-seconds":
            captured = exp.root.history_interval_s
        elif flag == "--run-seconds":
            captured = exp.run_seconds
        elif flag == "--p-top-pa":
            captured = exp.vertical.p_top
        elif flag == "--preprocess-backend":
            choices = backend_values()
            if choices and all(item == value for item in choices):
                captured = value
        elif flag == "--physics-profile":
            preflight = getattr(inputs, "preflight_arguments", {}) or {}
            assertion = getattr(inputs, "physics_profile_assertion", {}) or {}
            captured = preflight.get("physics_profile", assertion.get("profile"))
        elif flag == "--geog-root" and {"--static-input", "--static-cache"} & file_flags:
            records[flag] = {"basis": "unchanged already-prepared statics are role-bound; no geography construction", "value": value}
            continue
        try:
            matches = captured is not None and (value == captured if isinstance(captured, str) else float(value) == captured)
        except ValueError:
            matches = False
        if not matches:
            raise ValueError(f"ordinary recipe {flag}={value} has no matching captured initialization value; "
                             "this override cannot be proved to have prepared the bound source head")
        records[flag] = {"basis": "original typed config/preflight/cache selection", "value": captured}
    return records


class OrdinaryRecipeExecution:
    """A concrete source owner with no physical capture or member preparation.

    Specifications are the original acquisition handoffs. Inputs must be the
    original headed scalar/tree preflight results. Every original source seal
    remains mandatory; this owner adds member attribution, never a new source
    waiter, decoder, geometry builder or numeric initialization path.
    """
    SCHEMA = "gpuwm-ensemble-ordinary-recipe-execution.v1"

    def __init__(self, recipe, specifications, source_inputs, *, root, member_indices=None,
                 member_sources=None):
        if not isinstance(recipe, SourceRecipe) or recipe.kind == "recentered":
            raise ValueError("changed recentered fields require their physical provider execution owner")
        if recipe.kind not in ("control", "input-ensemble", "time-lagged", "multi-model"):
            raise ValueError("ordinary recipes require unchanged native source trajectories")
        self.recipe, self.specifications = recipe, MappingProxyType(dict(specifications))
        self.sources = MappingProxyType(dict(source_inputs))
        ids = tuple(member.index for member in recipe.members)
        if (not ids or len(set(ids)) != len(ids) or any(type(index) is not int or index < 0 for index in ids)
                or any(type(member.seed) is not int or not 0 <= member.seed < (1 << 64) for member in recipe.members)):
            raise ValueError("ordinary recipe needs unique nonnegative global indices and exact unsigned64 seeds")
        members = {member.index: member for member in recipe.members}
        self.member_order = tuple(members) if member_indices is None else tuple(member_indices)
        if (not self.member_order or len(set(self.member_order)) != len(self.member_order)
                or any(type(index) is not int or index not in members for index in self.member_order)):
            raise ValueError("ordinary recipe selection needs unique original member indices")
        self._members = {index: members[index] for index in self.member_order}
        acquisitions = {member.trajectory.identity: member.trajectory for member in self._members.values()}
        self.member_sources = MappingProxyType(dict(member_sources or {}))
        if any(type(index) is not int or index not in members for index in self.member_sources):
            raise ValueError("ordinary member source selection must use original recipe indices")
        if any(not isinstance(value, OrdinaryMemberSource) for value in self.member_sources.values()):
            raise TypeError("ordinary member source selection needs typed specification/input records")
        defaults = {member.trajectory.identity for index, member in self._members.items()
                    if index not in self.member_sources}
        if (not defaults <= self.specifications.keys() or not defaults <= self.sources.keys()
                or set(self.specifications) != set(self.sources)):
            raise ValueError("ordinary recipe must retain every selected original acquisition")
        if any(not isinstance(specification, PostedSourcePreparation) for specification in self.specifications.values()):
            raise TypeError("ordinary recipe requires concrete verified acquisition specifications")
        if any(not isinstance(value.specification, PostedSourcePreparation) or
                value.specification.trajectory != members[index].trajectory
                for index, value in self.member_sources.items()):
            raise ValueError("ordinary member specification must keep its exact original source trajectory")
        if recipe.kind == "control" and any(item != recipe.base for item in acquisitions.values()):
            raise ValueError("ordinary control members must keep the unchanged base trajectory")
        if recipe.kind != "control" and len(acquisitions) != len(self._members):
            raise ValueError("unchanged noncontrol recipes cannot duplicate a source trajectory")
        if _utc(recipe.end) <= _utc(recipe.start):
            raise ValueError("ordinary recipe needs a positive original forecast window")
        self._lock, self._records = threading.RLock(), {}
        self.root = Path(root).resolve()
        all_specifications = (*self.specifications.values(),
                              *(value.specification for value in self.member_sources.values()))
        protected = [path for item in all_specifications
                     for path in (item.acquisition_root, item.prepared_root)]
        if any(self.root.is_relative_to(Path(path).resolve()) or Path(path).resolve().is_relative_to(self.root)
               for path in protected):
            raise ValueError("ordinary recipe descriptor cannot overwrite original acquisition or prepared inputs")
        self._member_source_keys, self._bound_specifications, self._bound_inputs = {}, {}, {}
        self._bound_trajectories = {}
        for index, member in self._members.items():
            variant = self.member_sources.get(index)
            key = member.trajectory.identity if variant is None else f"member:{index}"
            self._member_source_keys[index] = key
            self._bound_specifications[key] = (self.specifications[key] if variant is None
                                                else variant.specification)
            self._bound_inputs[key] = self.sources[key] if variant is None else variant.inputs
            self._bound_trajectories[key] = member.trajectory
        self._source_heads, self._source_records = {}, {}
        for identity, specification in self._bound_specifications.items():
            trajectory = self._bound_trajectories[identity]
            if not isinstance(specification, PostedSourcePreparation) or specification.trajectory != trajectory:
                raise TypeError("ordinary recipe requires concrete verified acquisition specifications")
            specification.verify()
            inputs = self._bound_inputs[identity]
            if Path(inputs.prepared_root).resolve() != specification.prepared_root:
                raise ValueError("ordinary recipe inputs differ from the original prepared source root")
            from gpuwm.ingest.boundary_stream import read_head
            head = read_head(inputs.prepared_root)
            if inputs.stream_head is None or inputs.stream_head["head_sha256"] != head["head_sha256"]:
                raise ValueError("ordinary recipe requires its original immutable head-preflighted inputs")
            self._source_heads[identity] = head["head_sha256"]
            self._source_records[identity] = self._verify_source(identity)
        document = {"schema": self.SCHEMA, "recipe": recipe.describe(), "recipe_sha256": recipe.sha256,
            "member_order": list(self.member_order), "sources": self._source_records,
            **({"member_source_keys": self._member_source_keys} if self.member_sources else {}),
            "policy": "unchanged original scalar/tree inputs; no physical capture or repeated source preparation"}
        self._descriptor = self.root / "ordinary-roster.json"
        from gpuwm.ensemble.posted_physical import _publish_json
        _publish_json(self._descriptor, document)
        self._descriptor_sha256 = _digest(self._descriptor)
        self._plan_sha256 = hashlib.sha256(_canonical(document).encode()).hexdigest()

    def _verify_source(self, identity):
        from gpuwm.ingest.boundary_stream import bind_head, input_plan_sha256
        from gpuwm.ensemble.stochastic_authority import _same_configuration
        specification, inputs = self._bound_specifications[identity], self._bound_inputs[identity]
        specification.verify()
        head = bind_head(inputs.prepared_root, self._source_heads[identity])
        posted = head["basis"].get("as_posted")
        if not posted or input_plan_sha256(posted["input_plan"]) != posted["input_plan_sha256"]:
            raise ValueError("ordinary recipe needs its original immutable as-posted input-plan authority")
        control_sha = head["basis"]["cache"]["identity"].get("namelist_sha256")
        control_files = specification.configuration_files
        if control_sha not in {row["sha256"] for flag, row in control_files.items()
                               if flag in {"--experiment-config", "--namelist-input"}}:
            raise ValueError("ordinary recipe configuration differs from the captured native initialization")
        expected = _reader_signature(inputs)
        checked = _ordinary_preflight(inputs, head_sha256=head["head_sha256"])
        if _reader_signature(checked) != expected or not _same_configuration(inputs.experiment, checked.experiment):
            raise ValueError("ordinary recipe preflight changed original source, static, configuration or domain authority")
        if (_utc(inputs.experiment.start_time) != _utc(self.recipe.start)
                or float(inputs.experiment.run_seconds) != (self.recipe.end-self.recipe.start).total_seconds()):
            raise ValueError("ordinary member configuration does not cover the exact common recipe forecast window")
        file_bindings = _configured_file_bindings(specification, checked, head, expected)
        control_bindings = _native_control_bindings(specification, checked, head)
        source_binding = _planned_source_binding(specification, checked, head)
        return {"trajectory_sha256": specification.trajectory.identity, "source_head_sha256": head["head_sha256"],
            "input_plan_sha256": posted["input_plan_sha256"], "reader": expected,
            "configuration_files": control_files, "configured_file_bindings": file_bindings,
            "native_control_bindings": control_bindings, "source_binding": source_binding}

    def _selected(self, member_id):
        if type(member_id) is not int or member_id not in self._members:
            raise ValueError("ordinary recipe needs an original selected member index")
        member = self._members[member_id]
        return member, self._member_source_keys[member_id]

    @property
    def member_metadata(self):
        return {member.index: {"member_id": member.index, "seed": member.seed,
            "recipe_sha256": self.recipe.sha256, "trajectory_sha256": member.trajectory.identity,
            "source": member.trajectory.source, "cycle": member.trajectory.cycle.isoformat(),
            "source_member": member.trajectory.member} for member in self._members.values()}

    def planning_inputs(self, member_id):
        _, identity = self._selected(member_id)
        if _digest(self._descriptor) != self._descriptor_sha256:
            raise ValueError("ordinary recipe descriptor changed after member admission")
        if self._verify_source(identity) != self._source_records[identity]:
            raise ValueError("ordinary recipe source authority changed after member admission")
        return self._bound_inputs[identity]

    def stochastic_member_binding(self, member_id):
        member, identity = self._selected(member_id)
        self.planning_inputs(member_id)
        return PostedRuntimeMemberBinding(member_id, member.seed, self.recipe.sha256, member.trajectory,
            self._descriptor_sha256, self._plan_sha256, self._source_heads[identity], self._source_heads[identity])

    def run_member(self, member_id, *, forecast, observer=None):
        member, identity = self._selected(member_id)
        inputs = self.planning_inputs(member_id)
        with self._lock:
            if member_id in self._records:
                raise ValueError("an ordinary recipe member cannot run twice in one owner")
            record = {**self.member_metadata[member_id], "status": "forecasting",
                "source_head_sha256": self._source_heads[identity],
                "native_head_sha256": self._source_heads[identity],
                "native_preparation": "unchanged original prepared scalar/tree source"}
            self._records[member_id] = record
        try:
            result = forecast(inputs)
            status = result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
            if status not in (None, "PASS"):
                raise RuntimeError("original ordinary member returned a failing forecast receipt")
            self.planning_inputs(member_id)
            from gpuwm.ingest.boundary_stream import bind_head, verify_seal
            head = bind_head(inputs.prepared_root, self._source_heads[identity])
            seal = verify_seal(inputs.prepared_root, head=head)
            with self._lock:
                record.update(status="complete", ordinary_source_seal=seal)
            return result
        except BaseException as error:
            with self._lock:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise

    def receipt(self):
        with self._lock:
            records = {index: dict(record) for index, record in self._records.items()}
        pending = [index for index in self.member_order if records.get(index, {}).get("status") != "complete"]
        return {"schema": self.SCHEMA, "recipe_sha256": self.recipe.sha256,
            "provider_plan_sha256": self._plan_sha256, "descriptor_sha256": self._descriptor_sha256,
            "member_order": list(self.member_order), "members": records,
            **({"member_source_keys": dict(self._member_source_keys)} if self.member_sources else {}),
            "pending_members": pending, "status": "incomplete" if pending else "complete",
            "source_heads": dict(self._source_heads)}

    def require_complete(self):
        receipt = self.receipt()
        if receipt["pending_members"]:
            raise ValueError("ordinary recipe still has unfinished or failed member forecasts")
        return receipt
