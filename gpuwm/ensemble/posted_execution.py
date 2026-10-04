"""Execute posted recipes through the original native start-first lifecycle.

Planning uses already verified ordinary source inputs. Changed physical members
enter their original native initializer at their own head; future boundaries
remain streamed. Complete-window member bindings retain their separate API.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import threading

from gpuwm.ensemble.physical_store import digest_file
from gpuwm.ensemble.source_preparation import PostedPreparationFactory


@dataclass(frozen=True)
class PostedRuntimeMemberBinding:
    """Semantic restart authority for a member whose source is still posting."""
    member_id: int
    seed: int
    recipe_sha256: str
    trajectory: object
    provider_descriptor_sha256: str
    provider_plan_sha256: str
    source_head_sha256: str
    native_head_sha256: str | None


def preflight_member_inputs(source_inputs, *, prepared_root, head_sha256):
    """Apply the ordinary preflight to the new member's published authorities."""
    from gpuwm import stage_cli
    from gpuwm.prepared_single_domain_forecast import preflight_prepared_forecast

    root = Path(prepared_root).resolve()
    bundle = (stage_cli.resolve_bundle(root) if head_sha256 is None else
              stage_cli.resolve_head_bundle(root, head_sha256))
    arguments = dict(source_inputs.preflight_arguments)
    old_root = Path(source_inputs.prepared_root).resolve()
    for key in ("experiment_config", "wps_namelist", "domain_bundle"):
        value = arguments.get(key)
        if value is not None:
            path = Path(value).resolve()
            if path.is_relative_to(old_root):
                arguments[key] = root / path.relative_to(old_root)
    arguments.update(source=bundle["source"], prepared_root=root)
    if head_sha256 is None:
        digests = stage_cli.single_domain_digests(bundle)
        arguments.update(proof_sha256=digests["proof"],
                         source_manifest_sha256=digests["source_manifest"],
                         prepared_content_sha256=digests["prepared_content"])
    else:
        arguments.update(prepared_head_sha256=head_sha256,
                         source_manifest_sha256=bundle["source_manifest_sha256"])
    return preflight_prepared_forecast(**arguments)


class PostedRecipeExecution:
    """One source-neutral posted owner, selected before ensemble admission.

    ``source_inputs`` contains the ordinary, head-preflighted source trees keyed
    by canonical trajectory identity. Admission sees those exact configurations
    without preparing every member or waiting for the future source seal.
    ``run_member`` calls the original forecast callback with the actual member
    inputs. The scheduler owns card scopes, mutable restoration and progress.
    """

    SCHEMA = "gpuwm-ensemble-posted-execution.v1"

    def __init__(self, factory, source_inputs, *, member_root, member_indices=None):
        if not isinstance(factory, PostedPreparationFactory):
            raise TypeError("posted execution requires the concrete native source factory")
        self.factory = factory
        self.recipe = factory.provider.recipe
        self.root = Path(member_root).resolve()
        self.sources = dict(source_inputs)
        if set(self.sources) != set(factory.sources):
            raise ValueError("posted execution must retain the complete native source population")
        members = {member.index: member for member in self.recipe.members}
        self.member_order = (tuple(members) if member_indices is None else tuple(member_indices))
        if (not self.member_order or len(set(self.member_order)) != len(self.member_order) or
                any(type(index) is not int or index not in members for index in self.member_order)):
            raise ValueError("posted execution selection must retain unique original recipe indices")
        self._members = {index: members[index] for index in self.member_order}
        self._lock = threading.RLock()
        self._records = {}
        self._source_heads = {}
        self._descriptor_path = factory.provider.root / "provider-head.json"
        self._descriptor_sha256 = digest_file(self._descriptor_path)
        descriptor = json.loads(self._descriptor_path.read_bytes())
        if descriptor["plan"] != factory.provider.plan:
            raise ValueError("posted execution descriptor differs from its native provider plan")
        self._provider_plan_sha256 = factory.plan_sha256
        from gpuwm.ingest.boundary_stream import read_head
        for identity, inputs in self.sources.items():
            specification = factory.sources[identity].verify()
            if Path(inputs.prepared_root).resolve() != specification.prepared_root:
                raise ValueError("posted execution inputs differ from the bound ordinary source root")
            if inputs.source != specification.trajectory.source:
                raise ValueError("posted execution inputs identify another ordinary source")
            ordinary = descriptor["sources"][identity]
            head = read_head(inputs.prepared_root, expected_sha256=ordinary["prepared_head_sha256"])
            if (inputs.stream_head is not None and
                    inputs.stream_head["head_sha256"] != head["head_sha256"]):
                raise ValueError("posted execution head differs from the provider's ordinary source")
            self._source_heads[identity] = head["head_sha256"]

    @property
    def member_metadata(self):
        return {member.index: {"member_id": member.index, "seed": member.seed,
            "recipe_sha256": self.recipe.sha256,
            "trajectory_sha256": member.trajectory.identity,
            "source": member.trajectory.source, "cycle": member.trajectory.cycle.isoformat(),
            "source_member": member.trajectory.member}
            for member in self._members.values()}

    def _selected(self, member_id):
        if type(member_id) is not int or member_id not in self._members:
            raise ValueError("posted execution requires an original recipe member index")
        member = self._members[member_id]
        trajectory = self.recipe.base if self.recipe.kind == "recentered" else member.trajectory
        return member, trajectory

    def planning_inputs(self, member_id):
        """No forecast, member initialization or wait for future raw payloads."""
        _, trajectory = self._selected(member_id)
        if digest_file(self._descriptor_path) != self._descriptor_sha256:
            raise ValueError("posted execution provider descriptor changed after admission")
        if self.factory.plan_sha256 != self._provider_plan_sha256:
            raise ValueError("posted execution native provider plan changed after admission")
        self.factory.sources[trajectory.identity].verify()
        from gpuwm.ingest.boundary_stream import bind_head
        inputs = self.sources[trajectory.identity]
        bind_head(inputs.prepared_root, self._source_heads[trajectory.identity])
        return inputs

    def stochastic_member_binding(self, member_id):
        member, trajectory = self._selected(member_id)
        self.planning_inputs(member_id)
        with self._lock:
            native_head = self._records.get(member_id, {}).get("native_head_sha256")
        return PostedRuntimeMemberBinding(member_id, member.seed, self.recipe.sha256,
            member.trajectory, self._descriptor_sha256, self.factory.plan_sha256,
            self._source_heads[trajectory.identity], native_head)

    def run_member(self, member_id, *, forecast, observer=None):
        """Start the ordinary forecast at its actual immutable native head."""
        member, trajectory = self._selected(member_id)
        source_inputs = self.planning_inputs(member_id)
        with self._lock:
            if member_id in self._records:
                raise ValueError("a posted member forecast cannot run twice in one execution owner")
            record = {**self.member_metadata[member_id], "status": "preparing",
                "source_head_sha256": self._source_heads[trajectory.identity],
                "provider_plan_sha256": self.factory.plan_sha256,
                "native_preparation": "independent changed-field initialization" if
                    self.recipe.kind == "recentered" else "shared unchanged ordinary source"}
            self._records[member_id] = record

        def advance(inputs):
            from gpuwm.ingest.boundary_stream import read_head
            head = read_head(inputs.prepared_root)
            with self._lock:
                record["status"] = "forecasting"
                record["native_head_sha256"] = head["head_sha256"]
            result = forecast(inputs)
            with self._lock:
                record["status"] = "verifying_sources"
            return result

        try:
            if self.recipe.kind == "recentered":
                output = self.root / f"member-{member_id:04d}"
                def at_head(head_sha256):
                    inputs = preflight_member_inputs(source_inputs,
                        prepared_root=output, head_sha256=head_sha256)
                    from gpuwm.ingest.boundary_stream import read_head
                    physical = read_head(output)["basis"].get("ensemble_physical")
                    if (physical is None or physical["member_index"] != member_id or
                            physical["provider_plan"] != self.factory.provider.plan):
                        raise ValueError("posted member forecast lost its original provider head authority")
                    return advance(inputs)
                prepared, result = self.factory.prepare_member(member_id,
                    output_root=output, forecast=at_head, observer=observer)
                with self._lock:
                    record["native_preparation_receipt"] = prepared
            else:
                # The original source reader owns both every consumed boundary
                # segment and the final seal. No second physical preparation.
                result = advance(source_inputs)
            from gpuwm.ingest.boundary_stream import bind_head, verify_seal
            source_head = bind_head(source_inputs.prepared_root,
                                   self._source_heads[trajectory.identity])
            source_seal = verify_seal(source_inputs.prepared_root, head=source_head)
            with self._lock:
                record.update(status="complete", ordinary_source_seal=source_seal)
            return result
        except BaseException as error:
            with self._lock:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise

    def receipt(self):
        with self._lock:
            records = {member: dict(record) for member, record in self._records.items()}
        pending = [member for member in self.member_order
                   if records.get(member, {}).get("status") != "complete"]
        return {"schema": self.SCHEMA, "recipe_sha256": self.recipe.sha256,
            "provider_plan_sha256": self.factory.plan_sha256,
            "member_order": list(self.member_order), "members": records,
            "pending_members": pending, "status": "incomplete" if pending else "complete",
            "source_heads": dict(self._source_heads)}

    def require_complete(self):
        receipt = self.receipt()
        if receipt["pending_members"]:
            raise ValueError("posted execution has unfinished or failed member forecasts")
        return receipt
