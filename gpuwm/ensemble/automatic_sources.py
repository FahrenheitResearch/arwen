"""Automatic source selection with explicit provenance and frozen defaults.

This module resolves source work. It neither initializes a model nor runs a
forecast. The ordinary acquisition, preparation and ensemble execution owners
consume the selected recipe through :mod:`automatic_preparation`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

from gpuwm.ensemble.physical_store import digest_file
from gpuwm.ensemble.recipes import SourceRecipe, SourceTrajectory, build_recipe

POLICY_SCHEMA = "gpuwm-ensemble-source-policy.v1"
SELECTION_SCHEMA = "gpuwm-ensemble-source-selection.v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _plain(value):
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("automatic source context needs explicit UTC valid times")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class FileAuthority:
    role: str
    path: Path
    sha256: str

    @classmethod
    def capture(cls, role, path):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{role} authority is not a file: {path}")
        return cls(str(role), path, digest_file(path))

    def verify(self):
        if not self.path.is_file() or digest_file(self.path) != self.sha256:
            raise ValueError(f"automatic source {self.role} authority changed: {self.path}")
        return self

    def describe(self):
        return {"role": self.role, "path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class DomainSourceContext:
    """Actual hierarchy authorities retained independently for each domain."""
    grid_id: int
    parent_id: int | None
    activation_time: datetime
    authorities: tuple[FileAuthority, ...] = ()

    def __post_init__(self):
        if type(self.grid_id) is not int or self.grid_id < 1:
            raise ValueError("source domain needs a positive original grid_id")
        if self.parent_id is not None and (type(self.parent_id) is not int or
                                          self.parent_id < 1 or self.parent_id == self.grid_id):
            raise ValueError("source domain has an invalid parent_id")
        object.__setattr__(self, "activation_time", _utc(self.activation_time))
        object.__setattr__(self, "authorities", tuple(self.authorities))

    def describe(self):
        return {"grid_id": self.grid_id, "parent_id": self.parent_id,
                "activation_time": self.activation_time.isoformat(),
                "authorities": [item.describe() for item in self.authorities]}


@dataclass(frozen=True)
class NativeSourceTemplate:
    """The front door's concrete controls, keyed by registered native runner.

    Acquisition-owned options are obtained from the actual fetch handoff.
    Templates contain its existing config/static/geography/preprocessor flags.
    """
    runner: str
    arguments: tuple[str, ...]
    configuration_files: tuple[FileAuthority, ...]

    @classmethod
    def capture(cls, runner, arguments):
        from gpuwm.ensemble.source_preparation import _OWNED, _config_files, _flags
        from gpuwm.source_cli import preparation_runners
        if runner not in preparation_runners():
            raise ValueError(f"unknown native preparation runner {runner!r}")
        arguments = tuple(str(value) for value in arguments)
        if _flags(arguments) & _OWNED:
            raise ValueError("automatic native template cannot replace source or physical ownership")
        files = _config_files(arguments)
        return cls(runner, arguments, tuple(FileAuthority(flag, Path(item["path"]), item["sha256"])
                                           for flag, item in sorted(files.items())))

    def verify(self):
        from gpuwm.ensemble.source_preparation import _config_files, _resolved_configuration
        actual = _config_files(self.arguments)
        expected = {item.role: {"path": str(item.path), "sha256": item.sha256}
                    for item in self.configuration_files}
        if _resolved_configuration(actual) != _resolved_configuration(expected):
            raise ValueError("automatic native preparation template changed configuration authority")
        return self

    def describe(self):
        return {"runner": self.runner, "arguments": list(self.arguments),
                "configuration_files": [item.describe() for item in self.configuration_files]}


@dataclass(frozen=True)
class EnsembleSourceContext:
    """Source facts the existing door actually knows, before member admission.

    A requested trajectory is a fetch request, not an assertion about supplied
    artifact bytes. A verified-acquisition context requires its real binding.
    An artifact without that authority retains ``trajectory=None``.
    """
    start: datetime
    end: datetime
    ordinary_inputs: object = field(repr=False, compare=False)
    trajectory: SourceTrajectory | None = None
    provenance: str = "supplied-artifact"
    source_binding: object | None = field(default=None, repr=False, compare=False)
    templates: tuple[NativeSourceTemplate, ...] = ()
    domains: tuple[DomainSourceContext, ...] = ()
    authorities: tuple[FileAuthority, ...] = ()
    p_top_pa: float | None = None
    preflight_options: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "start", _utc(self.start))
        object.__setattr__(self, "end", _utc(self.end))
        for name in ("templates", "domains", "authorities"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        controls = dict(self.preflight_options)
        if set(controls) - {"physics_profile", "expert_acknowledgements", "tiles", "devices", "devices_options",
                            "history_interval_seconds"}:
            raise ValueError("source context preflight options may only preserve ordinary forecast controls")
        object.__setattr__(self, "preflight_options", MappingProxyType(controls))
        if self.end <= self.start:
            raise ValueError("automatic source context needs a positive forecast interval")
        if self.provenance not in {"supplied-artifact", "requested-trajectory", "verified-acquisition"}:
            raise ValueError("automatic source context has unknown provenance")
        if (self.trajectory is None) != (self.provenance == "supplied-artifact"):
            raise ValueError("opaque artifacts cannot acquire an inferred source cycle")
        if self.provenance == "verified-acquisition" and self.source_binding is None:
            raise ValueError("verified source context requires its original acquisition binding")
        if self.provenance != "verified-acquisition" and self.source_binding is not None:
            raise ValueError("source acquisition binding requires verified-acquisition provenance")
        if len({item.runner for item in self.templates}) != len(self.templates):
            raise ValueError("automatic source context repeats a native runner template")
        if self.domains:
            known = {}
            for domain in self.domains:
                if domain.grid_id in known or (domain.parent_id is not None and domain.parent_id not in known):
                    raise ValueError("source domains must retain unique IDs in parent-first order")
                if not self.start <= domain.activation_time < self.end:
                    raise ValueError("source domain activation lies outside the requested forecast")
                if domain.parent_id is not None and domain.activation_time < known[domain.parent_id].activation_time:
                    raise ValueError("source child cannot activate before its parent")
                known[domain.grid_id] = domain
            if sum(domain.parent_id is None for domain in self.domains) != 1:
                raise ValueError("automatic source context needs exactly one hierarchy root")
        if self.p_top_pa is not None and (isinstance(self.p_top_pa, bool) or
                not math.isfinite(self.p_top_pa) or self.p_top_pa <= 0):
            raise ValueError("automatic source context needs a positive model-top pressure")

    def verify(self):
        for authority in (*self.authorities, *(item for domain in self.domains for item in domain.authorities)):
            authority.verify()
        for template in self.templates:
            template.verify()
        if self.source_binding is not None:
            receipt = self.source_binding.verify()
            if (self.source_binding.trajectory != self.trajectory or
                    receipt.get("trajectory_sha256") != self.trajectory.identity):
                raise ValueError("automatic source context differs from its verified acquisition")
        return self

    def template_for(self, trajectory):
        from gpuwm.source_adapters import get_source_adapter
        runner = get_source_adapter(trajectory.source).runner
        candidates = tuple(item for item in self.templates if item.runner == runner)
        if len(candidates) != 1:
            raise ValueError(f"{trajectory.source} needs the existing door's native controls for {runner}; "
                             "capture config, WPS, geography/static and preprocess arguments")
        return candidates[0].verify()

    def describe(self):
        return {"start": self.start.isoformat(), "end": self.end.isoformat(),
                "provenance": self.provenance,
                "trajectory": None if self.trajectory is None else {
                    "source": self.trajectory.source, "cycle": self.trajectory.cycle.isoformat(),
                    "member": self.trajectory.member, "identity": self.trajectory.identity},
                "source_binding": None if self.source_binding is None else {
                    "path": str(self.source_binding.path), "sha256": self.source_binding.sha256,
                    "verification_sha256": _digest(self.source_binding.verification)},
                "templates": [item.describe() for item in self.templates],
                "domains": [item.describe() for item in self.domains],
                "authorities": [item.describe() for item in self.authorities],
                "p_top_pa": self.p_top_pa, "preflight_options": _plain(self.preflight_options)}


@dataclass(frozen=True)
class SourceSelectionPolicy:
    """A byte-pinned measured-default document, or the unfitted built-in policy."""
    document_json: str
    authority: FileAuthority | None = None

    @classmethod
    def unfitted(cls):
        # Which sources need a fitted policy is a column of the source
        # table, so a source that needs one is a row and not an edit here.
        from gpuwm.source_adapters import source_adapters
        required = sorted(adapter.source_id for adapter in source_adapters()
                          if adapter.requires_ensemble_calibration)
        return cls(_canonical({"schema": POLICY_SCHEMA, "policy_id": "operational-and-reference-v1",
            "source_defaults": {}, "requires_calibration": required,
            "artifact_fallback": {"sppt": True}, "nonensemble_fallback": {"sppt": True}}))

    @classmethod
    def default(cls):
        """Load the committed measured policy when its evidence is shipped."""
        path = Path(__file__).with_name("source_defaults.json")
        return cls.load(path) if path.is_file() else cls.unfitted()

    @classmethod
    def load(cls, path, *, sha256=None):
        authority = FileAuthority.capture("source-policy", path)
        if sha256 is not None and authority.sha256 != sha256:
            raise ValueError("ensemble source policy differs from its requested digest")
        result = cls(_canonical(json.loads(authority.path.read_bytes())), authority)
        result.verify()
        return result

    @classmethod
    def from_training_selection(cls, training_selection, held_out_summary, *,
                                source_defaults, policy_id):
        """Build only from the selector's exact result and held-out evidence.

        Source applicability, stochastic parameter controls and multi-model
        roster rules are explicit inputs. The measured winner and amplitudes
        are copied from the frozen result and verified again on every load.
        """
        from gpuwm.ensemble.calibrated_policy import reference, validated_selection, frozen_stochastic_controls
        training = reference(training_selection)
        held_out = reference(held_out_summary)
        defaults = {}
        for source, controls in source_defaults.items():
            controls = json.loads(_canonical(controls))
            evidence = validated_selection(training, held_out,
                validation_cases=controls.get("validation_cases"))
            winner, result, campaign = evidence["winner"], evidence["selection"], evidence["campaign"]
            row = {**controls, "kind": winner["recipe"], "stochastic_amplitude": winner["stochastic_amplitude"],
                "calibrated_member_count": evidence["calibrated_member_count"],
                "calibration_evidence": training, "held_out_evidence": held_out,
                "validation_cases": evidence["validation_cases"]}
            row.setdefault("stochastic", frozen_stochastic_controls(winner["stochastic_amplitude"], campaign["stochastic_on"]))
            if winner["recipe"] == "recentered":
                row.update(amplitude=result["selected_recenter_amplitude"], donor_source=campaign["donor_source"])
            elif winner["recipe"] == "time-lagged":
                row["max_lag_hours"] = campaign["time_lag_max_age_hours"]
            defaults[source] = row
        document = SourceSelectionPolicy.unfitted().document
        document.update(policy_id=policy_id, source_defaults=defaults)
        return cls(_canonical(document)).verify()

    def write(self, path):
        """Persist reviewed defaults with a small portable evidence closure."""
        self.verify()
        from gpuwm.ensemble.calibrated_policy import portable_policy
        path = portable_policy(self.document, path,
            evidence_base=None if self.authority is None else self.authority.path.parent)
        return SourceSelectionPolicy.load(path)

    @property
    def document(self):
        return json.loads(self.document_json)

    @property
    def sha256(self):
        return _digest(self.document)

    def verify(self):
        from gpuwm.ensemble.stochastic_model import StochasticModelProvider
        from gpuwm.source_adapters import get_source_adapter
        if self.authority is not None:
            self.authority.verify()
            if _canonical(json.loads(self.authority.path.read_bytes())) != self.document_json:
                raise ValueError("ensemble source policy document differs from its pinned file")
        document = self.document
        if (document.get("schema") != POLICY_SCHEMA or not isinstance(document.get("policy_id"), str)
                or not document["policy_id"] or not isinstance(document.get("source_defaults"), dict)):
            raise ValueError("ensemble source policy has an unsupported schema")
        for key in ("artifact_fallback", "nonensemble_fallback"):
            fallback = document.get(key)
            if fallback is not None:
                provider = StochasticModelProvider.from_mapping(fallback)
                if provider.sppt is None and provider.skebs_psi is None:
                    raise ValueError("source fallback needs active model-error controls independent of a physics suite")
        required = document.get("requires_calibration", [])
        if (not isinstance(required, list) or len(set(required)) != len(required) or
                any(get_source_adapter(source).source_id != source for source in required)):
            raise ValueError("source policy calibration requirements must name distinct canonical sources")
        for source, row in document["source_defaults"].items():
            if get_source_adapter(source).source_id != source or not isinstance(row, dict):
                raise ValueError("source policy defaults must name canonical sources")
            if row.get("kind") not in {"input-ensemble", "recentered", "time-lagged", "multi-model"}:
                raise ValueError("measured source default must retain the selected scientific recipe")
            from gpuwm.ensemble.calibrated_policy import validate_policy_row
            validate_policy_row(row, base=None if self.authority is None else self.authority.path.parent)
            if row["kind"] == "recentered":
                SourceTrajectory(row["donor_source"], datetime(2026, 1, 1, tzinfo=timezone.utc))
                amplitude = row.get("amplitude")
                if (isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)) or
                        not math.isfinite(amplitude) or amplitude < 0):
                    raise ValueError("measured recentering default needs its finite nonnegative amplitude")
            if row.get("stochastic") is not None:
                StochasticModelProvider.from_mapping(row["stochastic"])
        return self


@dataclass(frozen=True)
class AutomaticSourceSelection:
    mode: str
    recipe: SourceRecipe | None
    context_sha256: str
    policy_sha256: str
    reason: str
    stochastic_json: str | None = None
    amplitude: float | None = None
    calibration_evidence_json: str | None = None

    @property
    def stochastic(self):
        return None if self.stochastic_json is None else json.loads(self.stochastic_json)

    def describe(self):
        return {"schema": SELECTION_SCHEMA, "mode": self.mode,
                "context_sha256": self.context_sha256, "policy_sha256": self.policy_sha256,
                "reason": self.reason, "recipe": None if self.recipe is None else self.recipe.describe(),
                "recipe_sha256": None if self.recipe is None else self.recipe.sha256,
                "stochastic": self.stochastic, "amplitude": self.amplitude,
                "calibration_evidence": None if self.calibration_evidence_json is None else
                                        json.loads(self.calibration_evidence_json)}

    @property
    def sha256(self):
        return _digest(self.describe())


def _donor_for_window(source, context, max_age):
    from gpuwm.source_cycles import cycle_grid_for
    if type(max_age) is not int or max_age < 0:
        raise ValueError("donor maximum cycle age must be a nonnegative whole number of hours")
    grid = cycle_grid_for(source)
    if grid is None:
        raise ValueError(f"{source} has no declared donor initialization cadence")
    anchor = context.start.replace(minute=0, second=0, microsecond=0)
    refusals = []
    for age in range(max_age + 1):
        cycle = anchor - timedelta(hours=age)
        if cycle.hour not in grid.hours:
            continue
        trajectory = SourceTrajectory(source, cycle)
        try:
            trajectory.window(context.start, context.end, native_bracketing=True)
        except ValueError as error:
            refusals.append(str(error))
            continue
        return trajectory
    detail = refusals[0] if refusals else "no declared cycle within the selected age limit"
    raise ValueError(f"{source} has no donor trajectory covering this forecast within {max_age} h: {detail}")


def resolve_ensemble_sources(request, context: EnsembleSourceContext, *,
                             policy: SourceSelectionPolicy | None = None,
                             recipe: SourceRecipe | None = None,
                             amplitude: float | None = None):
    """Resolve ordinary, operational, measured or explicitly labelled fallback.

    The ordinary singleton exits before policy I/O or source preparation.
    An explicitly supplied recipe retains its original indices and seeds.
    """
    from gpuwm.ensemble.request import EnsembleRequest
    from gpuwm.source_adapters import get_source_adapter
    request = EnsembleRequest.from_mapping(request)
    context_hash = _digest(context.describe())
    if request.members == 1 and recipe is None:
        return AutomaticSourceSelection("ordinary", None, context_hash, "", "automatic singleton uses ordinary inputs")
    context.verify()
    policy = (policy or SourceSelectionPolicy.default()).verify()
    stochastic = request.stochastic
    def selected(mode, result, reason, *, spread=None, evidence=None):
        if evidence is None:
            from gpuwm.ensemble.calibration_admission import refuse_uncalibrated_random
            refuse_uncalibrated_random(stochastic=stochastic)
        return AutomaticSourceSelection(mode, result, context_hash, policy.sha256, reason,
            None if stochastic is None else _canonical(stochastic), spread,
            None if evidence is None else _canonical(evidence))
    if recipe is not None:
        if (len(recipe.members) != request.members or recipe.start != context.start or
                recipe.end != context.end or context.trajectory is not None and recipe.base != context.trajectory):
            raise ValueError("explicit recipe differs from requested member count, source or forecast window")
        if recipe.kind == "recentered" and (isinstance(amplitude, bool) or
                not isinstance(amplitude, (int, float)) or not math.isfinite(amplitude) or amplitude < 0):
            raise ValueError("explicit recentered recipe requires its finite nonnegative amplitude")
        return selected("source-recipe", recipe, "explicit source recipe", spread=amplitude)
    if request.sources or request.perturbation is not None:
        return selected("caller-owned", None, "existing explicit source or perturbation provider owns member selection")
    if context.trajectory is None:
        if stochastic is None:
            stochastic = policy.document.get("artifact_fallback")
        if stochastic is None:
            raise ValueError("supplied artifact has no operational source trajectory and no declared stochastic fallback")
        return selected("stochastic", None,
            "supplied artifact has no verified operational trajectory; reference model-error fallback is not calibrated")
    trajectory = context.trajectory
    adapter = get_source_adapter(trajectory.source)
    ensemble_source = adapter.source_id if adapter.member_set else getattr(adapter, "ensemble_source", None)
    if ensemble_source is not None and get_source_adapter(ensemble_source).runnable:
        # Preserve the exact ordinary recipe diagnostics for incomplete
        # operational products, invalid member counts and source horizons.
        result = build_recipe(source=trajectory.source, cycle=trajectory.cycle,
            start=context.start, end=context.end, count=request.members, base_seed=request.base_seed)
        return selected("source-recipe", result, "source's declared operational ensemble")
    row = policy.document["source_defaults"].get(trajectory.source)
    if row is None:
        if stochastic is not None:
            return selected("stochastic", None, "explicit model-error controls with ordinary source inputs")
        if trajectory.source not in policy.document.get("requires_calibration", ()):
            stochastic = policy.document.get("nonensemble_fallback")
            if stochastic is not None:
                return selected("stochastic", None,
                    "source has no complete operational ensemble; reference model-error fallback is not calibrated")
        capability = ("" if ensemble_source is None else
            f"its declared {ensemble_source} ensemble lacks a complete initialization route "
            f"({get_source_adapter(ensemble_source).composition_requirement}); ")
        raise ValueError(f"{trajectory.source} has no operational ensemble and no fitted automatic source policy; "
                         + capability + "supply a time-lagged or multi-model source recipe")
    if stochastic is None:
        stochastic = row.get("stochastic")
    evidence = row["calibration_evidence"]
    donor = None if row["kind"] != "recentered" else _donor_for_window(
        row["donor_source"], context, row.get("max_donor_age_hours", 24))
    trajectories = ()
    count = request.members
    if row["kind"] == "multi-model":
        from gpuwm.ensemble.calibrated_policy import trajectories_for_rule
        trajectories = trajectories_for_rule(row["trajectory_rule"], trajectory.cycle)
        count = len(trajectories)
        if request.members > count:
            raise ValueError(f"calibrated multi-model roster supplies {count} distinct trajectories, "
                             f"but {request.members} were requested; repeating trajectories would fabricate ensemble size")
    result = build_recipe(source=trajectory.source, cycle=trajectory.cycle,
        start=context.start, end=context.end, count=count, base_seed=request.base_seed,
        kind=row["kind"], donor=donor, trajectories=trajectories, max_lag_hours=row.get("max_lag_hours", 24))
    if row["kind"] == "multi-model" and request.members < count:
        result = result.select_members(row["selection_order"][:request.members])
    from dataclasses import replace
    result = replace(result, calibration="policy:" + policy.sha256)
    return selected("source-recipe", result, "measured source policy",
                    spread=row.get("amplitude"), evidence=evidence)
