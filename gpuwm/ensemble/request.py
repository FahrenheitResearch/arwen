"""The source-neutral ensemble option carried by the production run plan."""
from __future__ import annotations

from dataclasses import dataclass, field
from operator import index

from gpuwm.ensemble.batch_products import default_product_requests, DEFAULT_THRESHOLDS


@dataclass(frozen=True)
class EnsembleRequest:
    members: int
    keep_member_files: bool = False
    thresholds: dict = field(default_factory=dict)
    sources: tuple = ()
    perturbation: dict | None = None
    base_seed: int = 0
    member_device_ids: tuple[int, ...] | None = None
    retain_member_diagnostics: bool = False
    stochastic: dict | None = None
    #: Member source recipe this request runs: ``time-lagged`` (earlier
    #: cycles of the config's own source) or ``multi-model`` (the listed
    #: trajectories).  Real source trajectories, never random amplitudes.
    recipe: str | None = None
    trajectories: tuple = ()

    def __post_init__(self):
        if self.perturbation == "none":
            # The 2.8.4 spelling of no perturbation, same as an absent key.
            object.__setattr__(self, "perturbation", None)
        from gpuwm.ensemble.calibration_admission import refuse_uncalibrated_random
        refuse_uncalibrated_random(perturbation=self.perturbation, stochastic=self.stochastic)
        if isinstance(self.members, bool):
            raise ValueError("ensemble members must be a positive integer")
        members = index(self.members)
        if members < 1:
            raise ValueError("ensemble members must be a positive integer")
        object.__setattr__(self, "members", members)
        if not isinstance(self.keep_member_files, bool):
            raise ValueError("keep_member_files must be true or false")
        if not isinstance(self.retain_member_diagnostics, bool):
            raise ValueError("retain_member_diagnostics must be true or false")
        if (isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int)
                or self.base_seed < 0):
            raise ValueError("ensemble base_seed must be a nonnegative integer")
        thresholds = dict(self.thresholds)
        # The reduction contract validates finite FP32 thresholds and units.
        requests, _ = default_product_requests(DEFAULT_THRESHOLDS, thresholds=thresholds)
        object.__setattr__(self, "thresholds", {
            request.field: list(request.thresholds) for request in requests
            if request.field in thresholds})
        sources = tuple(self.sources)
        if sources and len(sources) != members:
            raise ValueError("ensemble sources must supply one descriptor per member")
        object.__setattr__(self, "sources", sources)
        trajectories = tuple(self.trajectories)
        if self.recipe is not None:
            from gpuwm.ensemble.recipe_door import RECIPES
            if self.recipe not in RECIPES:
                raise ValueError(f"ensemble recipe must be one of {list(RECIPES)}, got {self.recipe!r}")
            if sources or self.perturbation is not None:
                raise ValueError("an ensemble recipe selects every member's source; "
                                 "remove [ensemble] sources and perturbation")
        if trajectories:
            if self.recipe != "multi-model":
                raise ValueError('ensemble trajectories belong to recipe = "multi-model"')
            for item in trajectories:
                if (not isinstance(item, dict) or not {"source", "cycle"} <= item.keys()
                        or set(item) - {"source", "cycle", "member"}):
                    raise ValueError("each ensemble trajectory needs a source and a cycle "
                                     "(and a member where its source has members)")
            if len(trajectories) < members:
                raise ValueError(f"multi-model lists {len(trajectories)} trajectories, but {members} members "
                                 "were requested; repeating a trajectory would fabricate ensemble size")
        elif self.recipe == "multi-model":
            raise ValueError("a multi-model ensemble needs its trajectory list: "
                             "--trajectories FILE or [ensemble] trajectories")
        object.__setattr__(self, "trajectories", tuple(
            {**item, "cycle": item["cycle"] if isinstance(item["cycle"], str)
             else item["cycle"].isoformat()} for item in trajectories))
        if self.perturbation is not None and not isinstance(self.perturbation, dict):
            # Breakage it prevents: a provider name is bound only by the
            # tools.ensemble_forecast overlay command.  This request binds
            # none, so every member would run unperturbed under that name.
            from gpuwm.ensemble_admission import provider_name_refusal
            raise ValueError(provider_name_refusal(self.perturbation))
        if self.stochastic is not None:
            from gpuwm.ensemble.stochastic_model import StochasticModelProvider
            StochasticModelProvider.from_mapping(self.stochastic)
        if self.member_device_ids is not None:
            ids = tuple(self.member_device_ids)
            if (not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("member_device_ids must name distinct nonnegative cards")
            object.__setattr__(self, "member_device_ids", ids)

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            from gpuwm.ensemble.calibration_admission import refuse_uncalibrated_random
            refuse_uncalibrated_random(perturbation=value.perturbation, stochastic=value.stochastic)
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(value)
        if not isinstance(value, dict):
            raise ValueError("ensemble must be a member count or an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        from gpuwm.ensemble_admission import OVERLAY_MARKERS, TABLE_KEYS, overlay_table_refusal
        if unknown & OVERLAY_MARKERS:
            # The 2.8.4 overlay file has the same table name and another
            # schema.  Breakage it prevents: this door would read none of
            # it, and the run would not be the ensemble the file describes.
            raise ValueError(overlay_table_refusal(unknown & OVERLAY_MARKERS))
        if unknown:
            raise ValueError(f"unknown ensemble options: {sorted(unknown)}; "
                             f"the [ensemble] table takes {TABLE_KEYS}")
        if value.get("trajectories") and value.get("recipe") is None:
            # A trajectory list is the multi-model recipe.
            value = {**value, "recipe": "multi-model"}
        if "members" not in value:
            if value.get("recipe") == "multi-model" and value.get("trajectories"):
                # Every listed trajectory is one member.
                value = {**value, "members": len(value["trajectories"])}
            else:
                raise ValueError("ensemble requires members")
        return cls(**value)

    def receipt(self):
        return {"members": self.members, "keep_member_files": self.keep_member_files,
                "retain_member_diagnostics": self.retain_member_diagnostics,
                "stochastic": self.stochastic,
                "thresholds": self.thresholds, "sources": list(self.sources),
                "perturbation": self.perturbation, "base_seed": self.base_seed,
                "member_device_ids": (None if self.member_device_ids is None
                                      else list(self.member_device_ids)),
                # Stated only when a recipe is selected, so a request without
                # one keeps the receipt every earlier manifest recorded.
                **({} if self.recipe is None else {"recipe": self.recipe,
                    "trajectories": [dict(item) for item in self.trajectories]})}
