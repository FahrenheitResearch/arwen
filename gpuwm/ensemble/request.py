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
    #: Source trajectories or explicitly declared surface/land member arms.
    recipe: str | None = None
    trajectories: tuple = ()
    #: Named existing-land and fixed surface-state arms on selected sources.
    member_variants: tuple = ()
    family_weights: dict | str | None = None
    provenance: tuple = ()
    member_fetches: tuple = ()
    max_ordinary_members_per_device: int | None = None
    hydro: dict | None = None
    sounding_points: tuple = ()
    cyclone_products: dict | None = None
    comparison: dict | str | None = None

    def __post_init__(self):
        if self.hydro is not None:
            from gpuwm.ensemble.optional_routing import product_options
            object.__setattr__(self, "hydro", product_options(self.hydro))
        if self.member_fetches and (self.recipe != "wofs" or len(self.member_fetches) != self.members):
            raise ValueError("member_fetches needs one acquisition per WoFS member")
        if self.provenance and len(self.provenance) != self.members:
            raise ValueError("provenance needs one label per ensemble member")
        if self.provenance:
            from gpuwm.ensemble.lineage import validate_record
            for member, row in enumerate(self.provenance):
                validate_record(row, member)
        if not isinstance(self.sounding_points, (list, tuple)):
            raise ValueError("sounding_points must be a list of LAT,LON or NAME=LAT,LON points")
        from gpuwm.soundings import point
        for value in self.sounding_points:
            if not isinstance(value, str):
                raise ValueError("sounding_points entries must be LAT,LON or NAME=LAT,LON strings")
            point(value)
        object.__setattr__(self, "sounding_points", tuple(self.sounding_points))
        if self.perturbation == "none":
            # The 2.8.4 spelling of no perturbation, same as an absent key.
            object.__setattr__(self, "perturbation", None)
        from gpuwm.ensemble.calibration_admission import refuse_uncalibrated_random
        refuse_uncalibrated_random(perturbation=self.perturbation, stochastic=self.stochastic)
        from gpuwm.ensemble.surface_controls import is_surface_recipe, validate_surface_recipe
        if is_surface_recipe(self.perturbation):
            object.__setattr__(self, "perturbation", validate_surface_recipe(self.perturbation))
        from gpuwm.ensemble.initial_perturbation import is_initial_perturbation, validate_initial_perturbation
        if is_initial_perturbation(self.perturbation):
            object.__setattr__(self, "perturbation", validate_initial_perturbation(self.perturbation))
        if isinstance(self.members, bool):
            raise ValueError("ensemble members must be a positive integer")
        members = index(self.members)
        if members < 1:
            raise ValueError("ensemble members must be a positive integer")
        object.__setattr__(self, "members", members)
        if not isinstance(self.keep_member_files, bool):
            raise ValueError("keep_member_files must be true or false")
        if self.cyclone_products is not None and not isinstance(self.cyclone_products, dict):
            raise ValueError("cyclone_products must be a product request object")
        if self.sounding_points or self.cyclone_products is not None:
            object.__setattr__(self, "keep_member_files", True)
        if not isinstance(self.retain_member_diagnostics, bool):
            raise ValueError("retain_member_diagnostics must be true or false")
        if (isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int)
                or self.base_seed < 0):
            raise ValueError("ensemble base_seed must be a nonnegative integer")
        thresholds = dict(self.thresholds)
        # The reduction contract validates finite FP32 thresholds and units.
        requests, _ = default_product_requests(DEFAULT_THRESHOLDS, thresholds=thresholds, comparison=self.comparison)
        if isinstance(self.comparison, dict):
            object.__setattr__(self, "comparison", dict(self.comparison))
        object.__setattr__(self, "thresholds", {
            request.field: list(request.thresholds) for request in requests
            if request.field in thresholds})
        sources = tuple(self.sources)
        family_specs = bool(sources) and all(isinstance(row, dict) and "members" in row for row in sources)
        if family_specs:
            from gpuwm.ensemble.wofs_recipe import parse_source_spec
            sources = parse_source_spec(sources, members)
        if sources and not family_specs and len(sources) != members:
            raise ValueError("ensemble sources must supply one descriptor per member")
        object.__setattr__(self, "sources", sources)
        trajectories = tuple(self.trajectories)
        if not isinstance(self.member_variants, (tuple, list)):
            raise ValueError("member_variants must be a list of named member records")
        variants = tuple(self.member_variants)
        # Distinct source trajectories do not need artificial physics changes.
        if not variants and self.recipe == "member-roster" and len(trajectories) == members:
            variants = tuple({"name": f"trajectory-{i:03d}"} for i in range(members))
        if variants and self.recipe is None:
            object.__setattr__(self, "recipe", "member-roster")
        if variants or self.recipe in ("member-roster", "wofs") and not family_specs:
            if self.recipe not in ("member-roster", "wofs"):
                raise ValueError('ensemble member_variants belong to recipe = "member-roster"')
            if self.perturbation is not None:
                raise ValueError("member-roster records each member's fixed surface controls; "
                                 "remove the shared ensemble perturbation to prevent two competing surface states")
            from gpuwm.ensemble.member_variants import normalize_member_variants
            variants = normalize_member_variants(variants, members, ladder=self.recipe == "wofs" or bool(trajectories))
        object.__setattr__(self, "member_variants", variants)
        if self.recipe is not None:
            from gpuwm.ensemble.recipe_door import RECIPES
            if self.recipe not in RECIPES:
                raise ValueError(f"ensemble recipe must be one of {list(RECIPES)}, got {self.recipe!r}")
            if (sources and not family_specs) or (self.perturbation is not None and not is_surface_recipe(self.perturbation)):
                raise ValueError("an ensemble recipe selects every member's source; "
                                 "remove [ensemble] sources and perturbation")
            if self.recipe == "surface-state":
                from gpuwm.ensemble.surface_controls import shared_surface_options
                shared_surface_options(self.perturbation, members)
        if trajectories:
            if self.recipe not in ("multi-model", "member-roster", "wofs"):
                raise ValueError('ensemble trajectories belong to recipe = "multi-model" or "member-roster"')
            for item in trajectories:
                if (not isinstance(item, dict) or not {"source", "cycle"} <= item.keys()
                        or set(item) - {"source", "cycle", "member", "model", "parent_receipt", "weight", "stratum", "track_member_id", "tracks", "track_id", "configuration", "configuration_sha256", "fetch_root"}):
                    raise ValueError("each ensemble trajectory needs a source and a cycle "
                                     "(and a member where its source has members)")
                if "fetch_root" in item and (not isinstance(item["fetch_root"], str) or not item["fetch_root"].strip()):
                    raise ValueError("trajectory fetch_root must name a completed fetch directory")
            if len(trajectories) < members:
                raise ValueError(f"multi-model lists {len(trajectories)} trajectories, but {members} members "
                                 "were requested; repeating a trajectory would fabricate ensemble size")
            if self.recipe == "member-roster" and len(trajectories) != members:
                raise ValueError("member-roster needs one trajectory for each named arm; "
                                 "extra trajectories would leave source-to-arm ownership ambiguous")
            weighted = [item for item in trajectories if "weight" in item]
            if weighted:
                import math
                if len(weighted) != len(trajectories) or any(
                        isinstance(item["weight"], bool) or not isinstance(item["weight"], (int, float))
                        or not math.isfinite(item["weight"]) or item["weight"] <= 0 for item in weighted):
                    raise ValueError("supply finite positive weights for every trajectory")
                if not math.isclose(math.fsum(item["weight"] for item in weighted), 1., abs_tol=1e-12):
                    raise ValueError("trajectory weights must sum to one")
        elif self.recipe == "multi-model":
            raise ValueError("a multi-model ensemble needs its trajectory list: "
                             "--trajectories FILE or [ensemble] trajectories")
        from gpuwm.ensemble.lineage import parent_receipt, model_identity
        object.__setattr__(self, "trajectories", tuple(
            {**item, "source": model_identity(item["source"], item["model"])[1] if item.get("model") is not None else item["source"], **({"parent_receipt": parent_receipt(item["parent_receipt"])} if "parent_receipt" in item else {}), "cycle": item["cycle"] if isinstance(item["cycle"], str)
             else item["cycle"].isoformat()} for item in trajectories))
        from gpuwm.ensemble.lineage import family_weights, trajectory_records, weight_records
        if self.provenance:
            records, weights = weight_records(self.provenance, self.family_weights)
        elif trajectories:
            records, weights = trajectory_records(self.trajectories[:members], base_seed=self.base_seed,
                variants=variants, stochastic=self.stochastic, weights=self.family_weights)
        else:
            records, weights = (), family_weights(self.family_weights)
        from gpuwm.ensemble.lineage import record_stochastic
        records = record_stochastic(records, self.stochastic)
        object.__setattr__(self, "provenance", records)
        object.__setattr__(self, "family_weights", weights)
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
        if self.max_ordinary_members_per_device is not None:
            from gpuwm.ensemble.admission import _integer
            object.__setattr__(self, "max_ordinary_members_per_device", _integer(
                self.max_ordinary_members_per_device, "max_ordinary_members_per_device", positive=True))

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            from gpuwm.ensemble.calibration_admission import refuse_uncalibrated_random
            refuse_uncalibrated_random(perturbation=value.perturbation, stochastic=value.stochastic)
            if value.member_variants or value.hydro is not None:
                # Roster records contain mutable mappings. Revalidate and
                # detach them at the run door, as the initial reader does.
                return cls(**value.receipt())
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
        if value.get("sources") and all(isinstance(row, dict) and "members" in row for row in value["sources"]):
            from gpuwm.ensemble.wofs_recipe import parse_source_spec
            specs = parse_source_spec(value["sources"], value.get("members"))
            value = {**value, "sources": specs, "members": sum(row["members"] for row in specs),
                     "recipe": value.get("recipe", "wofs")}
        if value.get("trajectories") and value.get("recipe") is None:
            # A trajectory list is the multi-model recipe.
            value = {**value, "recipe": "multi-model"}
        if value.get("member_variants") and value.get("recipe") is None:
            value = {**value, "recipe": "member-roster"}
        if "member_variants" in value and not isinstance(value["member_variants"], (tuple, list)):
            raise ValueError("member_variants must be a list of named member records")
        if "members" not in value:
            if value.get("recipe") == "multi-model" and value.get("trajectories"):
                # Every listed trajectory is one member.
                value = {**value, "members": len(value["trajectories"])}
            elif value.get("recipe") == "member-roster" and value.get("member_variants"):
                value = {**value, "members": len(value["member_variants"])}
            else:
                raise ValueError("ensemble requires members")
        return cls(**value)

    def receipt(self):
        return {**({"cyclone_products": dict(self.cyclone_products)} if self.cyclone_products is not None else {}),
                **({"sounding_points": list(self.sounding_points)} if self.sounding_points else {}),
                "members": self.members, "keep_member_files": self.keep_member_files,
                "retain_member_diagnostics": self.retain_member_diagnostics,
                "stochastic": self.stochastic,
                "thresholds": self.thresholds, "sources": list(self.sources),
                **({"comparison": self.comparison} if self.comparison is not None else {}),
                "perturbation": self.perturbation, "base_seed": self.base_seed,
                "member_device_ids": (None if self.member_device_ids is None
                                      else list(self.member_device_ids)),
                **({} if self.max_ordinary_members_per_device is None else
                   {"max_ordinary_members_per_device": self.max_ordinary_members_per_device}),
                **({} if self.hydro is None else {"hydro": dict(self.hydro)}),
                # Stated only when a recipe is selected, so a request without
                # one keeps the receipt every earlier manifest recorded.
                **({} if self.recipe is None else {"recipe": self.recipe,
                    "trajectories": [dict(item) for item in self.trajectories]}),
                **({} if not self.member_variants else {"member_variants": [
                    {"name": item["name"], "physics": dict(item["physics"]),
                     "surface": dict(item["surface"]), **({"wind": item["wind"]} if item.get("wind") else {})} for item in self.member_variants]}),
                **({"family_weights": dict(self.family_weights)} if self.family_weights else {}),
                **({"provenance": list(self.provenance)} if self.provenance else {}),
                **({"member_fetches": list(self.member_fetches)} if self.member_fetches else {})}
