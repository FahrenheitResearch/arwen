"""Resolving the perturbation hook by provenance reference (EXPERIMENTAL).

The initial-condition perturbations themselves are another lane's work
and live in ``gpuwm.da.perturb`` behind::

    apply_perturbations(state, seed, cfg)

This module resolves a provenance string from ``[ensemble] perturbation``
to a callable with that signature and records what it resolved to, so a
member's manifest entry always says which code perturbed it.

Three references are accepted:

``none``
    No perturbation.  Members differ only in what the model itself
    derives from the seed.  Useful for proving the engine's plumbing.
``experimental-stub``
    The local stand-in below.  It is *not* science: it adds seeded
    Gaussian noise to one named field and says so in provenance.
``gpuwm.da.perturb``
    The real hook.  If the module or its entry point is absent the
    resolution fails closed rather than quietly degrading to the stub --
    an ensemble that believes it was perturbed and was not is worse than
    one that refuses to start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

NONE = "none"
STUB = "experimental-stub"
DA_LANE = "gpuwm.da.perturb"

#: Entry point every perturbation provider must expose.
ENTRY_POINT = "apply_perturbations"

#: The stub perturbs this field only, and only when an amplitude is
#: given.  Named in options, not hard-coded to a case or a source.
_STUB_DEFAULT_FIELD = "thp"


#: Keys the engine supplies from the domain the member runs on.  An
#: overlay that also names them is a refusal, not an override: the
#: grid spacing is a property of the grid, and a second copy in the
#: ensemble TOML is a place for the two to disagree silently -- with the
#: length scale, which is expressed in kilometres, absorbing the error.
GRID_SUPPLIED_KEYS = ("dx_km", "dy_km")


@dataclass(frozen=True)
class ResolvedPerturbation:
    """A perturbation hook plus the provenance to record for it."""

    provenance: str
    apply: Callable[..., object]
    detail: Mapping[str, object]

    def __call__(self, state, seed: int, cfg: Mapping[str, object],
                 *, grid_spacing_km=None):
        return self.apply(state, seed, cfg, grid_spacing_km=grid_spacing_km)


def _noop(state, seed, cfg, *, grid_spacing_km=None):  # noqa: ARG001
    return {"applied": False, "reason": "perturbation = none"}


def _stub(state, seed: int, cfg: Mapping[str, object], *,
          grid_spacing_km=None):  # noqa: ARG001 - uniform hook signature
    """Seeded Gaussian noise on one field.  A stand-in, not a method.

    Deterministic in ``seed`` alone: the same seed produces the same
    noise on the same grid, which is what the engine's byte-identity
    guarantee rests on.
    """
    import numpy as np

    options = dict(cfg or {})
    field = str(options.get("field", _STUB_DEFAULT_FIELD))
    amplitude = float(options.get("amplitude", 0.0))
    target = getattr(state, field, None)
    if target is None:
        raise ValueError(
            f"{STUB} is configured to perturb field {field!r}, which the "
            "state does not carry; set "
            "[ensemble.perturbation_options] field = \"<name>\"")
    if amplitude == 0.0:
        return {"applied": False, "field": field, "amplitude": 0.0,
                "reason": "amplitude 0 leaves the state untouched"}
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError(
            f"{STUB} amplitude must be finite and non-negative, got "
            f"{amplitude!r}")
    shape = tuple(np.shape(target))
    generator = np.random.default_rng(int(seed))
    noise = generator.standard_normal(shape) * amplitude

    from gpuwm.ensemble.increments import apply_increments

    receipt = apply_increments(state, {field: noise})
    return {"applied": True, "field": field, "amplitude": amplitude,
            "increment_receipt": receipt}


def _da_lane_adapter(module, entry) -> Callable[..., object]:
    """Bridge the engine's options table to ``gpuwm.da.perturb``'s config.

    Two things the two lanes did not agree on until assembly:

    * ``apply_perturbations`` takes a ``PerturbationConfig`` and refuses a
      plain mapping by design -- the engine's ``[ensemble.
      perturbation_options]`` is a TOML table, so something has to call
      ``PerturbationConfig.from_mapping``.  It is this adapter, so the
      refusal an operator sees names their TOML key rather than a type.
    * ``dx_km``/``dy_km`` are required by that config and are properties of
      the domain, not of the ensemble overlay.  The engine supplies them
      and **refuses** an overlay that also states them, because a length
      scale in kilometres against the wrong grid spacing is a wrong
      perturbation that still runs.
    """

    def apply(state, seed: int, cfg: Mapping[str, object], *,
              grid_spacing_km=None):
        options = dict(cfg or {})
        stated = sorted(key for key in GRID_SUPPLIED_KEYS if key in options)
        if stated:
            raise ValueError(
                f"[ensemble.perturbation_options] states "
                f"{', '.join(stated)}, which the engine supplies from the "
                "domain the member runs on. Remove it: two sources for the "
                "grid spacing is a place for them to disagree, and the "
                "length_scale_km would silently absorb the difference.")
        if grid_spacing_km is None:
            raise ValueError(
                f"{DA_LANE} needs the member's grid spacing and none was "
                "supplied; the engine passes it from the domain config, so "
                "a caller reaching this has bypassed run_member")
        dx_km, dy_km = (float(value) for value in grid_spacing_km)
        options["dx_km"] = dx_km
        options["dy_km"] = dy_km
        config = module.PerturbationConfig.from_mapping(options)
        return entry(state, seed, config)

    return apply


def resolve_perturbation(provenance: str) -> ResolvedPerturbation:
    """Resolve a ``[ensemble] perturbation`` reference.  Fails closed."""
    reference = (provenance or "").strip()
    if reference == NONE:
        return ResolvedPerturbation(
            provenance=NONE, apply=_noop,
            detail={"kind": "none", "stability": "experimental"})
    if reference == STUB:
        return ResolvedPerturbation(
            provenance=STUB, apply=_stub,
            detail={
                "kind": "stub",
                "stability": "experimental",
                "warning": ("seeded Gaussian noise on one field; a "
                            "placeholder for gpuwm.da.perturb, not an "
                            "initial-condition perturbation method"),
            })
    if reference == DA_LANE:
        try:
            from gpuwm.da import perturb as da_perturb
        except ImportError as error:
            raise ValueError(
                f"[ensemble] perturbation = {DA_LANE!r} but that module "
                f"is not importable ({error}). It is delivered by the "
                "perturbation lane. Until it lands, either wait or set "
                f"perturbation = {STUB!r} / {NONE!r} deliberately -- the "
                "engine will not silently substitute one for the other."
            ) from error
        entry = getattr(da_perturb, ENTRY_POINT, None)
        if not callable(entry):
            raise ValueError(
                f"{DA_LANE} exposes no callable {ENTRY_POINT}"
                f"(state, seed, cfg); the interface contract requires it")
        return ResolvedPerturbation(
            provenance=DA_LANE, apply=_da_lane_adapter(da_perturb, entry),
            detail={"kind": "module", "module": DA_LANE,
                    "entry_point": ENTRY_POINT,
                    "config_from": "PerturbationConfig.from_mapping",
                    "grid_supplied_keys": list(GRID_SUPPLIED_KEYS),
                    "module_file": getattr(da_perturb, "__file__", None)})
    raise ValueError(
        f"unknown perturbation reference {provenance!r}; known references "
        f"are {NONE!r}, {STUB!r} and {DA_LANE!r}")
