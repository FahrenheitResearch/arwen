"""Incremental analysis update: the analysis increment as a forcing.

The default cycle adds the whole analysis increment to the restored member
at the leg start, in one step.  Fields the analysis moved and fields it did
not are then out of balance with each other at the first model step, and the
model sheds the difference as gravity waves and spurious convection in the
first minutes of the forecast (Bloom et al. 1996; Lei and Whitaker 2016).
IAU instead adds the increment a fraction at a time over a window, as an
extra tendency the model adjusts to while it integrates.

What this module does
---------------------
:class:`IncrementalUpdate` wraps the domain's step function (the
``steppers`` seam of :func:`gpuwm.core.model.execute_experiment`, which binds
``gpuwm.core.dycore.step`` itself when nothing is configured).  After each
model step whose interval overlaps the window ``[t0, t0 + window]`` it adds
``increment * overlap / window`` to every analysed field, keeps the moisture
species non-negative, and refreshes the diagnosed pressure and density from
the updated prognostics (``gpuwm.core.diagnostics.update_diagnostics``, the
same post-condition every applier honours).  The weights are measured from
the step's own clock, so they sum to one exactly over the window whatever the
step length, and an adaptive step needs nothing extra.

The window is forward (``t0`` = the analysis time, the leg start): the
analysis is computed from the background at ``t0`` and spread over the first
``window`` seconds of the next leg.  A window centred on the analysis time
(-15 to +15 min) would need each member's state at ``t0 - 15 min``, a second
restart inside the leg that ends at the analysis; that is the declared
difference from the four-dimensional IAU of the global systems.

Nothing here changes a run that does not ask for it: without a window the
cycle applies the increment at the leg start as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

#: Fields kept non-negative after each fraction (the species and moments
#: a scheme carries; anything absent from a state is skipped).
NON_NEGATIVE_PREFIXES = ("q", "n")
SIGNED_FIELDS = ("thp", "u", "v", "w", "php", "mup", "ph", "mu")


class IauError(ValueError):
    """A refusal by this module."""


#: The fields IAU spreads by default: the dynamics.  Hydrometeor mass and
#: number increments are applied at the analysis time (the HRRR cloud
#: analysis puts its hydrometeors in at once, too).
DYNAMICS_FIELDS = ("thp", "qv", "u", "v", "w", "php", "mup")
IAU_FIELD_SETS = ("dynamics", "all")


#: How an analysis increment enters the next leg (tools/da_cycle_prepared
#: --iau-mode): all at its start; a fraction after each step
#: (:class:`IncrementalUpdate`); a held tendency inside every RK stage over
#: a forward window (:class:`IauForcing`, one box node); or the same over a
#: window centred on the analysis time, restarted from the member's own
#: trajectory at its start, with one node per analysed slot (4d).
IAU_MODES = ("oneshot", "split", "3d", "4d")


def resolve_iau_mode(mode, window_seconds) -> str:
    """The mode a run takes: the one asked for, else ``split`` when only a
    window is given (the flag's meaning before modes existed), else
    ``oneshot``.  A windowed mode without a window is a refusal."""
    if mode is None:
        return "split" if window_seconds is not None else "oneshot"
    if mode not in IAU_MODES:
        raise IauError(f"IAU mode {mode!r} is not one of {IAU_MODES}")
    if mode != "oneshot" and window_seconds is None:
        raise IauError(f"IAU mode {mode!r} needs --iau-window-seconds")
    if mode == "oneshot" and window_seconds is not None:
        raise IauError("--iau-mode oneshot with a window: the window would "
                       "be ignored")
    return mode


def split_fields(increments, which: str = "dynamics"):
    """``(spread, at_once)``: the increments IAU spreads and the rest."""
    if which not in IAU_FIELD_SETS:
        raise IauError(f"IAU field set {which!r} is not one of {IAU_FIELD_SETS}")
    if which == "all":
        return dict(increments), {}
    spread = {k: v for k, v in increments.items() if k in DYNAMICS_FIELDS}
    at_once = {k: v for k, v in increments.items() if k not in DYNAMICS_FIELDS}
    return spread, at_once


def _non_negative(name: str) -> bool:
    return name not in SIGNED_FIELDS and name.startswith(NON_NEGATIVE_PREFIXES)


@dataclass
class IncrementalUpdate:
    """Spread ``increments`` over ``[start_seconds, start_seconds + window]``.

    ``increments`` are host or device arrays shaped as the state's fields
    (winds on faces), the same mapping the cycle would hand
    :func:`gpuwm.ensemble.increments.apply_increments`.
    """

    increments: Mapping[str, object]
    start_seconds: float
    window_seconds: float
    hypsometric_opt: int = 2
    applied_fraction: float = 0.0
    steps: int = 0
    clipped: dict = field(default_factory=dict)
    _device: dict | None = None

    def __post_init__(self):
        if not np.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise IauError(
                f"IAU window {self.window_seconds!r} s must be finite and "
                "positive; without a window the increment is applied at the "
                "leg start")
        if not self.increments:
            raise IauError("IAU was given no increment to spread")

    @property
    def end_seconds(self) -> float:
        return float(self.start_seconds) + float(self.window_seconds)

    def _upload(self, state):
        if self._device is not None:
            return self._device
        device = {}
        for name, value in self.increments.items():
            target = getattr(state, name, None)
            if target is None:
                raise IauError(
                    f"the increment names {name!r}, which this state does "
                    "not carry")
            if tuple(np.shape(target)) != tuple(np.shape(value)):
                raise IauError(
                    f"increment {name!r} is {np.shape(value)}, the state's "
                    f"field is {np.shape(target)}")
            if hasattr(target, "__cuda_array_interface__"):
                import cupy as cp
                device[name] = cp.asarray(value, dtype=cp.float64)
            else:
                device[name] = np.asarray(value, dtype=np.float64)
        self._device = device
        return device

    def apply_fraction(self, state, fraction: float) -> None:
        """Add ``fraction`` of the increment to ``state`` and refresh the
        diagnosed fields."""
        from gpuwm.core.diagnostics import update_diagnostics

        if fraction <= 0.0:
            return
        device = self._upload(state)
        for name in sorted(device):
            target = getattr(state, name)
            xp = np
            if hasattr(target, "__cuda_array_interface__"):
                import cupy as xp  # noqa: N813
            updated = target.astype(xp.float64) + fraction * device[name]
            if _non_negative(name):
                negative = updated < 0
                count = int(negative.sum())
                if count:
                    self.clipped[name] = self.clipped.get(name, 0) + count
                    updated = xp.where(negative, 0.0, updated)
            target[...] = updated.astype(target.dtype)
        update_diagnostics(state, int(self.hypsometric_opt))
        self.applied_fraction += float(fraction)

    def stepper(self, step):
        """``step`` wrapped: the model step, then this step's fraction."""

        def iau_step(state, run, **kwargs):
            before = float(state.elapsed_seconds)
            result = step(state, run, **kwargs)
            after = float(state.elapsed_seconds)
            if after <= before:
                # the executor refreshes the clock after the solve; a step
                # that did not advance it is read from the run's dt
                after = before + float(getattr(run, "dt"))
            overlap = (min(after, self.end_seconds)
                       - max(before, float(self.start_seconds)))
            if overlap > 0 and self.applied_fraction < 1.0:
                fraction = min(overlap / float(self.window_seconds),
                               1.0 - self.applied_fraction)
                self.apply_fraction(state, fraction)
                self.steps += 1
            return result

        return iau_step

    def finish(self, state) -> dict:
        """Apply whatever the window did not reach (a leg shorter than the
        window) and return the receipt."""
        remainder = 1.0 - self.applied_fraction
        if remainder > 1e-12:
            self.apply_fraction(state, remainder)
        return self.receipt(remainder_at_end=max(remainder, 0.0))

    def receipt(self, **extra) -> dict:
        return {
            "schema": "gpuwm-da.iau.v1",
            "window_seconds": [float(self.start_seconds), self.end_seconds],
            "fields": sorted(self.increments),
            "steps_forced": int(self.steps),
            "applied_fraction": float(self.applied_fraction),
            "non_negative_clips": dict(self.clipped),
            "rule": ("after each model step, increment * overlap(step, "
                     "window) / window; moisture species clipped at zero; "
                     "pressure and density re-diagnosed"),
            **extra,
        }


# ---------------------------------------------------------------------------
# The increment as a tendency inside the RK step (DA design E1, lane 7)
# ---------------------------------------------------------------------------
#
# :class:`IncrementalUpdate` above adds each step's fraction AFTER the step,
# so the dynamics meet every fraction as a small insertion.  The forcing
# below is WRF's analysis-nudging slot instead (RTHNDGDTEN and kin: a held
# coupled tendency added in rk_addtend_dry on every RK stage and into the
# scalar update): the acoustic loop sees the increment as a rate while it
# integrates, the way it sees the physics, and the state is never written
# outside the step.
#
# One forcing holds one or more NODES.  A node is an increment and a
# piecewise-linear time density (1/s) that integrates to the node's share
# of one; a step [t, t + dt] adds ``increment * integral(density, t, t +
# dt)`` through a rate held over the step's three RK stages.  The weights
# are integrals of the density over the step's own clock, so they sum to the
# node's share exactly whatever the step length, adaptive steps included.
#
#   3D-IAU over W from t0:      one node, box density 1/W on [t0, t0 + W].
#   centred 3D-IAU:             one node, box on [ta - W/2, ta + W/2].
#   4D-IAU over [A, B]:         one node per slot time s_j, the linear
#                               interpolation basis on the s_j (constant
#                               beyond the end slots) divided by B - A; the
#                               bases sum to one, so the forcing at time tau
#                               is the slot increments interpolated to tau,
#                               over B - A (Lei and Whitaker 2016).
#
# Coupling follows the physics couplers (gpuwm.core.physics): theta and the
# scalars by the time-t column mass ``c1h*mu + c2h`` with the physical outer
# ring excluded on specified or nested domains (WRF add_a2a), theta divided
# by msft; the winds by their face mass and map factor with the outer faces
# excluded (add_a2c_u/v); ``php`` and ``w`` by the full-level mass
# ``c1f*mu + c2f`` over msft (the convention rph_t/rw_t carry).

#: The state attribute the dycore reads the forcing from.
IAU_STATE_ATTRIBUTE = "_iau_forcing"

IAU_FORCING_SCHEMA = "gpuwm-da.iau-forcing.v1"

#: Fields the forcing drives through the dycore's slow slots.
SLOW_FIELDS = ("thp", "u", "v", "w", "php")


def box_density(start: float, window: float) -> tuple:
    """Knots of the uniform density ``1/window`` on ``[start, start +
    window]``."""
    window = float(window)
    if not np.isfinite(window) or window <= 0:
        raise IauError(f"IAU window {window!r} s must be finite and positive")
    return ((float(start), 1.0 / window),
            (float(start) + window, 1.0 / window))


def interpolation_densities(slot_times, start: float, end: float) -> list:
    """One knot tuple per slot: the linear interpolation basis on
    ``slot_times`` (held constant beyond the end slots) over ``[start,
    end]``, divided by the window length.  The densities sum to ``1/(end -
    start)`` everywhere on the window, so their shares sum to one."""
    times = [float(t) for t in slot_times]
    start, end = float(start), float(end)
    if not times or sorted(times) != times or len(set(times)) != len(times):
        raise IauError(f"slot times {slot_times!r} must be distinct and "
                       "increasing")
    if not end > start or times[0] < start or times[-1] > end:
        raise IauError(f"slot times {times} must lie on the window "
                       f"[{start}, {end}]")
    height = 1.0 / (end - start)
    out = []
    for j, t in enumerate(times):
        knots = [(start, height)] if j == 0 else [(times[j - 1], 0.0)]
        knots.append((t, height))
        knots.append((end, height) if j == len(times) - 1
                     else (times[j + 1], 0.0))
        cleaned = []
        for knot in knots:                # a slot on a window edge
            if cleaned and knot[0] == cleaned[-1][0]:
                cleaned[-1] = (knot[0], max(cleaned[-1][1], knot[1]))
            else:
                cleaned.append(knot)
        out.append(tuple(cleaned))
    return out


def integrate_density(knots, a: float, b: float) -> float:
    """Exact integral of the piecewise-linear density over ``[a, b]``
    (zero outside its first and last knot)."""
    total = 0.0
    for (t0, f0), (t1, f1) in zip(knots[:-1], knots[1:]):
        lo, hi = max(a, t0), min(b, t1)
        if hi <= lo or t1 <= t0:
            continue
        slope = (f1 - f0) / (t1 - t0)
        total += 0.5 * ((f0 + slope * (lo - t0))
                        + (f0 + slope * (hi - t0))) * (hi - lo)
    return total


@dataclass
class IauNode:
    """One increment and its time density."""

    increments: Mapping[str, object]
    knots: tuple
    label: str = ""
    applied: float = 0.0
    #: Nodes of one group share one analysis and their shares sum to one
    #: (the slots of a 4D window); separate groups add separate parts of
    #: it (the dynamics, and the hydrometeors over their own short window).
    group: str = "analysis"

    @property
    def share(self) -> float:
        return integrate_density(self.knots, -np.inf, np.inf)

    @property
    def span(self) -> tuple:
        return float(self.knots[0][0]), float(self.knots[-1][0])


class IauStepTendencies:
    """One step's held tendencies: the physics' (``base``, may be None)
    plus the increment's rates.  Stands in for
    :class:`gpuwm.core.physics.PhysicsTendencies` for that step: the dycore
    and the scalar update read ``add_to_slow`` and ``scalar_for`` only,
    and the physics' own held arrays are never written."""

    def __init__(self, base, slow: dict, scalars: dict):
        self.base = base
        self.slow = slow
        self.scalars = scalars
        self._combined = {}

    def add_to_slow(self, state) -> None:
        if self.base is not None:
            self.base.add_to_slow(state)
        for slot, rate in self.slow.items():
            target = getattr(state, slot)
            target += rate

    def strict_stage_tendencies(self):
        """IAU rates already carry the slow-slot map scaling.

        Strict physics and scalars have been folded into held slots. Keep
        only the five dynamics rates here, including geopotential, so they
        enter each RK stage once without a second map-factor division.
        """
        return IauStepTendencies(None, self.slow, {})

    def scalar_for(self, name: str):
        own = self.scalars.get(name)
        base = (self.base.scalar_for(name) if self.base is not None
                else None)
        if own is None:
            return base
        if base is None:
            return own
        combined = self._combined.get(name)
        if combined is None:
            combined = base + own
            self._combined[name] = combined
        return combined

    def __getattr__(self, name):
        base = self.__dict__.get("base")
        if base is None:
            if name in ("ru", "rv", "rw", "rtheta"):
                return None
            raise AttributeError(name)
        return getattr(base, name)


class IauForcing:
    """The analysis increment(s) as tendencies inside the RK step.

    Attach with :func:`attach` for the integration (external data, like
    the radar heating: :func:`detach` before a restart is written); the
    dycore calls :func:`fold_step_tendencies` once per step, before its RK
    loop.
    """

    def __init__(self, state, nodes, *, provenance=None):
        if not nodes:
            raise IauError("an IAU forcing needs at least one node")
        from gpuwm.core.moist import moist_species

        transported = (set(moist_species(state))
                       if getattr(state, "qv", None) is not None else set())
        self.nodes = []
        self.fields = set()
        for node in nodes:
            device = {}
            for name, value in node.increments.items():
                target = getattr(state, name, None)
                if target is None:
                    raise IauError(
                        f"the increment names {name!r}, which this state "
                        "does not carry")
                if name not in SLOW_FIELDS and name not in transported:
                    raise IauError(
                        f"the increment names {name!r}, which is neither a "
                        "dynamics field nor a transported scalar; a forcing "
                        "through the RK step would never reach it")
                if tuple(np.shape(target)) != tuple(np.shape(value)):
                    raise IauError(
                        f"increment {name!r} is {np.shape(value)}, the "
                        f"state's field is {np.shape(target)}")
                if hasattr(target, "__cuda_array_interface__"):
                    import cupy as cp
                    device[name] = cp.asarray(value, dtype=target.dtype)
                else:
                    device[name] = np.asarray(value, dtype=target.dtype)
                self.fields.add(name)
            share = integrate_density(node.knots, -np.inf, np.inf)
            if not np.isfinite(share) or share <= 0:
                raise IauError(f"node {node.label!r} has no weight")
            self.nodes.append(IauNode(device, tuple(node.knots),
                                      node.label, group=node.group))
        groups = {}
        for node in self.nodes:
            groups[node.group] = groups.get(node.group, 0.0) + node.share
        for group, total in groups.items():
            if abs(total - 1.0) > 1e-9:
                raise IauError(
                    f"the {group!r} node shares sum to {total!r}, not one: "
                    "the forcing would add more or less than the analysis")
        self.groups = tuple(groups)
        self.provenance = dict(provenance or {})
        self.steps_forced = 0
        self.first_forced = None
        self.last_forced = None

    @property
    def span(self) -> tuple:
        starts, ends = zip(*(node.span for node in self.nodes))
        return min(starts), max(ends)

    @property
    def applied_fraction(self) -> float:
        """The least-applied group's fraction (one when all are done)."""
        done = {}
        for node in self.nodes:
            done[node.group] = done.get(node.group, 0.0) + node.applied
        return min(done.values())

    def weights(self, t0: float, dt: float) -> list:
        return [integrate_density(node.knots, t0, t0 + dt)
                for node in self.nodes]

    def fold(self, state, cfg, base):
        """``base`` (the step's physics tendencies, or None) with this
        step's increment rates beside it; ``base`` itself outside the
        window."""
        t0 = float(state.elapsed_seconds)
        dt = float(cfg.dt)
        weights = self.weights(t0, dt)
        if not any(w > 0.0 for w in weights):
            return base
        xp = np
        if hasattr(state.mup, "__cuda_array_interface__"):
            import cupy as xp  # noqa: N813
        rates = {}
        for node, weight in zip(self.nodes, weights):
            if weight <= 0.0:
                continue
            for name, delta in node.increments.items():
                term = delta * delta.dtype.type(weight / dt)
                rates[name] = term if name not in rates else rates[name] + term
            node.applied += weight
        slow, scalars = _couple(xp, state, cfg, rates)
        self.steps_forced += 1
        if self.first_forced is None:
            self.first_forced = t0
        self.last_forced = t0 + dt
        return IauStepTendencies(base, slow, scalars)

    def receipt(self) -> dict:
        return {
            "schema": IAU_FORCING_SCHEMA,
            "fields": sorted(self.fields),
            "nodes": [{"label": node.label, "group": node.group,
                       "span_seconds": list(node.span),
                       "share": node.share,
                       "applied": node.applied} for node in self.nodes],
            "applied_fraction": self.applied_fraction,
            "steps_forced": int(self.steps_forced),
            "forced_seconds": [self.first_forced, self.last_forced],
            "rule": ("per RK step, sum_j increment_j * integral(density_j "
                     "over the step) / dt, coupled like the physics "
                     "tendencies and held over the three RK stages"),
            **({"provenance": self.provenance} if self.provenance else {}),
        }


def _ring_mask(array) -> None:
    array[..., 0, :] = 0.0
    array[..., -1, :] = 0.0
    array[..., :, 0] = 0.0
    array[..., :, -1] = 0.0


def _face_mass(xp, mu, axis: int):
    """Column mass on the u (axis=1) or v (axis=0) faces of a (ny, nx)
    field: the two-cell mean inside, the adjacent cell on the edges."""
    if axis == 1:
        inner = 0.5 * (mu[:, 1:] + mu[:, :-1])
        return xp.concatenate([mu[:, :1], inner, mu[:, -1:]], axis=1)
    inner = 0.5 * (mu[1:, :] + mu[:-1, :])
    return xp.concatenate([mu[:1, :], inner, mu[-1:, :]], axis=0)


def _couple(xp, state, cfg, rates: dict):
    """Coupled slow-slot and scalar tendencies from uncoupled rates."""
    boundary = bool(getattr(cfg, "specified", False)
                    or getattr(cfg, "nested", False))
    open_x = bool(getattr(cfg, "open_x", False))
    open_y = bool(getattr(cfg, "open_y", False))
    has_msf = bool(getattr(state, "has_msf", False))
    dtype = state.mup.dtype
    mu = state.mub2d + state.mup
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    slow, scalars = {}, {}
    chm = None
    for name, rate in rates.items():
        if name in ("u", "v"):
            face_mu = _face_mass(xp, mu, 1 if name == "u" else 0)
            coupled = (c1h * face_mu[None] + c2h) * rate
            if boundary:
                _ring_mask(coupled)
            elif name == "u" and open_x:
                coupled[:, :, 0] = 0.0
                coupled[:, :, -1] = 0.0
            elif name == "v" and open_y:
                coupled[:, 0, :] = 0.0
                coupled[:, -1, :] = 0.0
            if has_msf:
                coupled = coupled / (state.msfu if name == "u"
                                     else state.msfv)[None]
            slow["ru_t" if name == "u" else "rv_t"] = coupled.astype(dtype)
        elif name in ("php", "w"):
            coupled = (state.c1f[:, None, None] * mu[None]
                       + state.c2f[:, None, None]) * rate
            if has_msf:
                coupled = coupled / state.msft[None]
            slow["rph_t" if name == "php" else "rw_t"] = coupled.astype(
                dtype)
        else:
            if chm is None:
                chm = c1h * mu[None] + c2h
            coupled = chm * rate
            if boundary:
                _ring_mask(coupled)
            if name == "thp":
                if has_msf:
                    coupled = coupled / state.msft[None]
                slow["rth_t"] = coupled.astype(dtype)
            else:
                scalars[name] = xp.ascontiguousarray(coupled.astype(dtype))
    return slow, scalars


def attach(state, forcing: IauForcing) -> None:
    """Attach ``forcing`` to ``state`` for the coming integration."""
    if not isinstance(forcing, IauForcing):
        raise IauError(f"expected an IauForcing, got "
                       f"{type(forcing).__name__}")
    if getattr(state, IAU_STATE_ATTRIBUTE, None) is not None:
        raise IauError("this state already carries an IAU forcing; detach "
                       "it first, or two analyses would share one window")
    setattr(state, IAU_STATE_ATTRIBUTE, forcing)


def detach(state):
    """Remove and return the state's forcing (``None`` if it had none)."""
    forcing = getattr(state, IAU_STATE_ATTRIBUTE, None)
    if IAU_STATE_ATTRIBUTE in vars(state):
        delattr(state, IAU_STATE_ATTRIBUTE)
    return forcing


def cap_vapour_increment(state, increments: Mapping[str, object]):
    """The increment with its vapour bounded as the one-shot applier bounds
    it (:func:`gpuwm.ensemble.increments.saturation_limit`, the one owner
    of the formula): background plus increment at most liquid saturation
    at the analysed temperature, or the background's own supersaturation
    ratio times it.  Without it the spread vapour the one-shot cap would
    have removed is condensed by the scheme over the window instead, the
    first-hour saturation burst stretched rather than removed.

    ``state`` is the background with current diagnostics.  Returns
    ``(increments, receipt)``; the mapping is the input when nothing is
    capped."""
    from gpuwm.ensemble.increments import saturation_limit

    receipt = {"schema": "gpuwm-da.iau-vapour-cap.v1", "cells": 0,
               "vapour_removed_kg_per_kg_sum": 0.0}
    if "qv" not in increments and "thp" not in increments:
        receipt["reason"] = "the increment moves neither thp nor qv"
        return increments, receipt
    xp = np
    if hasattr(state.qv, "__cuda_array_interface__"):
        import cupy as xp  # noqa: N813
    qv = state.qv.astype(xp.float64)
    dqv = (xp.asarray(increments["qv"], dtype=xp.float64)
           if "qv" in increments else xp.zeros_like(qv))
    dthp = (xp.asarray(increments["thp"], dtype=xp.float64)
            if "thp" in increments else None)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        limit = saturation_limit(xp, state.p, state.alt, state.qv, dthp)
    result = qv + dqv
    over = result > limit
    count = int(over.sum())
    receipt["cells"] = count
    if not count:
        return increments, receipt
    capped = xp.where(over, limit - qv, dqv)
    receipt["vapour_removed_kg_per_kg_sum"] = float((dqv - capped).sum())
    out = dict(increments)
    out["qv"] = capped
    return out, receipt


def fold_step_tendencies(state, cfg, physics_tendencies):
    """The dycore's one call: the attached forcing's step tendencies beside
    the physics', or ``physics_tendencies`` unchanged."""
    forcing = getattr(state, IAU_STATE_ATTRIBUTE, None)
    if forcing is None:
        return physics_tendencies
    return forcing.fold(state, cfg, physics_tendencies)
