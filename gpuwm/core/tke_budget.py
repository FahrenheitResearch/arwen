"""Term-by-term prognostic-TKE budget, accumulated on device per step.

WP-L7 window preparation for the LES program (AC-L5.1): the volume-
integrated ``de/dt`` compared against the terms that produced it, with the
residual COMMITTED as a receipt rather than scored against a bound cut from
itself.  Nothing here gates; nothing here is a band.

What is measured
----------------
Every term is a COUPLED tendency -- the same ``(c1*mu + c2) * e`` currency
the RK scalar update consumes -- summed over each horizontal slab in FP64 by
one device launch per model step.  The terms partition the completed step
exactly:

``shear``, ``buoyancy``, ``dissipation``, ``limiter``
    the four pieces ``wrf_tke_rhs`` deposits (module_diffusion_em.F
    tke_rhs :6099-6229); ``limiter`` is what the positivity floor at :6221
    added on top of the physical source, exported instead of folded into
    ``dissipation`` so a floor-dominated regime is legible as one.
``diffusion_h``, ``diffusion_v``
    TKE self-diffusion with 2*Km, horizontal and vertical
    (:3988-3996 / :4896-4904).
``diffusion_6th``
    the sixth-order filter row, present only under ``diff_6th_opt > 0``
    with ``tke_mix6_off`` false.
``transport``
    the final RK stage's advective flux divergence (WRF ``rk_scalar_tend``
    for tke, solve_em.F:2362-2399).
``storage``
    ``(chm*e_new - chm0*e_old)/dt`` for the completed step.
``clip``
    what ``bound_tke`` (module_em.F:2490-2520) and the PD zero clamp moved
    at the final update.  Non-conservative by construction and therefore
    never absorbed into the residual.

``residual = storage - sum(all other terms)`` is then a pure discretization
/ FP32-rounding measure of the port, which is exactly the number AC-L5.1
asks to be committed.  The accompanying sensitivity claim -- dropping
dissipation from the trajectory moves the budget by orders of magnitude --
is the bound-free control, and it lives in tests/test_tke_km2.py.

Ownership
---------
The accumulator lives in ``DomainState._scratch`` under the ``tke_budget_``
slot family (restart class REBUILT: a report-only per-window accumulator the
caller drains, never trajectory state).  No new ``DomainState`` attribute,
no new restart-serialized field.  ``cfg.tke_budget`` is a restart-boundary-
adjustable diagnostic toggle exactly like ``nwp_diagnostics``.
"""
from __future__ import annotations

import numpy as np

#: Terms written as full 3-D fields during the step, in packed-buffer order.
TERM_FIELDS = (
    "shear", "buoyancy", "dissipation", "limiter",
    "diffusion_h", "diffusion_v", "diffusion_6th", "transport",
)
#: Terms the reduction kernel derives from the state itself.
DERIVED_TERMS = ("storage", "clip")
#: Every accumulated row, in accumulator order.
TERMS = TERM_FIELDS + DERIVED_TERMS

#: Terms whose sum must reproduce ``storage``.
SOURCE_TERMS = TERM_FIELDS + ("clip",)

#: Slot names are written as LITERALS at every ``state.scratch`` call
#: below, deliberately: tests/test_preflight.py's AST completeness gate
#: resolves literal slots against the static registry and would have to
#: allowlist this module wholesale if the names arrived through constants.

_TPB = 256


def enabled(cfg) -> bool:
    """Whether this configuration accumulates the budget at all."""
    return bool(getattr(cfg, "tke_budget", 0)) and cfg.km_opt == 2


def term_buffer(state, cfg):
    """The packed ``(len(TERM_FIELDS), nz, ny, nx)`` term buffer, or None."""
    if not enabled(cfg):
        return None
    nz, ny, nx = state.p.shape
    return state.scratch((len(TERM_FIELDS), nz, ny, nx), "tke_budget_terms")


def term(state, cfg, name: str):
    """One term's 3-D view, or None when the budget is off.

    Callers write into the returned view with ``[...] =`` (a fresh value
    each step); an unknown name is a programming error, not a silent no-op.
    """
    if name not in TERM_FIELDS:
        raise KeyError(
            f"{name!r} is not a TKE budget field term; known terms are "
            f"{TERM_FIELDS}")
    buf = term_buffer(state, cfg)
    if buf is None:
        return None
    return buf[TERM_FIELDS.index(name)]


def raw_carrier(state, cfg):
    """Scratch holding the final update BEFORE bound_tke, or None."""
    if not enabled(cfg):
        return None
    return state.scratch(state.p.shape, "tke_budget_raw")


def accumulator(state, cfg):
    """The ``(len(TERMS), nz)`` FP64 slab accumulator, or None."""
    if not enabled(cfg):
        return None
    nz = state.p.shape[0]
    return state.scratch((len(TERMS), nz), "tke_budget_acc", dtype=np.float64)


def reset(state, cfg) -> None:
    """Zero the accumulator and the step counter (start a new window)."""
    acc = accumulator(state, cfg)
    if acc is None:
        return
    acc[...] = 0.0
    state.scratch((1,), "tke_budget_steps", dtype=np.float64)[...] = 0.0


def clear_fields(state, cfg) -> None:
    """Zero the packed term buffer at the top of a step.

    Terms that are not produced by this configuration (``diffusion_6th``
    without the filter) must read as an honest zero, not as last step's
    value.
    """
    buf = term_buffer(state, cfg)
    if buf is not None:
        buf[...] = 0


def accumulate(state, cfg) -> None:
    """Fold the completed step into the accumulator (one device launch)."""
    if not enabled(cfg):
        return
    from gpuwm.core.kernels import get_kernel

    nz, ny, nx = state.p.shape
    terms = term_buffer(state, cfg)
    raw = raw_carrier(state, cfg)
    acc = accumulator(state, cfg)
    # Fresh contiguous (ny, nx) totals, exactly the chm0/chm the RK scalar
    # update formed (mub2d broadcasts for flat terrain).
    mu0 = state.scratch((ny, nx), "tke_budget_mu0")
    mu = state.scratch((ny, nx), "tke_budget_mu")
    mu0[...] = state.mub2d + state.mup0
    mu[...] = state.mub2d + state.mup
    grid = (nz, len(TERMS), 1)
    get_kernel("tke_budget", "tke_budget_accumulate")(
        grid, (_TPB, 1, 1),
        (terms, state.tke0, state.tke, raw, mu0, mu,
         state.c1h, state.c2h,
         np.float32(1.0 / cfg.dt), acc,
         np.int32(len(TERM_FIELDS)),
         np.int32(nz), np.int32(ny), np.int32(nx)),
        shared_mem=_TPB * 8)
    state.scratch((1,), "tke_budget_steps", dtype=np.float64)[...] += 1.0


def drain(state, cfg, *, reset_window: bool = True) -> dict | None:
    """Read the window's mean per-step budget to the host, then reset.

    Returns per-level profiles (``(nz,)`` each, coupled units divided by the
    accumulated step count) plus their volume sums and the closure residual.
    ``None`` when the budget is off or no step has been accumulated yet.
    """
    if not enabled(cfg):
        return None
    acc = accumulator(state, cfg)
    steps_buf = state.scratch((1,), "tke_budget_steps", dtype=np.float64)
    steps = float(_host(steps_buf)[0])
    if steps <= 0.0:
        return None
    host = _host(acc) / steps
    profiles = {name: host[index].copy()
                for index, name in enumerate(TERMS)}
    volume = {name: float(values.sum())
              for name, values in profiles.items()}
    source_profile = sum(profiles[name] for name in SOURCE_TERMS)
    residual_profile = profiles["storage"] - source_profile
    residual = float(residual_profile.sum())
    scale = max(
        max(abs(volume[name]) for name in SOURCE_TERMS),
        abs(volume["storage"]),
    )
    out = {
        "steps": steps,
        "terms": TERMS,
        "profiles": profiles,
        "volume": volume,
        "residual": residual,
        "residual_profile": residual_profile,
        "residual_rel": (residual / scale) if scale > 0.0 else 0.0,
    }
    if reset_window:
        reset(state, cfg)
    return out


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


__all__ = [
    "DERIVED_TERMS", "SOURCE_TERMS", "TERMS", "TERM_FIELDS",
    "accumulate", "accumulator", "clear_fields", "drain", "enabled",
    "raw_carrier", "reset", "term", "term_buffer",
]
