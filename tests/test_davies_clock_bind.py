"""Davies lateral-boundary clock bind: root external dtbc consumption.

Seam-closure pins (CLAUDE-FABLE-GPUWM-DAVIES-CLOCK-DOSSIER-20260727).
WRF resets ``dtbc`` when a boundary interval is read
(share/mediation_integrate.F:1515-1522) and increments it by one model
step at solve entry, before any Davies consumer runs
(dyn_em/solve_em.F:371-372).  The first solve after a read therefore
relaxes toward ``B0 + dt*Bdot``, and the last solve of an interval
finalizes the specified ring from the OLD record at ``dtbc = T_bdy``
(solve_em.F:4531-4639) -- the new record is read only at the top of the
following step (frame/module_integrate.F:393-396).

gpuwm's root :class:`DomainClock` already implements the exact recurrence;
these pins bind its CONSUMPTION on the root's external boundaries:

1. the production tree build binds the root external mirror to the root
   clock (``bind_lateral_boundary_clock`` gains its production caller);
2. the bound final-ring overwrite honours WRF's old-record seam ownership
   instead of ``interval_at``'s half-open new-record/0 selection;
3. the N5S manual builder binds its root identically;
4. (in tests/test_restart.py) restart headers carry the
   root-external-LBC-clock semantic identity so pre-bind elapsed-based
   checkpoints fail closed.

The historical F20 adjudication (root deliberately UNBOUND to protect the
frozen Phase-4 anchor bytes) is retired by the same change; the N-series
ratchets regenerate against a new anchor epoch in the same batch.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The only modules that may reference the binder: its definition, the
#: production tree build, and the N5S restored-model builder.  Sweeping the
#: whole package keeps the F20-era hardening (no aliased/getattr caller can
#: slip in an unadjudicated bind site) while inverting the old negative pin
#: into the positive production contract.
SANCTIONED_BINDER_MODULES = (
    Path("ingest") / "lateral_bc.py",
    Path("core") / "model.py",
    Path("verify") / "cases" / "real74_n5s.py",
)


def _binder_references(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        hit = (
            (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == "bind_lateral_boundary_clock")
            or (isinstance(node, ast.Name)
                and node.id == "bind_lateral_boundary_clock")
            or (isinstance(node, ast.Attribute)
                and node.attr == "bind_lateral_boundary_clock")
            or (isinstance(node, ast.Constant)
                and node.value == "bind_lateral_boundary_clock")
            or (isinstance(node, ast.ImportFrom)
                and any(alias.name == "bind_lateral_boundary_clock"
                        for alias in node.names)))
        if hit:
            lines.append(node.lineno)
    return lines


def _function_calls_binder(path: Path, function_name: str) -> bool:
    """True iff ``function_name`` in ``path`` calls the binder by name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == function_name):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    name = (func.id if isinstance(func, ast.Name) else
                            func.attr if isinstance(func, ast.Attribute)
                            else None)
                    if name == "bind_lateral_boundary_clock":
                        return True
    return False


def test_production_tree_build_binds_root_external_boundary_clock():
    """Piece 1: ``build_experiment`` calls the binder (dossier section 5.1).

    The insertion point is immediately after the root ``DomainNode``
    construction and before the ``nodes`` mapping, so the bind exists
    before any solve or restart restoration.  This is the positive
    replacement for the retired F20 negative pin: production MUST bind
    the root's external Davies clock so every root boundary launch
    consumes WRF's post-increment ``dtbc_launch_fp32``
    (dyn_em/solve_em.F:371-372) instead of the one-step-lagged
    elapsed-based value.
    """
    model_py = REPO_ROOT / "gpuwm" / "core" / "model.py"
    assert _function_calls_binder(model_py, "build_experiment"), (
        "gpuwm.core.model.build_experiment does not call "
        "bind_lateral_boundary_clock: the root's external Davies "
        "relaxation target stays one d01 step behind WRF "
        "(dtbc=0..T-dt instead of dt..T)")


def test_n5s_restored_builder_binds_root_external_boundary_clock():
    """Piece 3: the N5S manual builder binds its root (dossier 5.3).

    ``build_restored_model`` constructs DomainNodes itself instead of
    calling ``build_experiment``; without its own binder call the
    regenerated matched-physics shadow would keep measuring the retired
    root-clock deviation after production was corrected.
    """
    n5s_py = REPO_ROOT / "gpuwm" / "verify" / "cases" / "real74_n5s.py"
    assert _function_calls_binder(n5s_py, "build_restored_model"), (
        "real74_n5s.build_restored_model does not bind the restored "
        "root's external boundary clock")


def test_binder_references_are_exactly_the_sanctioned_callers():
    """Package sweep: definition + the two adjudicated callers, nothing
    else.  Retains the F20 review hardening (names, attributes, string
    literals, import aliases) as a positive inventory."""
    package_root = REPO_ROOT / "gpuwm"
    expected = {package_root / rel for rel in SANCTIONED_BINDER_MODULES}
    referencing = set()
    for path in sorted(package_root.rglob("*.py")):
        if _binder_references(path):
            referencing.add(path)
    missing = sorted(str(p.relative_to(package_root))
                     for p in expected - referencing)
    extra = sorted(str(p.relative_to(package_root))
                   for p in referencing - expected)
    assert not missing and not extra, (
        f"bind_lateral_boundary_clock callers drifted from the "
        f"adjudicated inventory: missing {missing}, extra {extra}")


# ---------------------------------------------------------------------------
# Functional pins (GPU): the bound consumption semantics themselves.
# ---------------------------------------------------------------------------


def _root_clock(*, dt_s: int, lbc_interval_s: int, run_s: int):
    """A real root DomainClock on a 1-tick-per-second lattice."""
    from gpuwm.core.clock import DomainClock, DomainTicks

    spec = DomainTicks(
        grid_id=1, parent_id=0, parent_time_step_ratio=1,
        step_ticks=int(dt_s), dt_fp32=np.float32(dt_s),
        history_ticks=int(run_s), restart_ticks=None,
        radt_ticks=None, stepra=None, cudt_ticks=None, stepcu=None,
        bldt_ticks=None, stepbl=None,
        lbc_interval_ticks=int(lbc_interval_s))
    return DomainClock(spec, 1, int(run_s))


@requires_gpu
@pytest.mark.gpu
def test_bound_root_first_post_seam_launch_consumes_postincrement_dt():
    """First post-reset launch: bound roots consume ``dtbc = dt``.

    WRF's first solve after every boundary read uses ``dtbc = dt``
    (reset at mediation_integrate.F:1522, increment at solve_em.F:371-372
    -- dtbc=0 never reaches a Davies consumer on a commensurate clock).
    The bound branch of ``_active_device_interval`` must hand kernels the
    post-increment FP32 recurrence; the unbound legacy branch is the
    retired one-step-lagged compatibility fallback (dtbc=0 at the same
    instant), kept only for direct paths without a DomainClock.
    """
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.ingest.lateral_bc import (_active_device_interval,
                                         attach_lateral_boundaries,
                                         bind_lateral_boundary_clock,
                                         build_state_lateral_boundaries)

    cfg = RunConfig(nx=14, ny=12, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=60.0, run_seconds=120.0,
                    specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    snapshots = [init_at_rest(cfg, coord, base) for _ in range(3)]
    boundaries = build_state_lateral_boundaries(
        snapshots, [0.0, 120.0, 240.0])

    bound = init_at_rest(cfg, coord, base)
    attach_lateral_boundaries(bound, boundaries)
    clock = _root_clock(dt_s=60, lbc_interval_s=120, run_s=240)
    bind_lateral_boundary_clock(bound, clock)
    # Executor protocol at the t=0 seam step: lbc reset, then the pre-solve
    # increment (clock.py execute_schedule STEP op).
    assert clock.lbc_reset_due()
    clock.mark_force()
    clock.prepare_step()
    _, dtbc, _, _ = _active_device_interval(bound, cfg)
    assert np.float32(dtbc).view(np.uint32) == np.float32(60.0).view(
        np.uint32), (
        f"bound first post-seam launch consumed dtbc={float(dtbc)!r}, "
        "expected WRF's post-increment dt=60")

    unbound = init_at_rest(cfg, coord, base)
    attach_lateral_boundaries(unbound, boundaries)
    _, legacy_dtbc, _, _ = _active_device_interval(unbound, cfg)
    assert float(legacy_dtbc) == 0.0, (
        "the unbound compatibility fallback is expected to keep the "
        "legacy solve-entry value (one step behind WRF)")


@requires_gpu
@pytest.mark.gpu
def test_bound_final_ring_overwrite_owns_old_record_at_interval_seam():
    """Bound final overwrite at an interior seam: OLD interval at T_bdy.

    WRF's last solve of a boundary interval runs ``spec_bdy_final`` with
    the old record still loaded and the step-constant post-increment
    ``dtbc = T_bdy`` (solve_em.F:371-372, :4531-4639); the new record is
    read at the top of the NEXT step.  gpuwm's half-open ``interval_at``
    on the end-of-step time instead selects the NEW interval at dtbc=0 --
    equal-valued for continuous tables but not FP32-bit-identical, since
    the old endpoint is reconstructed as value + T*tendency.

    Control construction: a second state runs the identical two steps
    against tables holding ONLY the first interval, where the final-
    endpoint ownership rule (lateral_bc.interval_at, last endpoint
    belongs to the last interval) already yields old-record-at-T
    semantics regardless of the bind.  Both states are bound to
    identical clocks and identical first-interval tables, so every
    launch before the final ring overwrite is bit-identical; the ONLY
    free difference is the seam-step record selection.  The pin
    requires full bitwise state equality -- WRF's ownership -- and at
    the unfixed base it fails with the candidate's ring carrying the
    new record's exact values instead of the old record's FP32
    reconstruction.
    """
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         bind_lateral_boundary_clock,
                                         build_state_lateral_boundaries)

    cfg = RunConfig(nx=14, ny=12, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=60.0, run_seconds=120.0,
                    specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)

    def snapshot(scale: float):
        s = init_at_rest(cfg, coord, base)
        # Spatially varying non-dyadic deltas: the FP32 tendency
        # reconstruction value + 120*((B1-B0)/120) must NOT be bit-equal
        # to the directly stored endpoint B1 somewhere on the ring.  A
        # uniform delta over the flat-terrain rest state collapses every
        # ring element to one bit pattern (a single rounding coin flip);
        # a ramp of hundreds of distinct magnitudes makes coincidental
        # equality of the two seam ownerships vanishingly unlikely, and
        # the RED run at the pre-fix base is the witness that it does
        # differ.
        def ramp(shape):
            cells = int(np.prod(shape))
            pattern = (cp.arange(cells, dtype=cp.float32) % cp.float32(97)
                       ) * cp.float32(1.0e-3)
            return pattern.reshape(shape)

        s.thp[...] = s.thp + cp.float32(scale) * (
            cp.float32(0.7000123) + ramp(s.thp.shape))
        s.u[...] = s.u + cp.float32(scale) * (
            cp.float32(0.3500061) + ramp(s.u.shape))
        return s

    snap0, snap1, snap2 = snapshot(0.0), snapshot(1.0), snapshot(2.0)
    two_intervals = build_state_lateral_boundaries(
        [snap0, snap1, snap2], [0.0, 120.0, 240.0])
    first_interval_only = build_state_lateral_boundaries(
        [snap0, snap1], [0.0, 120.0])

    def run_two_steps(boundaries):
        state = init_at_rest(cfg, coord, base)
        attach_lateral_boundaries(state, boundaries)
        clock = _root_clock(dt_s=60, lbc_interval_s=120, run_s=240)
        bind_lateral_boundary_clock(state, clock)
        for _ in range(2):
            if clock.lbc_reset_due():
                clock.mark_force()
            clock.prepare_step()
            step(state, cfg)
            clock.advance()
        assert clock.dtbc_launch_fp32.view(np.uint32) == np.float32(
            120.0).view(np.uint32)
        return state

    candidate = run_two_steps(two_intervals)
    control = run_two_steps(first_interval_only)

    mismatches = {}
    for name in ("u", "v", "w", "thp", "php", "mup"):
        cand = cp.asnumpy(getattr(candidate, name))
        ctrl = cp.asnumpy(getattr(control, name))
        if not np.array_equal(cand.view(np.uint32), ctrl.view(np.uint32)):
            delta = np.abs(cand.astype(np.float64)
                           - ctrl.astype(np.float64))
            mismatches[name] = (int((cand.view(np.uint32)
                                     != ctrl.view(np.uint32)).sum()),
                                float(delta.max()))
    assert not mismatches, (
        "the seam-step final ring overwrite does not own the OLD record "
        "at dtbc=T_bdy (WRF solve_em.F:4531-4639 with the pre-read "
        "record): candidate (two intervals, bound) differs from the "
        "old-record control by {field: (bad_words, max_abs)} = "
        f"{mismatches}")
