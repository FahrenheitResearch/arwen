# tests/test_mapped_mass_closure.py  (WP-5 workstream A: mapped column-mass
# closure on a synthetic Lambert open-BC domain)
"""The mapped counterpart of the flat closure test in tests/test_openbc.py.

Everything here is measured through the production ``step()`` call on a
state whose map factors come from the production Lambert projection, so
the closure is scored against what the shipped acoustic kernel does
rather than against a re-derivation of it.

The measured residuals this module reports are committed as data.  The
absolute residual BOUND is deliberately absent: pinning it is
measure-once-then-commit under owner decision D-15, and inventing one
here would calibrate a gate to its own first measurement without the
sign-off that decision exists to provide.  What the module does assert,
bound-free, is that the closure is strictly better than each mutant --
the sensitivity claim, which is the half that can be established without
D-15.
"""
import dataclasses

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu

#: Unpinned pending D-15 (measure-once-then-commit, sign-off owner).  When
#: the decision lands, the value comes from the committed receipt of the
#: first measurement, never from re-running until a number looks tidy.
MASS_CLOSURE_RESIDUAL_BOUND = None

#: Synthetic mapped domain: a Lambert secant grid wide enough that the map
#: factor actually varies across it, so a diagnostic that ignored the
#: weighting could not pass by accident.
_MAPPED_DX_M = 50000.0


def _stratified_sounding(z):
    """Same dry stratified profile the flat closure test integrates."""
    return 300.0 + 0.004 * np.asarray(z, dtype=np.float64)


def _mapped_lambert_case(*, mapped=True):
    """Build a synthetic mapped-Lambert open-BC state and its config."""
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.static.lambert import LambertGrid

    cfg = RunConfig(nx=10, ny=8, nz=8, dx=_MAPPED_DX_M, dy=_MAPPED_DX_M,
                    ztop=8000.0, dt=0.5, run_seconds=0.5,
                    open_x=True, open_y=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, _stratified_sounding,
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    if mapped:
        grid = LambertGrid(ref_lat=40.0, ref_lon=-90.0,
                           truelat1=30.0, truelat2=60.0, stand_lon=-90.0,
                           dx=cfg.dx, dy=cfg.dy,
                           e_we=cfg.nx + 1, e_sn=cfg.ny + 1)
        state.set_map_coriolis(msft=grid.mapfac_m(), msfu=grid.mapfac_u(),
                               msfv=grid.mapfac_v())
        assert state.has_msf
    state.u[...] = cp.asarray(
        np.linspace(-2.0, 5.0, cfg.nx + 1, dtype=np.float32)[None, None, :])
    state.v[...] = cp.asarray(
        np.linspace(-1.0, 3.0, cfg.ny + 1, dtype=np.float32)[None, :, None])
    return state, cfg


def _weighted_measure(state, weight):
    """``sum(mu * weight)`` in FP64 for an arbitrary weight field."""
    import cupy as cp
    return float(cp.sum(state.total_mu().astype(cp.float64) * weight,
                        dtype=cp.float64))


# ---- baseline control: the public tree refuses the mapped call ------------

def test_public_baseline_refused_mapped_domains():
    """Pin the BEFORE behaviour the fix replaces.

    ``boundary_mass_tendency`` on public main
    (``product/v1 @ 9c9c20cf``, gpuwm/core/dycore.py) raised
    NotImplementedError for ``has_msf`` states rather than reporting a
    flat-domain number for a mapped one.  This test asserts the refusal
    text is gone from the module AND that the mapped path is now a real
    computation, so the transition is pinned by a test at both ends
    instead of being asserted in prose.
    """
    import inspect

    from gpuwm.core import dycore
    source = inspect.getsource(dycore.boundary_mass_tendency)
    assert "NotImplementedError" not in source
    mapped = inspect.getsource(dycore._boundary_mass_tendency_mapped)
    assert "msfu" in mapped and "msfv" in mapped


# ---- FUNCTIONAL + MEASURED, NOT PREDICTED --------------------------------

@requires_gpu
def test_mapped_boundary_mass_tendency_is_finite():
    from gpuwm.core.dycore import boundary_mass_tendency
    state, cfg = _mapped_lambert_case()
    value = boundary_mass_tendency(state, cfg)
    assert isinstance(value, float)
    assert np.isfinite(value)


@requires_gpu
def test_mapped_step_mass_closes_and_mutants_separate(capsys):
    """|dM - sum(increments)| / M0 through the production step().

    M = sum((mub2d + mup) / msft**2).  Both mutants are scored with the
    SAME production increments; only the measure's weighting changes, so
    the separation isolates the cell-area weighting itself.
    """
    import cupy as cp
    from gpuwm.core.dycore import domain_mass_measure, step

    state, cfg = _mapped_lambert_case()
    weight = state.cell_area_weight()
    ones = cp.ones_like(weight)
    # mutant B: one INTERIOR msft perturbed in the weighting only, so the
    # weight no longer matches the m2 the kernel formed for that cell.
    perturbed = weight.copy()
    perturbed[cfg.ny // 2, cfg.nx // 2] *= 1.01

    m0 = domain_mass_measure(state)
    m0_unweighted = _weighted_measure(state, ones)
    m0_perturbed = _weighted_measure(state, perturbed)
    increments = []
    step(state, cfg, mass_flux_observer=increments.append)
    m1 = domain_mass_measure(state)
    m1_unweighted = _weighted_measure(state, ones)
    m1_perturbed = _weighted_measure(state, perturbed)

    assert len(increments) == cfg.time_step_sound
    assert abs(sum(increments)) > 0.0
    total = sum(increments)
    residual = abs((m1 - m0) - total) / m0
    mutant_no_weight = abs((m1_unweighted - m0_unweighted) - total) / m0
    mutant_perturbed = abs((m1_perturbed - m0_perturbed) - total) / m0

    print(f"mapped mass closure: residual {residual:.6e}; "
          f"mutant(no 1/msft**2 weighting) {mutant_no_weight:.6e}; "
          f"mutant(one interior msft perturbed) {mutant_perturbed:.6e}; "
          f"bound unpinned pending D-15")

    assert np.isfinite(residual)
    # Sensitivity, asserted without predicting any magnitude.
    assert residual < mutant_no_weight
    assert residual < mutant_perturbed
    assert MASS_CLOSURE_RESIDUAL_BOUND is None, (
        "a pinned bound arrives with its D-15 sign-off and its committed "
        "first measurement, not from this test")


# ---- REDUCTION CONTROL ----------------------------------------------------

@requires_gpu
def test_mapped_helper_reduces_to_the_flat_one_bit_for_bit():
    """Identity map factors: both branches, one measure, same bits."""
    from gpuwm.core.dycore import (_boundary_mass_tendency_flat,
                                   _boundary_mass_tendency_mapped,
                                   domain_mass_measure, step)
    state, cfg = _mapped_lambert_case(mapped=False)
    assert not state.has_msf
    step(state, cfg, mass_flux_observer=lambda _inc: None)
    flat = float(_boundary_mass_tendency_flat(state, cfg))
    mapped = float(_boundary_mass_tendency_mapped(state, cfg))
    assert mapped.hex() == flat.hex()
    # and the measure itself reduces to the plain column-mass sum
    import cupy as cp
    plain = float(cp.sum(state.total_mu().astype(cp.float64),
                         dtype=cp.float64))
    assert domain_mass_measure(state).hex() == plain.hex()


# ---- EQUIVALENCE CONTROL --------------------------------------------------

@requires_gpu
def test_device_accumulator_equals_host_observer_sum_bit_for_bit():
    from gpuwm.core.dycore import MassFluxAccumulator, step

    host_state, cfg = _mapped_lambert_case()
    increments = []
    step(host_state, cfg, mass_flux_observer=increments.append)

    device_state, cfg2 = _mapped_lambert_case()
    accumulator = MassFluxAccumulator()
    step(device_state, cfg2, mass_flux_accumulator=accumulator)

    assert accumulator.count == len(increments) == cfg.time_step_sound
    host_total = 0.0
    for value in increments:
        host_total += value
    assert accumulator.total().hex() == host_total.hex()


# ---- ALLOCATION BOUND -----------------------------------------------------

#: More substep increments than any step this suite runs takes, so a per-add
#: allocation cannot hide under a small loop.
_PIN_ADDS = 2048

#: Enough full steps that a per-step or per-output-interval reallocation would
#: have happened by the end of the loop.
_PIN_STEPS = 8


@requires_gpu
def test_the_mass_flux_accumulator_allocates_once_and_never_per_substep():
    """The bound behind ``dycore.py``'s ``__init__``/``reset`` inventory rows.

    ``tests/test_physics_allocation_inventory.py`` records two device
    allocations in this module that are not drawn from any workspace, and a
    recorded row is only honest while its bound is true.  The bound claimed
    there is *one 8-byte scalar per accumulator object* -- which is a claim
    about ``add``, not about ``__init__``: ``add`` runs once per acoustic
    substep of every final RK stage, so if it allocated, the row would scale
    with the run length and belong in the ratchet as a defect rather than as
    a cost.

    It does not, because ``self._total += increment`` is cupy's in-place
    ``__iadd__``.  That is a property of one operator in one line, and the
    one-character edit that breaks it (``self._total = self._total +
    increment``) is invisible to the AST ratchet -- the site count does not
    move -- so it is measured here instead, on the device, against a malloc
    hook.

    The failing form runs first.  Without it a green result would be equally
    consistent with a hook that never fires.
    """
    import cupy as cp
    from gpuwm.core.dycore import MassFluxAccumulator, step

    class _CountingHook(cp.cuda.MemoryHook):
        """Every device malloc inside the ``with`` block, whoever asked."""

        name = "gpuwm-mass-flux-accumulator-allocation-pin"

        def __init__(self):
            super().__init__()
            self.mallocs = 0

        def malloc_postprocess(self, **kwargs):
            self.mallocs += 1

    # -- the shape the inventory row records -------------------------------
    accumulator = MassFluxAccumulator()
    assert accumulator._total.shape == ()
    assert accumulator._total.ndim == 0
    assert accumulator._total.dtype == cp.float64
    assert accumulator._total.nbytes == 8

    increment = cp.zeros((), dtype=cp.float64) + 1.5

    # -- the failing form, first -------------------------------------------
    class _OutOfPlace(MassFluxAccumulator):
        """``add`` spelled so it rebinds: one fresh scalar per substep."""

        def add(self, increment):
            self._total = self._total + increment
            self.count += 1

    leaky = _OutOfPlace()
    hook = _CountingHook()
    with hook:
        for _ in range(_PIN_ADDS):
            leaky.add(increment)
    assert hook.mallocs >= _PIN_ADDS, (
        "the malloc hook did not see the out-of-place accumulator allocate, "
        f"so this measurement cannot detect one: {hook.mallocs} mallocs over "
        f"{_PIN_ADDS} adds")

    # -- and the shipped one, device increments and host floats alike ------
    for name, value in (("device scalar", increment), ("host float", 0.25)):
        pointer = accumulator._total.data.ptr
        hook = _CountingHook()
        with hook:
            for _ in range(_PIN_ADDS):
                accumulator.add(value)
        assert hook.mallocs == 0, (
            f"MassFluxAccumulator.add allocated on the {name} path: "
            f"{hook.mallocs} device mallocs over {_PIN_ADDS} adds, so the "
            "inventory's one-scalar-per-object bound is false and the row "
            "scales with the acoustic substep count")
        assert accumulator._total.data.ptr == pointer, (
            f"the accumulator's storage moved during the {name} adds")
    assert accumulator.count == 2 * _PIN_ADDS

    # -- and across real steps, which is where a per-step site would show --
    # Object identity, not just the pointer: ``__init__`` and ``reset`` are
    # the only two statements that bind ``_total``, and the memory pool can
    # hand a freed 512-byte block straight back at the same address, so ``is``
    # is the assertion that actually excludes a rebind.
    state, cfg = _mapped_lambert_case()
    run = MassFluxAccumulator()
    storage = run._total
    pointer = run._total.data.ptr
    for _ in range(_PIN_STEPS):
        step(state, cfg, mass_flux_accumulator=run)
    assert run.count == _PIN_STEPS * cfg.time_step_sound
    assert run._total is storage, (
        f"the accumulator rebound _total during a {_PIN_STEPS}-step run, so "
        "one of the two recorded allocation sites is reachable per step "
        "rather than once per object")
    assert run._total.data.ptr == pointer
    assert run._total.nbytes == 8


@requires_gpu
def test_supplying_both_observer_keywords_raises():
    from gpuwm.core.dycore import MassFluxAccumulator, step
    state, cfg = _mapped_lambert_case()
    with pytest.raises(ValueError, match="mutually exclusive"):
        step(state, cfg, mass_flux_observer=[].append,
             mass_flux_accumulator=MassFluxAccumulator())


def test_existing_observer_callers_are_untouched():
    """The two live ``mass_flux_observer`` callers keep their call shape."""
    import inspect
    from gpuwm.core.dycore import step
    parameters = inspect.signature(step).parameters
    assert "mass_flux_observer" in parameters
    assert "mass_flux_accumulator" in parameters
    assert parameters["mass_flux_accumulator"].default is None
    assert parameters["mass_flux_observer"].default is None


# ---- token hygiene --------------------------------------------------------

def test_no_case_token_in_the_mapped_closure_surface():
    """Scoped to the surface this package adds.

    ``gpuwm/core/dycore.py`` and ``gpuwm/core/state.py`` already carry
    case tokens in comments describing a compatibility integrator
    (dycore.py lines 1312, 1983, 2011; state.py line 572); retiring those
    is not this package's edit, so the assertion covers the symbols this
    package introduced rather than pretending the files are clean.
    """
    import inspect
    import re
    from gpuwm.core import dycore
    from gpuwm.core.state import DomainState
    pattern = re.compile(r"real74|1974|ohio|hrrr", re.IGNORECASE)
    added = (dycore.boundary_mass_tendency,
             dycore.boundary_mass_tendency_device,
             dycore._boundary_mass_tendency_flat,
             dycore._boundary_mass_tendency_mapped,
             dycore.domain_mass_measure,
             dycore.MassFluxAccumulator,
             DomainState.cell_area_weight)
    for obj in added:
        text = inspect.getsource(obj)
        assert not pattern.search(text), getattr(obj, "__name__", obj)
    assert pattern.search("configs/real74_placeholder.toml")


# ---- D-30: the observer stays opt-in, and off means byte-identical --------

def _state_digest(state):
    """SHA-256 over every device array the state carries, name-ordered.

    Hashing the arrays rather than comparing a handful of named fields
    means a trajectory change anywhere in the state breaks the check,
    including in a field this test's author did not think to list.
    """
    import hashlib

    import cupy as cp

    digest = hashlib.sha256()
    for name in sorted(vars(state)):
        value = getattr(state, name)
        if isinstance(value, cp.ndarray):
            digest.update(name.encode("utf-8"))
            digest.update(cp.asnumpy(value).tobytes())
    return digest.hexdigest()


@requires_gpu
def test_the_certified_path_is_byte_identical_with_the_observer_off():
    """The default (observer off) and the observed run agree bit for bit.

    ``boundary_mass_tendency`` only READS the state, so an observed run
    must leave exactly the trajectory an unobserved one does.  This is
    the property that lets the observer be opt-in without splitting the
    certified path in two: turning it on cannot move a number.
    """
    from gpuwm.core.dycore import step

    plain_state, cfg = _mapped_lambert_case()
    step(plain_state, cfg)                       # certified path: no keyword
    plain = _state_digest(plain_state)

    observed_state, cfg2 = _mapped_lambert_case()
    increments = []
    step(observed_state, cfg2, mass_flux_observer=increments.append)
    observed = _state_digest(observed_state)

    assert len(increments) == cfg.time_step_sound
    assert observed == plain, (
        "the opt-in observer moved the trajectory it is supposed to watch")


@requires_gpu
def test_a_state_mutating_observer_breaks_the_byte_identity_check():
    """FAILURE CONTROL: the identity check above can fail.

    A byte-identity assertion that nothing can break is not evidence.
    This observer writes one ULP into the state it is handed, and the
    same digest comparison must reject it.
    """
    from gpuwm.core.dycore import step

    plain_state, cfg = _mapped_lambert_case()
    step(plain_state, cfg)
    plain = _state_digest(plain_state)

    mutating_state, cfg2 = _mapped_lambert_case()

    def _mutating_observer(increment):
        mutating_state.mup[cfg2.ny // 2, cfg2.nx // 2] *= np.float32(
            1.0 + 2.0 ** -23)

    step(mutating_state, cfg2, mass_flux_observer=_mutating_observer)
    assert _state_digest(mutating_state) != plain


@requires_gpu
def test_the_device_accumulator_is_also_trajectory_neutral():
    """The zero-host-sync form takes the same identity requirement."""
    from gpuwm.core.dycore import MassFluxAccumulator, step

    plain_state, cfg = _mapped_lambert_case()
    step(plain_state, cfg)

    accumulated_state, cfg2 = _mapped_lambert_case()
    accumulator = MassFluxAccumulator()
    step(accumulated_state, cfg2, mass_flux_accumulator=accumulator)

    assert accumulator.count == cfg.time_step_sound
    assert _state_digest(accumulated_state) == _state_digest(plain_state)
