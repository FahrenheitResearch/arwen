"""The inflow seeding generator, proved on fields whose answer is known.

The generator's whole contract is determinism plus a byte-untouched OFF
path, so the tests pin exactly that: the same key reproduces the same
draw bit for bit, every key ingredient changes it, the block expansion
is piecewise-constant at the pinned width, the coupled-units table write
touches only the registered relax rows, and a default configuration
builds no generator object at all.  The full-scale OFF/zero-amplitude
byte gates run against the retained dual-certified pair
(INFLOW-GENERATOR-ACCEPTANCE-V2, G1/G2) and are receipts, not tests.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core import constants as c
from gpuwm.core import inflow_perturbation as ip


# ---------------------------------------------------------------------------
# RNG keying
# ---------------------------------------------------------------------------

def test_same_key_same_bits():
    a = ip.draw_unit_pattern(7, 3, "west", 41, 12, 6)
    b = ip.draw_unit_pattern(7, 3, "west", 41, 12, 6)
    assert a.shape == (12, 6)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("mutate", [
    {"seed": 8}, {"grid_id": 2}, {"face": "east"}, {"refresh": 42},
])
def test_every_key_ingredient_changes_the_draw(mutate):
    base = dict(seed=7, grid_id=3, face="west", refresh=41)
    a = ip.draw_unit_pattern(
        base["seed"], base["grid_id"], base["face"], base["refresh"], 12, 6)
    base.update(mutate)
    b = ip.draw_unit_pattern(
        base["seed"], base["grid_id"], base["face"], base["refresh"], 12, 6)
    assert not np.array_equal(a, b)


def test_draws_are_bounded_and_centered():
    draw = ip.draw_unit_pattern(0, 3, "south", 0, 49, 51)
    assert draw.min() >= -1.0 and draw.max() < 1.0
    assert abs(draw.mean()) < 0.1


def test_negative_key_refused():
    with pytest.raises(ValueError):
        ip.draw_unit_pattern(-1, 3, "west", 0, 4, 4)
    with pytest.raises(ValueError):
        ip.draw_unit_pattern(0, 3, "west", -1, 4, 4)


# ---------------------------------------------------------------------------
# Refresh index
# ---------------------------------------------------------------------------

def test_refresh_holds_for_pinned_seconds():
    # The measured case: parent dt 3.75 s, step 15 ticks (tick = 0.25 s).
    # 100 s / 3.75 s rounds to 27 forces = 405 ticks per refresh.
    step, dt = 15, 3.75
    indices = [ip.refresh_index(t, step, dt)
               for t in range(0, 3 * 27 * step, step)]
    assert indices[0] == 0
    assert indices.count(0) == 27
    assert indices.count(1) == 27
    assert max(indices) == 2


def test_refresh_never_divides_by_zero_forces():
    # A parent dt longer than the hold still redraws every force.
    assert ip.refresh_index(0, 100, 300.0) == 0
    assert ip.refresh_index(100, 100, 300.0) == 1


# ---------------------------------------------------------------------------
# Block expansion and vertical extent
# ---------------------------------------------------------------------------

def test_blocks_are_piecewise_constant_at_pinned_width():
    pattern = ip.draw_unit_pattern(1, 3, "north", 5, 2, 7)  # 7 blocks
    expanded = ip.expand_blocks(pattern, 50)                # 50 = 6*8 + 2
    assert expanded.shape == (2, 50)
    for b in range(6):
        block = expanded[:, b * 8:(b + 1) * 8]
        assert np.all(block == block[:, :1])
        assert np.all(block[:, 0] == pattern[:, b])
    assert np.all(expanded[:, 48:] == pattern[:, 6:7])


def test_block_count_mismatch_refused():
    with pytest.raises(ValueError):
        ip.expand_blocks(np.zeros((2, 6)), 50)


def test_perturbed_levels_contiguous_from_surface():
    z = np.array([50.0, 150.0, 400.0, 900.0, 1600.0, 2600.0])
    assert ip.perturbed_level_count(z, 1000.0) == 4
    assert ip.perturbed_level_count(z, 10.0) == 0
    assert ip.perturbed_level_count(z, 0.0) == 0
    assert ip.perturbed_level_count(z, float("nan")) == 0
    assert ip.perturbed_level_count(z, 1.0e9) == len(z)


# ---------------------------------------------------------------------------
# Face selection and amplitude
# ---------------------------------------------------------------------------

def test_flow_dep_bdy_selection_and_mutation_mode():
    means = {"west": 4.0, "east": -2.0, "south": 0.0, "north": 1.5}
    assert ip.select_faces(means, "inflow") == ["west", "north"]
    assert ip.select_faces(means, "outflow") == ["east"]
    with pytest.raises(ValueError):
        ip.select_faces(means, "everywhere")


def test_eckert_amplitude_convention():
    # theta_max = scale * U^2 / (Ec * cp), sign-free in U.
    expected = 25.0 / (ip.ECKERT * c.CP)
    assert ip.face_amplitude(5.0, 1.0) == pytest.approx(expected)
    assert ip.face_amplitude(-5.0, 1.0) == pytest.approx(expected)
    assert ip.face_amplitude(5.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# The coupled-units device write
# ---------------------------------------------------------------------------

class _TableState:
    """Just the fields add_coupled_theta reads."""

    def __init__(self, cp, nz, ny, nx):
        rng = np.random.default_rng(0)
        self.mub2d = cp.asarray(
            rng.uniform(30000.0, 60000.0, (ny, nx)).astype(np.float32))
        self.c1h = cp.asarray(
            np.linspace(1.0, 0.2, nz).astype(np.float32))
        self.c2h = cp.asarray(
            np.linspace(0.0, 40000.0, nz).astype(np.float32))


def _tables(cp, nz, ny, nx, width):
    rng = np.random.default_rng(1)

    def pair(shape):
        return (cp.asarray(rng.normal(size=shape).astype(np.float32)),
                cp.asarray(rng.normal(size=shape).astype(np.float32)))

    theta = {"west": pair((nz, ny, width)), "east": pair((nz, ny, width)),
             "south": pair((nz, width, nx)), "north": pair((nz, width, nx))}
    mu = {"west": pair((1, ny, width)), "east": pair((1, ny, width)),
          "south": pair((1, width, nx)), "north": pair((1, width, nx))}
    return {"theta": theta, "mu": mu}


@pytest.mark.parametrize("face", ["west", "east", "south", "north"])
def test_coupled_write_touches_only_relax_rows(face):
    cp = pytest.importorskip("cupy")
    nz, ny, nx, width, spec_zone, relax_zone = 5, 24, 20, 5, 1, 4
    state = _TableState(cp, nz, ny, nx)
    fields = _tables(cp, nz, ny, nx, width)
    before = {name: {side: (pair[0].copy(), pair[1].copy())
                     for side, pair in sides.items()}
              for name, sides in fields.items()}
    length = ny if face in ("west", "east") else nx
    delta = np.zeros((nz, length), dtype=np.float32)
    delta[:3] = 0.25
    ip.add_coupled_theta(fields, face, delta, state,
                         spec_zone=spec_zone, relax_zone=relax_zone)

    for name, sides in fields.items():
        for side, (value, tendency) in sides.items():
            old_value, old_tendency = before[name][side]
            # Tendencies never move; other fields and faces never move.
            assert cp.array_equal(tendency, old_tendency)
            if name != "theta" or side != face:
                assert cp.array_equal(value, old_value)

    value = fields["theta"][face][0]
    old = before["theta"][face][0]
    changed = value != old
    axis = 2 if face in ("west", "east") else 1
    # The spec-zone row and the beyond-relax rows are byte-untouched.
    for row in (*range(spec_zone), *range(relax_zone, width)):
        assert not bool(changed.take(row, axis=axis).any())
    # Every relax row moved by ch * delta, in float32.
    mu_value = fields["mu"][face][0][0]
    if face in ("west", "east"):
        columns = state.mub2d[:, :width] if face == "west" \
            else state.mub2d[:, ::-1][:, :width]
        ch = (state.c1h[:, None, None] * (columns + mu_value)[None]
              + state.c2h[:, None, None])
        expected = old + ch * cp.asarray(delta)[:, :, None]
        rows = (slice(None), slice(None), slice(spec_zone, relax_zone))
    else:
        columns = state.mub2d[:width, :] if face == "south" \
            else state.mub2d[::-1, :][:width, :]
        ch = (state.c1h[:, None, None] * (columns + mu_value)[None]
              + state.c2h[:, None, None])
        expected = old + ch * cp.asarray(delta)[:, None, :]
        rows = (slice(None), slice(spec_zone, relax_zone), slice(None))
    assert cp.array_equal(value[rows], expected[rows])
    # Levels the caller zeroed are untouched even inside relax rows.
    zeroed = value[(slice(3, None), *rows[1:])]
    assert cp.array_equal(zeroed, old[(slice(3, None), *rows[1:])])


# ---------------------------------------------------------------------------
# Configuration schema
# ---------------------------------------------------------------------------

def _nested_cfg(**overrides):
    keys = dict(
        nx=24, ny=24, nz=8, dx=250.0, dy=250.0, ztop=4000.0, dt=1.0,
        run_seconds=60.0, nested=True, km_opt=3, isfflx=1,
        sf_sfclay_physics=91, sf_surface_physics=2, bl_pbl_physics=0,
        moist=True, mp_physics=6, hybrid_opt=2, hypsometric_opt=2,
        h_sca_adv_order=5, spec_exp=0.0)
    keys.update(overrides)
    return RunConfig(**keys)


def test_default_config_is_off():
    cfg = _nested_cfg()
    assert cfg.inflow_perturbation is False
    assert cfg.inflow_perturbation_seed == 0
    assert cfg.inflow_perturbation_amplitude_scale == 1.0
    assert cfg.inflow_perturbation_faces == "inflow"
    validate_run_config(cfg)


def test_on_requires_a_nest_child():
    cfg = _nested_cfg(nested=False, specified=True,
                      inflow_perturbation=True)
    with pytest.raises(ValueError, match="nest-boundary mechanism"):
        validate_run_config(cfg)
    validate_run_config(_nested_cfg(inflow_perturbation=True))


def test_companion_keys_fail_loud_even_when_off():
    with pytest.raises(ValueError, match="inflow_perturbation_faces"):
        validate_run_config(_nested_cfg(inflow_perturbation_faces="both"))
    with pytest.raises(ValueError, match="amplitude_scale"):
        validate_run_config(
            _nested_cfg(inflow_perturbation_amplitude_scale=-1.0))
    with pytest.raises(ValueError, match="seed"):
        validate_run_config(_nested_cfg(inflow_perturbation_seed=-3))


def test_zero_scale_is_schema_legal():
    validate_run_config(_nested_cfg(
        inflow_perturbation=True,
        inflow_perturbation_amplitude_scale=0.0))


def test_outflow_mutation_mode_is_schema_legal():
    validate_run_config(_nested_cfg(
        inflow_perturbation=True, inflow_perturbation_faces="outflow"))


# ---------------------------------------------------------------------------
# The OFF contract at the coupler seam
# ---------------------------------------------------------------------------

class _CfgNode:
    """The two attributes build_inflow_perturbation reads."""

    class _Cfg:
        def __init__(self, run, grid_id, parent_id):
            self.run = run
            self.grid_id = grid_id
            self.parent_id = parent_id
            self.i_parent_start = 3
            self.j_parent_start = 3
            self.parent_grid_ratio = 3

    def __init__(self, run, parent=None, grid_id=3, parent_id=2):
        self.cfg = self._Cfg(run, grid_id, parent_id)
        self.parent = parent


def test_off_builds_no_generator_object():
    child = _CfgNode(_nested_cfg())
    assert ip.build_inflow_perturbation(child) is None


def test_on_refuses_a_pbl_off_parent():
    parent = _CfgNode(_nested_cfg(bl_pbl_physics=0), grid_id=2, parent_id=1)
    child = _CfgNode(_nested_cfg(inflow_perturbation=True), parent=parent)
    with pytest.raises(ValueError, match="parent-diagnosed PBLH"):
        ip.build_inflow_perturbation(child)


def test_on_builds_against_a_pbl_parent():
    parent_run = _nested_cfg(bl_pbl_physics=1, km_opt=4)
    parent = _CfgNode(parent_run, grid_id=2, parent_id=1)
    child = _CfgNode(_nested_cfg(inflow_perturbation=True), parent=parent)
    built = ip.build_inflow_perturbation(child)
    assert isinstance(built, ip.InflowPerturbation)
    assert built.faces_mode == "inflow"
    assert built.amplitude_scale == 1.0


# ---------------------------------------------------------------------------
# The refresh cadence under an adaptive parent (399d95a86)
#
# refresh_index is not a per-step physical rate like dtbc or the Davies
# weight -- those are the readings 399d95a86 correctly moved from the
# CONFIGURED step to the LIVE one.  It is a QUANTIZER of absolute model
# time into REFRESH_SECONDS buckets, and a quantizer whose bucket moves
# is not a function of time: floor(T / D(t)) with D varying is neither
# monotone nor onto.  These tests state the cadence contract the module
# docstring already claims ("held for 100 s of model time ... the same
# draw on every card, every run, every restart") against a parent whose
# step breathes, which is the one case the pinned fixed-clock test
# (test_refresh_holds_for_pinned_seconds, a run-CONSTANT pair) cannot
# see.
# ---------------------------------------------------------------------------

#: A parent on the adaptive tick lattice: dt breathes 4-8 s around a
#: CONFIGURED 5.0 s, tick = 1/100 s (399d95a86 folds 100 into the
#: denominator when the adaptive clock is on).
_TICK_DEN = 100
_SPEC_DT = 5.0
_SPEC_STEP = int(_SPEC_DT * _TICK_DEN)
#: round(REFRESH_SECONDS / _SPEC_DT) forces of _SPEC_STEP ticks each.
_SPEC_BUCKET_TICKS = 20 * _SPEC_STEP


class _Spec:
    """The FROZEN configured pair (core/clock.py:322-334, :336-344)."""

    def __init__(self, step_ticks, dt_fp32):
        self.step_ticks = int(step_ticks)
        self.dt_fp32 = float(dt_fp32)


class _Clock:
    """The tick counter plus both step pairs, live and configured.

    ``spec.*`` is config-resolution output and stays frozen; the bare
    attributes are what the model is integrating with RIGHT NOW, and the
    only writers of them anywhere are core/adaptive_clock.py:780,785 and
    io/restart.py:4643-4644, both adaptive-only.  On a fixed clock the
    two agree for the whole run, which is why every shipped case is
    blind to the difference.
    """

    def __init__(self, step_ticks, dt_fp32):
        self.spec = _Spec(step_ticks, dt_fp32)
        self.ticks = 0
        self.step_ticks = int(step_ticks)
        self.dt_fp32 = float(dt_fp32)

    def at(self, ticks, step_ticks=None, dt_fp32=None):
        self.ticks = int(ticks)
        if step_ticks is not None:
            self.step_ticks = int(step_ticks)
        if dt_fp32 is not None:
            self.dt_fp32 = float(dt_fp32)
        return self


class _ClockNode:
    """A parent as the force hook sees it: one clock, nothing else."""

    def __init__(self, clock):
        self.clock = clock


def _adaptive_trajectory(hours=3.0):
    """(ticks, step_ticks, dt) at every parent force of a breathing run.

    The controller quantises dt onto the tick lattice and ``advance()``
    increments ``ticks`` by the LIVE step, so the tick stamps are not a
    uniform ladder -- which is the whole point.
    """
    rows = []
    ticks = 0
    limit = int(hours * 3600 * _TICK_DEN)
    while ticks < limit:
        dt = 6.0 + 2.0 * math.sin(
            2.0 * math.pi * (ticks / _TICK_DEN) / 1200.0)
        step = max(1, int(round(dt * _TICK_DEN)))
        rows.append((ticks, step, step / _TICK_DEN))
        ticks += step
    return rows


def _on_generator():
    """One ON generator against a legal (PBL-on) parent."""
    parent = _CfgNode(_nested_cfg(bl_pbl_physics=1, km_opt=4),
                      grid_id=2, parent_id=1)
    child = _CfgNode(_nested_cfg(inflow_perturbation=True), parent=parent)
    return ip.build_inflow_perturbation(child)


def _indices_over(rows, generator, clock):
    parent = _ClockNode(clock)
    out = []
    for row in rows:
        clock.at(*row)
        out.append(generator._refresh_for(parent))
    return out


def _hold_spans(rows, indices):
    """Model seconds each index value spanned, in force order."""
    spans = []
    start, current = rows[0][0], indices[0]
    for (ticks, _step, _dt), index in zip(rows[1:], indices[1:]):
        if index != current:
            spans.append((ticks - start) / _TICK_DEN)
            start, current = ticks, index
    return spans


def test_the_refresh_index_never_goes_backward_under_a_varying_parent_step():
    """A backward index re-emits a Philox draw the face already imprinted.

    The draw is keyed on (seed, grid_id, face, refresh) alone, so index
    k names ONE pattern for the life of the run.  Revisiting k puts that
    pattern back on the boundary and the Davies zone drives the relax
    rows toward a target it has already relaxed toward -- a repeating
    imprint instead of the decorrelating one the mechanism exists to
    make.
    """
    generator = _on_generator()
    clock = _Clock(_SPEC_STEP, _SPEC_DT)
    rows = _adaptive_trajectory()
    indices = _indices_over(rows, generator, clock)
    backward = [(a, b) for a, b in zip(indices, indices[1:]) if b < a]
    assert not backward, (
        f"{len(backward)} backward transitions over {len(rows)} forces, "
        f"largest jump {max(a - b for a, b in backward)} indices "
        f"(e.g. {backward[0][0]} -> {backward[0][1]})")


def test_one_draw_is_held_for_the_pinned_seconds_under_a_varying_step():
    """REFRESH_SECONDS is a contract on MODEL TIME, not on force count.

    The index changes at fixed tick boundaries, so a force can only land
    on one to within a single parent step: the tolerance is the longest
    LIVE step on the trajectory, and nothing wider.
    """
    generator = _on_generator()
    clock = _Clock(_SPEC_STEP, _SPEC_DT)
    rows = _adaptive_trajectory()
    spans = _hold_spans(rows, _indices_over(rows, generator, clock))
    slack = max(dt for _t, _s, dt in rows)
    bad = [s for s in spans if abs(s - ip.REFRESH_SECONDS) > slack]
    assert not bad, (
        f"{len(bad)} of {len(spans)} holds are further than one parent "
        f"step ({slack:.1f} s) from the pinned {ip.REFRESH_SECONDS:.1f} s: "
        f"spans run {min(spans):.1f} s to {max(spans):.1f} s, median "
        f"{sorted(spans)[len(spans) // 2]:.1f} s")


def test_a_replayed_refresh_index_is_refused():
    """The un-regressable half: any future re-conversion raises here."""
    generator = _on_generator()
    clock = _Clock(_SPEC_STEP, _SPEC_DT)
    parent = _ClockNode(clock)
    clock.at(3 * _SPEC_BUCKET_TICKS)
    assert generator._refresh_for(parent) == 3
    clock.at(_SPEC_BUCKET_TICKS)
    with pytest.raises(ValueError, match="already imprinted"):
        generator._refresh_for(parent)


def test_the_generator_reads_the_frozen_parent_step_not_the_live_one():
    """The index at one model time may not depend on the live step.

    ``(400, 4.0)`` agrees with the frozen pair at this tick count and
    ``(600, 6.0)`` does not; the sweep is here so agreeing by luck is
    never mistaken for the contract.
    """
    at_ticks = 100 * _SPEC_BUCKET_TICKS
    for live_step, live_dt in ((_SPEC_STEP, _SPEC_DT), (400, 4.0),
                               (600, 6.0), (700, 7.0), (800, 8.0)):
        generator = _on_generator()
        clock = _Clock(_SPEC_STEP, _SPEC_DT).at(at_ticks, live_step, live_dt)
        got = generator._refresh_for(_ClockNode(clock))
        assert got == 100, (
            f"live (step={live_step} ticks, dt={live_dt} s) moved the index "
            f"at a fixed model time to {got}; the configured pair "
            f"(step={_SPEC_STEP}, dt={_SPEC_DT}) buckets it at 100")


def test_the_cadence_harness_passes_on_a_constant_divisor():
    """The deliberately-wrong input for the two tests above.

    Same trajectory, same assertions, but the divisor handed in is the
    run constant it is supposed to be.  If this ever fails, those two
    are failing on the harness rather than on the varying divisor.
    """
    rows = _adaptive_trajectory()
    indices = [ip.refresh_index(ticks, _SPEC_STEP, _SPEC_DT)
               for ticks, _step, _dt in rows]
    assert all(b >= a for a, b in zip(indices, indices[1:]))
    slack = max(dt for _t, _s, dt in rows)
    assert all(abs(s - ip.REFRESH_SECONDS) <= slack
               for s in _hold_spans(rows, indices))


def test_the_pinned_fixed_clock_sequence_is_unchanged_by_the_frozen_read():
    """Bit-identity on every shipped case.

    The acceptance-v2 pin (item 7) driven through the generator seam
    instead of the bare function: on a fixed clock live == spec, so the
    integer sequence must be the one
    ``test_refresh_holds_for_pinned_seconds`` already holds.  If this
    moves, the change is not a no-op and G1/G2 owe a re-run.
    """
    generator = _on_generator()
    clock = _Clock(15, 3.75)
    parent = _ClockNode(clock)
    ticks = list(range(0, 3 * 27 * 15, 15))
    through_seam = []
    for tick in ticks:
        clock.at(tick)
        through_seam.append(generator._refresh_for(parent))
    assert through_seam == [ip.refresh_index(t, 15, 3.75) for t in ticks]


# ---------------------------------------------------------------------------
# What the run injected -- the outflow control's amplitude match
#
# faces = "outflow" (acceptance G5) is the registered negative control for
# the fetch-reduction claim: the same code, the same seed, the same
# amplitude convention, on the complementary faces, so nothing is advected
# in and the scored inflow face's relax rows are never touched.  It only
# READS as a control if it injected a comparable amount, and it does not
# by construction -- each face's theta_max comes from its own wind and
# enters as U squared.  These tests drive the real force hook and prove
# the receipt says which case a null result is.
# ---------------------------------------------------------------------------

#: Face-mean boundary-normal winds the fake state below produces.
#: west and north ENTER; east and south LEAVE, and more slowly, which is
#: the asymmetry the receipt exists to expose.
_INWARD_MS = {"west": 8.0, "east": -2.0, "south": -1.0, "north": 3.0}


class _ForceState:
    """Just the state fields the force hook and the table write read."""

    def __init__(self, cp, nz, ny, nx):
        rng = np.random.default_rng(4)
        # Full-level base geopotential: 0-4,000 m, so the half levels
        # sit at 250, 750, ... and a 1,000 m PBLH admits exactly two.
        self.phb = cp.asarray(
            (np.linspace(0.0, 4000.0, nz + 1) * c.G).astype(np.float64))
        u = np.zeros((nz, ny, nx), dtype=np.float32)
        v = np.zeros((nz, ny, nx), dtype=np.float32)
        u[:, :, 0] = _INWARD_MS["west"]
        u[:, :, -1] = -_INWARD_MS["east"]
        v[:, 0, :] = _INWARD_MS["south"]
        v[:, -1, :] = -_INWARD_MS["north"]
        self.u = cp.asarray(u)
        self.v = cp.asarray(v)
        self.mub2d = cp.asarray(
            rng.uniform(30000.0, 60000.0, (ny, nx)).astype(np.float32))
        self.c1h = cp.asarray(np.linspace(1.0, 0.2, nz).astype(np.float32))
        self.c2h = cp.asarray(np.linspace(0.0, 40000.0, nz).astype(np.float32))


class _PhysicsDriver:
    def __init__(self, pblh):
        self.fields = {"pblh": pblh}


class _ParentState:
    def __init__(self, physics):
        self.physics = physics


class _ForceNode:
    """A child node as ``apply_at_force`` reads it."""

    def __init__(self, run, state, parent, grid_id=3, parent_id=2):
        self.cfg = _CfgNode._Cfg(run, grid_id, parent_id)
        self.state = state
        self.parent = parent


def _force_pair(cp, faces_mode, nz=8, ny=24, nx=24):
    """One built generator plus the node and tables it perturbs."""
    parent_run = _nested_cfg(bl_pbl_physics=1, km_opt=4)
    parent_cfg = _CfgNode(parent_run, grid_id=2, parent_id=1)
    parent_cfg.state = _ParentState(
        _PhysicsDriver(cp.full((16, 16), 1000.0, dtype=cp.float32)))
    parent_cfg.clock = _Clock(_SPEC_STEP, _SPEC_DT)
    child_cfg = _CfgNode(
        _nested_cfg(inflow_perturbation=True,
                    inflow_perturbation_faces=faces_mode),
        parent=parent_cfg)
    generator = ip.build_inflow_perturbation(child_cfg)
    node = _ForceNode(child_cfg.cfg.run, _ForceState(cp, nz, ny, nx),
                      parent_cfg)
    return generator, node, _tables(cp, nz, ny, nx, 5)


def test_the_force_hook_reports_what_it_injected():
    """The receipt carries amplitude, face length and level count."""
    cp = pytest.importorskip("cupy")
    generator, node, fields = _force_pair(cp, "inflow")
    assert generator.injection_receipt()["totals"] == {}
    generator.apply_at_force(node, fields)

    receipt = generator.injection_receipt()
    assert receipt["faces_mode"] == "inflow"
    assert set(receipt["last_force"]) == {"west", "north"}
    for face, row in receipt["last_force"].items():
        assert row["inward_mean_ms"] == pytest.approx(_INWARD_MS[face],
                                                      rel=1e-5)
        assert row["theta_max_k"] == pytest.approx(
            ip.face_amplitude(_INWARD_MS[face], 1.0), rel=1e-5)
        assert row["face_cells"] == 24
        assert row["levels"] == 2          # 250 m and 750 m, under 1,000 m
        assert row["refresh"] == 0
    assert receipt["totals"]["west"]["forces"] == 1

    generator.apply_at_force(node, fields)
    assert generator.injection_receipt()["totals"]["west"]["forces"] == 2


def test_the_outflow_control_takes_the_complementary_faces():
    """Nothing the control injects lands on the face the meter scores."""
    cp = pytest.importorskip("cupy")
    generator, node, fields = _force_pair(cp, "outflow")
    before = {face: pair[0].copy()
              for face, pair in fields["theta"].items()}
    generator.apply_at_force(node, fields)

    receipt = generator.injection_receipt()
    assert receipt["faces_mode"] == "outflow"
    assert set(receipt["last_force"]) == {"east", "south"}
    for face in ("west", "north"):
        assert cp.array_equal(fields["theta"][face][0], before[face])
    for face in ("east", "south"):
        assert not cp.array_equal(fields["theta"][face][0], before[face])


def test_the_outflow_arm_is_not_amplitude_matched_by_construction():
    """Which is the whole reason the receipt has to carry the number.

    ``theta_max`` goes as U squared on each face's OWN wind, so a
    control on slower leaving faces injects quadratically less.  A null
    D90 result from this arm means "seeding the leaving edges does not
    shorten the fetch" only if the amplitudes are comparable; otherwise
    it means almost nothing was seeded, and the receipt is what tells
    the two apart.
    """
    cp = pytest.importorskip("cupy")
    peak = {}
    for mode in ("inflow", "outflow"):
        generator, node, fields = _force_pair(cp, mode)
        generator.apply_at_force(node, fields)
        peak[mode] = max(row["theta_max_k_max"] for row
                         in generator.injection_receipt()["totals"].values())
    # 8 m/s entering against 2 m/s leaving: 16x, not 1x.
    assert peak["inflow"] / peak["outflow"] == pytest.approx(16.0, rel=1e-4)


def test_zero_amplitude_still_records_the_faces_it_selected():
    """G2's arm must read as "selected, injected nothing", not "absent"."""
    cp = pytest.importorskip("cupy")
    parent_run = _nested_cfg(bl_pbl_physics=1, km_opt=4)
    parent_cfg = _CfgNode(parent_run, grid_id=2, parent_id=1)
    parent_cfg.state = _ParentState(
        _PhysicsDriver(cp.full((16, 16), 1000.0, dtype=cp.float32)))
    parent_cfg.clock = _Clock(_SPEC_STEP, _SPEC_DT)
    child_cfg = _CfgNode(
        _nested_cfg(inflow_perturbation=True,
                    inflow_perturbation_amplitude_scale=0.0),
        parent=parent_cfg)
    generator = ip.build_inflow_perturbation(child_cfg)
    node = _ForceNode(child_cfg.cfg.run, _ForceState(cp, 8, 24, 24),
                      parent_cfg)
    fields = _tables(cp, 8, 24, 24, 5)
    before = {face: pair[0].copy()
              for face, pair in fields["theta"].items()}
    generator.apply_at_force(node, fields)

    receipt = generator.injection_receipt()
    assert set(receipt["last_force"]) == {"west", "north"}
    assert all(row["theta_max_k"] == 0.0
               for row in receipt["last_force"].values())
    # G2: the classification ran and not one table byte moved.
    for face, pair in fields["theta"].items():
        assert cp.array_equal(pair[0], before[face])
