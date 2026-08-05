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
