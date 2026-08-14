"""Experiment/domain schema tests (Phase-5 Task 1, architecture sec. A).

One failing fixture per load-time rejection rule, the derived-chain
exact-rational pins over the committed ``configs/real74_4dom.toml``
(dx 12000/3000/1000/(1000/3) m, dt 60/15/5/(5/3) s), the migration
wrapper, and the CLI experiment-path routing.  All CPU.
"""

from __future__ import annotations

import dataclasses
import struct
import textwrap
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import gpuwm.cli as cli
from gpuwm.experiment import (ExperimentConfig, ProjectionConfig,
                              VerticalConfig, _assert_derived_copies,
                              experiment_from_run_config,
                              is_experiment_toml, load_experiment)
from gpuwm.case_data import load_experiment_case


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", np.float32(value).tobytes())[0]

REPO = Path(__file__).resolve().parents[1]

#: Synthetic two-domain experiment: d01 100x80 at 12 km / 60 s, child
#: ratio (3,3) at (40,30) with 60x60 mass cells (span 20 parent rows,
#: clearances 39/41 west-east and 29/31 south-north, all >= 10).
BASE = """\
[experiment]
name = "synth"
start_time = 1974-04-03T12:00:00
run_seconds = 3600.0
{experiment}

[shared]
nz = 8
ztop = 12000.0
{shared}

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 100
ny = 80
time_step = 60
dx = 12000.0
history_interval_s = 3600.0
{d01}

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 40
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
e_we = 61
e_sn = 61
history_interval_s = 900.0
{d02}
"""


def _write(tmp_path, *, experiment="restart_interval_s = 0.0",
           shared="", d01="", d02="", text=None):
    path = tmp_path / "exp.toml"
    if text is None:
        text = BASE.format(experiment=experiment, shared=shared,
                           d01=d01, d02=d02)
    path.write_text(textwrap.dedent(text))
    return path


def test_load_synthetic_two_domain(tmp_path):
    exp = load_experiment(_write(tmp_path))
    assert exp.name == "synth"
    assert exp.start_time == datetime(1974, 4, 3, 12)
    assert [dc.grid_id for dc in exp.domains] == [1, 2]
    assert exp.root is exp.domains[0]
    assert exp.domain(2).parent_id == 1
    assert [dc.grid_id for dc in exp.children_of(1)] == [2]
    assert exp.children_of(2) == ()
    with pytest.raises(KeyError, match="grid_id=9"):
        exp.domain(9)
    # derived exact rationals and their float projections
    assert exp.dt_exact(1) == Fraction(60)
    assert exp.dt_exact(2) == Fraction(20)
    assert exp.dx_exact(2) == Fraction(4000)
    d02 = exp.domain(2)
    assert d02.run.dt == 20.0 and d02.run.dx == 4000.0
    assert d02.run.nx == 60 and d02.run.ny == 60  # e_we/e_sn - 1
    assert d02.time_step is None
    # flags default per role; clock_dt retired (0.0) in the experiment path
    assert exp.root.run.specified is True and exp.root.run.nested is False
    assert d02.run.specified is False and d02.run.nested is True
    assert exp.root.run.clock_dt == 0.0 and d02.run.clock_dt == 0.0
    assert exp.root.run.grid_id == 1 and d02.run.grid_id == 2
    # history feeds output_interval_s; restart stays on the root clock
    assert exp.root.run.output_interval_s == 3600.0
    assert d02.run.output_interval_s == 900.0
    assert d02.run.restart_interval_s == 0.0
    assert exp.column_chunk == 3125


def test_experiment_acknowledgements_are_a_first_class_repeatable_list(
        tmp_path):
    exp = load_experiment(_write(
        tmp_path,
        experiment=(
            "restart_interval_s = 0.0\n"
            'acknowledgements = ["expert-tuple-v1", "site-policy-v2"]'
        ),
    ))
    assert exp.acknowledgements == ("expert-tuple-v1", "site-policy-v2")


@pytest.mark.parametrize(
    "literal",
    ('"expert-tuple-v1"', '["expert-tuple-v1", 7]', '[""]'),
)
def test_experiment_acknowledgements_refuse_non_list_or_non_ids(
        tmp_path, literal):
    with pytest.raises(ValueError, match="acknowledgements"):
        load_experiment(_write(
            tmp_path,
            experiment=(
                "restart_interval_s = 0.0\n"
                f"acknowledgements = {literal}"
            ),
        ))


def test_loads_explicit_mp8_to_mp18_domain_transition(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", "1")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(tmp_path))
    shared = "moist = true\nmoist_cq = true\nmp_physics = 8"
    transition = (
        "mp_physics = 18\n"
        "nest_microphysics_transition = "
        '"mp8-to-mp18-mass-diagnosed-v1"'
    )
    exp = load_experiment(_write(
        tmp_path, shared=shared, d02=transition))
    assert [dc.run.mp_physics for dc in exp.domains] == [8, 18]
    assert exp.domain(2).run.nest_microphysics_transition == \
        "mp8-to-mp18-mass-diagnosed-v1"

    with pytest.raises(ValueError, match="requires explicit"):
        load_experiment(_write(tmp_path, shared=shared, d02="mp_physics = 18"))


def test_column_chunk_is_positive_experiment_integer(tmp_path):
    exp = load_experiment(_write(
        tmp_path,
        experiment="restart_interval_s = 0.0\ncolumn_chunk = 6250"))
    assert exp.column_chunk == 6250

    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="column_chunk"):
            load_experiment(_write(
                tmp_path,
                experiment=("restart_interval_s = 0.0\n"
                            f"column_chunk = {str(value).lower()}")))


def test_load_real74_4dom_derived_chain_pins():
    """THE gate pins: the committed bundle-resolved TOML derives the
    exact rational chain dx 12000/3000/1000/(1000/3) m and dt
    60/15/5/(5/3) s -- the d04 values are exact fractions, never a
    hand-typed 500 m or 1.6667 s."""
    exp, _ = load_experiment_case(REPO / "configs" / "real74_4dom.toml")
    assert exp.name == "real74_4dom"
    assert exp.start_time == datetime(1974, 4, 3, 12)
    assert exp.run_seconds == 43200.0
    assert exp.restart_interval_s == 10800.0
    assert exp.feedback == 0 and exp.smooth_option == 0
    assert exp.blend_width == 5 and exp.spec_bdy_width == 5
    assert [dc.grid_id for dc in exp.domains] == [1, 2, 3, 4]
    assert [exp.dx_exact(g) for g in (1, 2, 3, 4)] == [
        Fraction(12000), Fraction(3000), Fraction(1000),
        Fraction(1000, 3)]
    assert [exp.dt_exact(g) for g in (1, 2, 3, 4)] == [
        Fraction(60), Fraction(15), Fraction(5), Fraction(5, 3)]
    assert exp.root.time_step == 60
    assert exp.root.time_step_fract_num == 0
    assert exp.root.time_step_fract_den == 1
    # run.dt carries WRF's CHAINED-FP32 REAL grid%dt exactly (§C,
    # set_timekeeping.F:368): np.float32(60)/4/3/3, NOT float division
    # of the rational -- d04 is 0x3FD55555, the WRF bit pattern.
    chained = np.float32(60)
    for g, ratio in ((1, 1), (2, 4), (3, 3), (4, 3)):
        chained = chained / np.float32(ratio)
        assert exp.domain(g).run.dt == float(chained), g
    assert exp.domain(4).run.dt == 1.6666666269302368
    assert _fp32_bits(exp.domain(4).run.dt) == 0x3FD55555
    assert [exp.domain(g).run.dt for g in (1, 2, 3)] == [60.0, 15.0, 5.0]
    assert [exp.domain(g).run.dx for g in (1, 2, 3, 4)] == [
        12000.0, 3000.0, 1000.0, float(Fraction(1000, 3))]
    # staggered namelist dims -> mass dims; one vertical grid,
    # single-sourced from the F1 VerticalConfig value object
    assert [(exp.domain(g).run.nx, exp.domain(g).run.ny)
            for g in (1, 2, 3, 4)] == [(250, 200), (500, 400),
                                       (501, 501), (600, 600)]
    assert all(exp.domain(g).run.nz == 49 for g in (1, 2, 3, 4))
    assert isinstance(exp.vertical, VerticalConfig)
    assert len(exp.vertical.eta_levels) == 50
    assert exp.vertical.p_top == 10000.0
    assert exp.vertical.hybrid_opt == 2 and exp.vertical.etac == 0.2
    # per-domain scalars from the namelist
    assert [exp.domain(g).run.radt for g in (1, 2, 3, 4)] == [
        12.0, 3.0, 1.0, 1.0]
    assert [exp.domain(g).run.diff_6th_factor for g in (1, 2, 3, 4)] == [
        0.12, 0.10, 0.08, 0.06]
    assert [exp.domain(g).run.epssm for g in (1, 2, 3, 4)] == [
        0.5, 0.1, 0.1, 0.1]
    assert all(exp.domain(g).run.moist_cq for g in (1, 2, 3, 4))
    assert [exp.domain(g).run.cu_physics for g in (1, 2, 3, 4)] == [
        1, 0, 0, 0]
    assert [exp.domain(g).history_interval_s for g in (1, 2, 3, 4)] == [
        3600.0, 900.0, 900.0, 900.0]
    assert [(exp.domain(g).run.specified, exp.domain(g).run.nested)
            for g in (1, 2, 3, 4)] == [(True, False), (False, True),
                                       (False, True), (False, True)]
    # F14 timing authority: every RunConfig carries the derived copies
    # of the experiment's authoritative timing values
    for dc in exp.domains:
        assert dc.run.restart_interval_s == exp.restart_interval_s
        assert dc.run.run_seconds == exp.run_seconds
        assert dc.run.output_interval_s == dc.history_interval_s
    assert [dc.parent_id for dc in exp.domains] == [0, 1, 2, 3]
    assert exp.projection is not None
    assert exp.projection.map_proj == "lambert"
    assert exp.projection.ref_lat == 39.6848


def test_experiment_from_run_config_wraps_verbatim():
    from gpuwm.verify.cases import real74_d01, straka
    cfg = real74_d01.phase3_config()
    exp = experiment_from_run_config(cfg, datetime(1974, 4, 3, 12))
    assert isinstance(exp, ExperimentConfig)
    assert exp.name == "real74_d01"
    assert len(exp.domains) == 1
    dom = exp.root
    assert dom.run is cfg                      # carried VERBATIM
    assert dom.grid_id == 1 and dom.parent_id == 0
    assert dom.time_step == 60
    assert exp.dt_exact(1) == Fraction(60)
    assert exp.run_seconds == cfg.run_seconds
    assert dom.history_interval_s == cfg.output_interval_s
    # projection = None is RESERVED for wrapped configs; the vertical
    # value object carries the hybrid selectors as compatibility copies
    assert exp.projection is None
    assert exp.vertical == VerticalConfig(
        eta_levels=(), p_top=0.0, hybrid_opt=cfg.hybrid_opt,
        etac=cfg.etac)
    # fractional dt decomposes exactly into the WRF rational clock keys
    frac = experiment_from_run_config(straka.default_config(),
                                      datetime(2000, 1, 1))
    dt = frac.root.run.dt
    assert frac.dt_exact(frac.root.grid_id) == Fraction(dt)
    assert (frac.root.time_step
            + Fraction(frac.root.time_step_fract_num,
                       frac.root.time_step_fract_den)) == Fraction(dt)


# ---------------------------------------------------------------------------
# One failing fixture per rejection rule (architecture section A).
# ---------------------------------------------------------------------------

def test_rejects_missing_experiment_table(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[[domain]]\ngrid_id = 1\n")
    with pytest.raises(ValueError, match=r"must carry an \[experiment\]"):
        load_experiment(path)


def test_rejects_missing_domain_tables(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[experiment]\nname = "x"\n')
    with pytest.raises(ValueError, match=r"at least one \[\[domain\]\]"):
        load_experiment(path)


def test_refuses_unknown_top_level_table(tmp_path):
    """A stray table is named and REFUSED, not named and dropped.

    This test used to codify warn-and-drop, and the fleet's
    hostile-input node showed what that costs: ``[grid]``,
    ``[dynamics]`` and ``[run]`` are the LEGACY config schema's table
    names, so pasting a familiar block into an experiment config used
    to drop every setting in it behind one stderr line and run the
    defaults under the reader's name for them.  The rest of this bar
    lives in tests/test_dynamics_key_validation.py.
    """
    with pytest.raises(ValueError) as caught:
        load_experiment(_write(tmp_path, text=BASE.format(
            experiment="restart_interval_s = 0.0", shared="", d01="",
            d02="") + "\n[grid]\nnx = 4\n"))
    assert "grid" in str(caught.value)


def test_rejects_non_root_first_domain(tmp_path):
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace("parent_id = 0",
                                               "parent_id = 2", 1)
    with pytest.raises(ValueError, match="must be the root"):
        load_experiment(_write(tmp_path, text=text))


def test_rejects_second_root(tmp_path):
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace("parent_id = 1",
                                               "parent_id = 0")
    with pytest.raises(ValueError, match="second root"):
        load_experiment(_write(tmp_path, text=text))


def test_root_flag_violations_are_corrected_with_a_warning(
        tmp_path, capsys):
    """Warn-not-block: the flags derive from parent_id; a contradiction
    is corrected in one line, never a refusal."""
    for override in ("specified = false", "nested = true"):
        exp = load_experiment(_write(tmp_path, d01=override))
        err = capsys.readouterr().err
        assert "warning:" in err and "corrected" in err
        assert exp.root.run.specified is True
        assert exp.root.run.nested is False


def test_child_flag_violations_are_corrected_with_a_warning(
        tmp_path, capsys):
    for override in ("specified = true", "nested = false"):
        exp = load_experiment(_write(tmp_path, d02=override))
        err = capsys.readouterr().err
        assert "warning:" in err and "corrected" in err
        assert exp.domain(2).run.nested is True
        assert exp.domain(2).run.specified is False


def test_rejects_child_ratio_below_two(tmp_path):
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace(
        "parent_grid_ratio = 3", "parent_grid_ratio = 1")
    with pytest.raises(ValueError,
                       match=r"parent_grid_ratio = 1 .* >= 2"):
        load_experiment(_write(tmp_path, text=text))
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace(
        "parent_time_step_ratio = 3", "parent_time_step_ratio = 1")
    with pytest.raises(ValueError,
                       match=r"parent_time_step_ratio = 1 .* >= 2"):
        load_experiment(_write(tmp_path, text=text))


def test_rejects_insufficient_footprint_clearance(tmp_path):
    # west clearance 4 < spec_bdy_width + blend_width = 10
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace("i_parent_start = 40",
                                               "i_parent_start = 5")
    with pytest.raises(ValueError, match="parent-row clearance"):
        load_experiment(_write(tmp_path, text=text))
    # a footprint running off the parent edge is negative clearance
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace("i_parent_start = 40",
                                               "i_parent_start = 95")
    with pytest.raises(ValueError, match="parent-row clearance"):
        load_experiment(_write(tmp_path, text=text))


def test_rejects_unaligned_footprint(tmp_path):
    # 61 mass cells % ratio 3 != 0 (WPS: e_we = n*ratio + 1)
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace("e_we = 61", "e_we = 62")
    with pytest.raises(ValueError, match="integer multiple"):
        load_experiment(_write(tmp_path, text=text))


def test_rejects_vertical_nesting(tmp_path):
    """F1 amendment: ANY per-domain vertical key is rejected outright --
    the vertical grid is single-sourced from ExperimentConfig.vertical,
    so even a value MATCHING [shared] is a load error."""
    for key in ("nz = 7", "e_vert = 8", "p_top = 5000.0",
                "eta_levels = [1.0, 0.5, 0.0]",
                "nz = 8",              # matches [shared] -- still rejected
                "hybrid_opt = 2", "etac = 0.2", "ztop = 12000.0"):
        with pytest.raises(ValueError,
                           match="vertical nesting is rejected"):
            load_experiment(_write(tmp_path, d02=key))
    with pytest.raises(ValueError, match="vertical nesting is rejected"):
        load_experiment(_write(tmp_path, d01="eta_levels = [1.0, 0.0]"))
    with pytest.raises(ValueError, match="vert_refine_method"):
        load_experiment(_write(tmp_path, shared="vert_refine_method = 1"))


def test_rejects_non_sint_interp_method(tmp_path):
    with pytest.raises(ValueError, match="only SINT"):
        load_experiment(_write(tmp_path, shared="interp_method_type = 1"))
    # the WRF default (2 = SINT, Registry.EM_COMMON:2301) is accepted
    exp = load_experiment(_write(tmp_path,
                                 shared="interp_method_type = 2"))
    assert len(exp.domains) == 2


def test_rejects_isobaric_nest_interp_coord(tmp_path):
    with pytest.raises(ValueError, match="nest_interp_coord"):
        load_experiment(_write(tmp_path, shared="nest_interp_coord = 1"))


def test_rejects_input_from_hires(tmp_path):
    with pytest.raises(ValueError, match="input_from_hires"):
        load_experiment(_write(tmp_path,
                               shared="input_from_hires = true"))


def test_rejects_moving_nest_keys(tmp_path):
    with pytest.raises(ValueError, match="moving-nest"):
        load_experiment(_write(tmp_path, shared="num_moves = 4"))
    with pytest.raises(ValueError, match="moving-nest"):
        load_experiment(_write(tmp_path, d02="vortex_interval = 15"))


def test_nonzero_child_spec_exp_is_forced_to_zero_with_a_warning(
        tmp_path, capsys):
    """WRF's nested lbc_fcx_gcx branch has no sponge term
    (module_bc_em.F:1297-1341): a nonzero child spec_exp is forced to 0
    with one warning line, which is exactly how WRF would run this
    namelist -- the N1 pin (children force with spec_exp=0) holds."""
    exp = load_experiment(_write(tmp_path, d02="spec_exp = 0.33"))
    err = capsys.readouterr().err
    assert "warning:" in err and "spec_exp" in err
    assert exp.domain(2).run.spec_exp == 0.0
    # the root (specified branch) legitimately carries a sponge exponent
    exp = load_experiment(_write(tmp_path, d01="spec_exp = 0.33"))
    assert exp.root.run.spec_exp == 0.33
    assert exp.domain(2).run.spec_exp == 0.0


def test_accepts_tree_wide_experimental_feedback_and_rejects_other_values(
        tmp_path):
    exp = load_experiment(_write(
        tmp_path,
        experiment="restart_interval_s = 0.0\nfeedback = 1"))
    assert exp.feedback == 1
    with pytest.raises(ValueError, match=r"feedback must be 0 .* or 1"):
        load_experiment(_write(
            tmp_path,
            experiment="restart_interval_s = 0.0\nfeedback = 2"))


def test_rejects_nonzero_smooth_option(tmp_path):
    with pytest.raises(ValueError, match="smooth_option"):
        load_experiment(_write(
            tmp_path,
            experiment="restart_interval_s = 0.0\nsmooth_option = 2"))


def test_rejects_non_divisible_cadences(tmp_path):
    # history: 50 s on the child's exact 20 s step = 2.5 steps
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace(
        "history_interval_s = 900.0", "history_interval_s = 50.0")
    with pytest.raises(ValueError, match="whole number"):
        load_experiment(_write(tmp_path, text=text))
    # radt: 0.7 min = 42 s on the 60 s root step (radiation active)
    with pytest.raises(ValueError, match="radt"):
        load_experiment(_write(tmp_path, shared="ra_physics = 90",
                               d01="radt = 0.7"))
    # cudt: 0.7 min = 42 s with Kain-Fritsch on the root.  The moist state is
    # not decoration: Kain-Fritsch is a moist convective scheme, and both
    # validate_run_config and gpuwm/core/physics.py initialize_physics refuse
    # a cumulus scheme on a dry DomainState -- so a dry cu_physics=1 config
    # never reached the cadence check it is this leg's job to exercise.
    with pytest.raises(ValueError, match="cudt"):
        load_experiment(_write(tmp_path, shared="moist = true",
                               d01="cu_physics = 1\ncudt_minutes = 0.7"))
    # bldt: 0.7 min = 42 s with a PBL scheme selected.  The surface layer
    # rides along for the same reason as the moist state above: a PBL scheme
    # without one is refused before any cadence is examined.
    with pytest.raises(ValueError, match="bldt"):
        load_experiment(_write(
            tmp_path,
            shared="bl_pbl_physics = 1\nsf_sfclay_physics = 91",
            d01="bldt = 0.7"))
    # restart: 90 s on the 60 s root step
    with pytest.raises(ValueError, match="restart_interval_s"):
        load_experiment(_write(tmp_path,
                               experiment="restart_interval_s = 90.0"))
    # zero-cadence (every step) and exact multiples stay legal
    exp = load_experiment(_write(
        tmp_path, shared="bl_pbl_physics = 1\nsf_sfclay_physics = 91",
        d01="bldt = 0.0"))
    assert exp.root.run.bldt == 0.0


def test_rejects_hand_typed_child_dx_mismatch(tmp_path):
    """THE pinned 500-m fixture: a hand-typed 500 m child dx against
    ratio 3 from 1 km is a hard error -- the chain derives 1000/3 m."""
    text = """\
    [experiment]
    name = "chain"
    start_time = 1974-04-03T12:00:00
    run_seconds = 3600.0
    restart_interval_s = 0.0

    [shared]
    nz = 8
    ztop = 12000.0

    [[domain]]
    grid_id = 1
    parent_id = 0
    i_parent_start = 1
    j_parent_start = 1
    parent_grid_ratio = 1
    parent_time_step_ratio = 1
    nx = 100
    ny = 80
    time_step = 60
    dx = 3000.0
    history_interval_s = 3600.0

    [[domain]]
    grid_id = 2
    parent_id = 1
    i_parent_start = 40
    j_parent_start = 30
    parent_grid_ratio = 3
    parent_time_step_ratio = 3
    nx = 60
    ny = 60
    history_interval_s = 900.0

    [[domain]]
    grid_id = 3
    parent_id = 2
    i_parent_start = 25
    j_parent_start = 25
    parent_grid_ratio = 3
    parent_time_step_ratio = 3
    nx = 30
    ny = 30
    history_interval_s = 900.0
    {d03}
    """
    # the chain is d01 3 km -> d02 1 km -> d03 exactly 1000/3 m
    exp = load_experiment(_write(tmp_path,
                                 text=text.format(d03="")))
    assert exp.dx_exact(3) == Fraction(1000, 3)
    assert exp.dt_exact(3) == Fraction(20, 3)
    with pytest.raises(ValueError, match="never hand-typed"):
        load_experiment(_write(tmp_path,
                               text=text.format(d03="dx = 500.0")))
    # a hand-typed dt mismatch is equally fatal
    with pytest.raises(ValueError, match="never hand-typed"):
        load_experiment(_write(tmp_path,
                               text=text.format(d03="dt = 5.0")))
    # a truncated decimal of the true value passes the cross-check
    exp = load_experiment(_write(tmp_path,
                                 text=text.format(d03="dx = 333.333333")))
    assert exp.dx_exact(3) == Fraction(1000, 3)
    assert exp.domain(3).run.dx == float(Fraction(1000, 3))


def test_a_child_time_step_that_matches_the_chain_is_accepted_quietly(
        tmp_path, capsys):
    """A child time_step is redundant, not wrong, when it agrees.

    This test used to assert a WARNING here, on the value 20 -- which is
    exactly what the chain derives in this fixture (60 / 3).  The user
    wrote 20 and got 20, so the assertion was satisfied identically by
    "warn and honour what was written" and by "warn and substitute", and
    it read as a ruling that the substitution was fine.  The disagreeing
    case, which is the one that mattered, had no test at all.
    """
    exp = load_experiment(_write(tmp_path, d02="time_step = 20"))
    assert "time_step" not in capsys.readouterr().err
    assert exp.dt_exact(2) == exp.dt_exact(1) / 3


def test_a_child_time_step_against_the_chain_is_refused(tmp_path):
    """The sixth warn-and-drop, and its own sibling contradicted it.

    `dt` on a child refuses through _cross_check; `time_step` on the
    same table -- the same quantity, the same number -- warned and ran
    the derived step at exit 0.  The old warning's `why=` even
    advertised the check it declined to perform.

    That WRF has no per-child time_step is the reason to REFUSE: the
    model cannot deliver the step that was asked for.
    """
    for key in ("time_step = 5", "time_step_fract_num = 1",
                "time_step_fract_den = 7"):
        with pytest.raises(ValueError, match="time_step"):
            load_experiment(_write(tmp_path, d02=key))


def test_rejects_root_without_time_step(tmp_path):
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace("time_step = 60\n", "")
    with pytest.raises(ValueError, match="must carry time_step"):
        load_experiment(_write(tmp_path, text=text))


def test_a_root_dt_that_matches_time_step_is_accepted_quietly(
        tmp_path, capsys):
    """Same shape as the child case above, and the same blind spot.

    The old test asserted a warning on `dt = 60.0` in a fixture whose
    `time_step` IS 60, so the user wrote 60 and got 60 either way.  The
    warning was also emitted verbatim when the two AGREED, so it carried
    no information about the only thing worth knowing.
    """
    exp = load_experiment(_write(tmp_path, d01="dt = 60.0"))
    assert "dt" not in capsys.readouterr().err
    assert exp.dt_exact(1) == 60


def test_a_root_dt_against_its_time_step_is_refused(tmp_path):
    """Measured on the published 1.4.0 wheel: `dt = 60.0` beside
    `time_step = 5` warned, loaded, and ran a 5 s step at rc 0 -- the
    root step was discarded with no comparison performed at all."""
    with pytest.raises(ValueError, match=r"hand-types dt=5\.0"):
        load_experiment(_write(tmp_path, d01="dt = 5.0"))
    # and the fractional part is compared too, not just the integer:
    # time_step 60 + 1/2 is 60.5 s, so a bare dt = 60.0 disagrees.
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="dt = 60.0", d02="").replace(
        "time_step = 60\n",
        "time_step = 60\ntime_step_fract_num = 1\n"
        "time_step_fract_den = 2\n")
    with pytest.raises(ValueError, match="hand-types dt"):
        load_experiment(_write(tmp_path, text=text))


def test_rejects_root_without_dx(tmp_path):
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace("dx = 12000.0\n", "")
    with pytest.raises(ValueError, match="must carry dx"):
        load_experiment(_write(tmp_path, text=text))


def test_rejects_anisotropic_root(tmp_path):
    with pytest.raises(ValueError, match="isotropic"):
        load_experiment(_write(tmp_path, d01="dy = 6000.0"))


def test_rejects_duplicate_grid_id(tmp_path):
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace("grid_id = 2",
                                               "grid_id = 1")
    with pytest.raises(ValueError, match="duplicate grid_id"):
        load_experiment(_write(tmp_path, text=text))


def test_rejects_unknown_parent_reference(tmp_path):
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace("parent_id = 1",
                                               "parent_id = 7")
    with pytest.raises(ValueError, match="parent-before-child"):
        load_experiment(_write(tmp_path, text=text))


def test_rejects_inconsistent_staggered_and_mass_dims(tmp_path):
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        load_experiment(_write(tmp_path, d02="nx = 61"))  # e_we=61 -> 60


def test_unknown_keys_and_misplaced_keys_both_refuse_by_name(tmp_path):
    """An unknown key refuses by name, and so does a misplaced one.

    The unknown-key half used to warn and drop.  A key with a default
    is optional by construction, so no required-key or value check ever
    catches a misspelling of one -- the run simply integrates with the
    built-in value under the reader's name for it.  Both halves now say
    what they refused; only the sentence differs, because a MISPLACED
    key can additionally say which table it belongs in.
    """
    with pytest.raises(ValueError, match=r"does not have a key"):
        load_experiment(_write(tmp_path, shared="not_a_key = 1"))
    # The [[domain]] half of the same refusal.  The key here has to be a
    # genuine NON-key: `hill_height` is a real RunConfig field that is
    # merely in the wrong table, so it now takes the more specific
    # misplaced-key branch asserted at the end of this test rather than
    # the unknown-key one.
    with pytest.raises(ValueError, match=r"does not have a key"):
        load_experiment(_write(tmp_path, d02="not_a_domain_key = 5.0"))
    # KEEP-HARD negatives: misplaced and retired keys refuse by name.
    with pytest.raises(ValueError, match=r"belongs in \[\[domain\]\]"):
        load_experiment(_write(tmp_path, shared="dx = 1000.0"))
    with pytest.raises(ValueError, match="retired"):
        load_experiment(_write(tmp_path, shared="clock_dt = 60.0"))
    # A real RunConfig field in [[domain]] is NOT a typo, and used to
    # warn and drop -- so `hill_height = 5.0` on a nest ran the shared
    # 100 m hill (gpuwm/core/terrain.py:26 reads cfg.hill_height) and
    # reported success.  That is the docstring's own "honoring or
    # dropping it silently would change the run", so it refuses now.
    with pytest.raises(ValueError,
                       match="are run settings that are NOT per domain"):
        load_experiment(_write(tmp_path, d02="hill_height = 5.0"))


def test_rejects_bad_eta_levels(tmp_path):
    with pytest.raises(ValueError, match="e_vert = nz \\+ 1"):
        load_experiment(_write(
            tmp_path, shared="eta_levels = [1.0, 0.5, 0.0]"))
    good = "eta_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0]"
    exp = load_experiment(_write(tmp_path, shared=good))
    assert len(exp.vertical.eta_levels) == 9
    with pytest.raises(ValueError, match="strictly decreasing"):
        load_experiment(_write(
            tmp_path,
            shared="eta_levels = [1.0, 0.9, 0.9, 0.7, 0.6, 0.5, 0.4, "
                   "0.2, 0.0]"))


@pytest.mark.parametrize("mass_levels", [4, 7, 49, 80, 113])
def test_explicit_eta_level_count_is_fully_parameterized(tmp_path, mass_levels):
    eta = np.linspace(1.0, 0.0, mass_levels + 1)
    eta_text = ", ".join(repr(float(value)) for value in eta)
    text = BASE.format(
        experiment="restart_interval_s = 0.0",
        shared=f"eta_levels = [{eta_text}]",
        d01="",
        d02="",
    ).replace("nz = 8\n", f"nz = {mass_levels}\n", 1)
    exp = load_experiment(_write(tmp_path, text=text))
    assert exp.root.run.nz == mass_levels
    assert exp.vertical.mass_level_count == mass_levels
    assert exp.vertical.interface_level_count == mass_levels + 1


def test_vertical_count_accepts_wrf_e_vert_and_derives_nz(tmp_path):
    eta = ", ".join(repr(float(value))
                    for value in np.linspace(1.0, 0.0, 12))
    text = BASE.format(
        experiment="restart_interval_s = 0.0",
        shared=f"e_vert = 12\neta_levels = [{eta}]",
        d01="",
        d02="",
    ).replace("nz = 8\n", "", 1)
    exp = load_experiment(_write(tmp_path, text=text))
    assert exp.root.run.nz == 11
    assert exp.vertical.interface_level_count == 12


def test_vertical_count_derives_from_eta_and_rejects_conflicts(tmp_path):
    eta = ", ".join(repr(float(value))
                    for value in np.linspace(1.0, 0.0, 10))
    text = BASE.format(
        experiment="restart_interval_s = 0.0",
        shared=f"eta_levels = [{eta}]",
        d01="",
        d02="",
    ).replace("nz = 8\n", "", 1)
    exp = load_experiment(_write(tmp_path, text=text))
    assert exp.root.run.nz == 9

    conflict = text.replace(
        f"eta_levels = [{eta}]", f"nz = 9\ne_vert = 12\neta_levels = [{eta}]")
    with pytest.raises(ValueError, match="inconsistent vertical counts"):
        load_experiment(_write(tmp_path, text=conflict))


def test_experiment_path_runs_runconfig_invariant_battery(tmp_path):
    """Review F1: the per-domain RunConfigs pass through the SAME
    invariant battery as load_config (validate_run_config) -- the three
    probe cases that previously loaded cleanly all fail loudly."""
    with pytest.raises(ValueError, match="mp_physics"):
        load_experiment(_write(tmp_path,
                               shared="moist = true\nmp_physics = 55"))
    with pytest.raises(ValueError, match="time_step_sound must be even"):
        load_experiment(_write(tmp_path, shared="time_step_sound = 3"))
    with pytest.raises(NotImplementedError, match="khdif"):
        load_experiment(_write(tmp_path, shared="khdif = 100.0"))
    with pytest.raises(ValueError, match="diff_6th_opt"):
        load_experiment(_write(
            tmp_path, shared="moist = true\nmp_physics = 10\n"
                             "diff_6th_opt = 1"))
    with pytest.raises(ValueError, match="hypsometric_opt"):
        load_experiment(_write(tmp_path, shared="hypsometric_opt = 3"))


def test_chained_fp32_dt_diverges_from_direct_rounding(tmp_path):
    """§C chained-FP32 policy on a chain where chained and direct
    rounding DIFFER: root 1 s with ratios (2, 7, 5) -- WRF's REAL
    grid%dt = parent/ratio at every edge lands one ULP away from
    float32(1/70)."""
    text = """\
    [experiment]
    name = "chain275"
    start_time = 2000-01-01T00:00:00
    run_seconds = 3600.0
    restart_interval_s = 0.0

    [shared]
    nz = 8
    ztop = 12000.0

    [[domain]]
    grid_id = 1
    parent_id = 0
    i_parent_start = 1
    j_parent_start = 1
    parent_grid_ratio = 1
    parent_time_step_ratio = 1
    nx = 200
    ny = 200
    time_step = 1
    dx = 70000.0
    history_interval_s = 3600.0

    [[domain]]
    grid_id = 2
    parent_id = 1
    i_parent_start = 50
    j_parent_start = 50
    parent_grid_ratio = 2
    parent_time_step_ratio = 2
    nx = 100
    ny = 100
    history_interval_s = 3600.0

    [[domain]]
    grid_id = 3
    parent_id = 2
    i_parent_start = 40
    j_parent_start = 40
    parent_grid_ratio = 7
    parent_time_step_ratio = 7
    nx = 70
    ny = 70
    history_interval_s = 3600.0

    [[domain]]
    grid_id = 4
    parent_id = 3
    i_parent_start = 25
    j_parent_start = 25
    parent_grid_ratio = 5
    parent_time_step_ratio = 5
    nx = 50
    ny = 50
    history_interval_s = 3600.0
    """
    exp = load_experiment(_write(tmp_path, text=text))
    assert exp.dt_exact(4) == Fraction(1, 70)
    chained = (np.float32(1) / np.float32(2) / np.float32(7)
               / np.float32(5))
    direct = np.float32(1.0 / 70.0)
    assert _fp32_bits(float(chained)) != _fp32_bits(float(direct))
    assert exp.domain(4).run.dt == float(chained)
    assert _fp32_bits(exp.domain(4).run.dt) == _fp32_bits(float(chained))


def test_rejects_non_finite_vertical_and_projection(tmp_path):
    """Shadow S3: NaN/inf vertical or projection values must not pass
    the fail-loud schema; the invariants live on the frozen value
    objects so programmatic construction cannot bypass them."""
    with pytest.raises(ValueError, match="not finite"):
        load_experiment(_write(
            tmp_path,
            shared="eta_levels = [1.0, 0.9, 0.8, 0.7, nan, 0.5, 0.4, "
                   "0.2, 0.0]"))
    with pytest.raises(ValueError, match="p_top"):
        load_experiment(_write(tmp_path, shared="p_top = inf"))
    with pytest.raises(ValueError, match="not finite"):
        VerticalConfig(eta_levels=(1.0, float("nan"), 0.0), p_top=0.0,
                       hybrid_opt=0, etac=0.2)
    with pytest.raises(ValueError, match="etac"):
        VerticalConfig(eta_levels=(), p_top=0.0, hybrid_opt=0,
                       etac=float("nan"))
    proj = dict(map_proj="lambert", ref_lat=39.7, ref_lon=-83.9,
                truelat1=30.0, truelat2=60.0, stand_lon=-83.9)
    with pytest.raises(ValueError, match="ref_lat"):
        ProjectionConfig(**{**proj, "ref_lat": float("nan")})
    with pytest.raises(ValueError, match="ref_lat"):
        ProjectionConfig(**{**proj, "ref_lat": 95.0})
    with pytest.raises(ValueError, match="stand_lon"):
        ProjectionConfig(**{**proj, "stand_lon": float("inf")})
    with pytest.raises(
            ValueError,
            match=r"angular dx/dy.*polar filter.*map_proj == 6"):
        ProjectionConfig(**{**proj, "map_proj": "rotated-lat-lon"})
    base = BASE.format(experiment="restart_interval_s = 0.0",
                       shared="map_proj = 1", d01="", d02="")
    text = base + (
        "\n[projection]\nmap_proj = \"lambert\"\nref_lat = nan\n"
        "ref_lon = -83.9\ntruelat1 = 30.0\ntruelat2 = 60.0\n"
        "stand_lon = -83.9\n")
    with pytest.raises(ValueError, match=r"\[projection\].*ref_lat"):
        load_experiment(_write(tmp_path, text=text))


def _projection_table(name="lambert"):
    return (f"\n[projection]\nmap_proj = \"{name}\"\nref_lat = 39.7\n"
            "ref_lon = -83.9\ntruelat1 = 30.0\ntruelat2 = 60.0\n"
            "stand_lon = -83.9\n")


def test_projection_and_map_proj_disagreement(tmp_path, capsys):
    # KEEP-HARD: Lambert experiments (map_proj = 1) REQUIRE a
    # [projection] table -- there is nothing to infer the parameters
    # from.
    with pytest.raises(ValueError, match=r"no \[projection\]"):
        load_experiment(_write(tmp_path, shared="map_proj = 1"))
    # OMITTED, not contradicted.  `shared=""` writes no map_proj at all,
    # so RunConfig's default 0 is the dataclass's, not the user's, and
    # the table completes the config without anything disagreeing.  This
    # case previously emitted a warning and this test asserted it,
    # calling the omission "a contradiction the table resolves" and
    # citing warn-not-block -- which is how the genuine contradiction
    # below came to be dropped through the same branch.  An omission
    # earns no warning.
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="") + _projection_table()
    exp = load_experiment(_write(tmp_path, text=text))
    err = capsys.readouterr().err
    assert "map_proj" not in err
    assert exp.projection is not None
    assert exp.root.run.map_proj == 1


def test_declared_map_proj_against_the_table_is_refused(tmp_path):
    """A [shared] map_proj the user wrote, disagreeing with the
    [projection] table, is refused rather than replaced.

    Both values are the user's, and the loser names the projection the
    grid is defined on.  `map_proj = 3` beside a "lambert" table used to
    warn once and integrate on Lambert anyway, with `gpuwm check` PASS
    at rc 0 -- measured on the published 1.4.0 wheel.
    """
    for declared, table in ((3, "lambert"), (1, "mercator"), (0, "lambert")):
        text = BASE.format(
            experiment="restart_interval_s = 0.0",
            shared=f"map_proj = {declared}", d01="", d02="",
        ) + _projection_table(table)
        with pytest.raises(ValueError, match=r"contradicts the \[projection\]"):
            load_experiment(_write(tmp_path, text=text))
    # The agreeing case stays quiet and keeps working.
    text = BASE.format(experiment="restart_interval_s = 0.0",
                       shared="map_proj = 1", d01="", d02="") \
        + _projection_table()
    assert load_experiment(
        _write(tmp_path, text=text)).root.run.map_proj == 1


def test_rejects_moving_nest_keys_in_experiment_table(tmp_path):
    with pytest.raises(ValueError, match="moving-nest"):
        load_experiment(_write(
            tmp_path, experiment="restart_interval_s = 0.0\n"
                                 "num_moves = 4"))


def test_fingerprint_semantics_of_value_objects():
    """F1: frozen/hashable value objects inside the experiment identity
    -- experiments differing in one eta level or one projection
    parameter compare distinct."""
    exp, _ = load_experiment_case(REPO / "configs" / "real74_4dom.toml")
    assert isinstance(hash(exp), int)          # hashable end-to-end
    eta = list(exp.vertical.eta_levels)
    eta[25] = eta[25] - 1.0e-6
    bumped = dataclasses.replace(exp.vertical,
                                 eta_levels=tuple(eta))
    assert bumped != exp.vertical
    assert dataclasses.replace(exp, vertical=bumped) != exp
    proj = dataclasses.replace(exp.projection, ref_lat=39.6849)
    assert dataclasses.replace(exp, projection=proj) != exp


def test_derived_copy_assertions_fire_on_divergence():
    """F14 timing authority: a diverged compatibility copy is a loud
    error (by construction the loader cannot produce one; the assertion
    guards future drift)."""
    exp, _ = load_experiment_case(REPO / "configs" / "real74_4dom.toml")
    _assert_derived_copies(exp, "self-check")   # green on a loaded exp
    bad = dataclasses.replace(exp, run_seconds=exp.run_seconds + 60.0)
    with pytest.raises(ValueError, match="F14 timing authority"):
        _assert_derived_copies(bad, "test")


def test_rejects_non_datetime_start_time(tmp_path):
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace(
        "start_time = 1974-04-03T12:00:00",
        'start_time = "1974-04-03"')
    with pytest.raises(ValueError, match="offset-free TOML datetime"):
        load_experiment(_write(tmp_path, text=text))


def test_rejects_delayed_start_between_parent_steps_with_exact_ratio(
        tmp_path):
    path = _write(
        tmp_path,
        d02="start_time = 1974-04-03T12:00:50")
    with pytest.raises(
            ValueError,
            match=(r"d02.*parent step boundary.*d01 dt = 60 s exactly.*"
                   r"child-parent start offset.*5/6")):
        load_experiment(path)


# ---------------------------------------------------------------------------
# CLI routing: [[domain]]/[experiment] sniff -> experiment path.
# ---------------------------------------------------------------------------

def test_is_experiment_toml_sniff(tmp_path):
    assert is_experiment_toml(REPO / "configs" / "real74_4dom.toml")
    assert not is_experiment_toml(REPO / "configs" / "real74_d01.toml")
    exp_only = tmp_path / "e.toml"
    exp_only.write_text('[experiment]\nname = "x"\n')
    assert is_experiment_toml(exp_only)


def test_cli_routes_experiment_toml_to_experiment_path(tmp_path, capsys):
    """A valid experiment TOML routes to the Task-2 experiment runtime,
    which requires declared [case_data] inputs; an INVALID one surfaces
    its validation error (proof the sniff routes before the legacy
    loader).  The pre-T2 NotImplementedError placeholder is retired."""
    path = _write(tmp_path)
    for command in ("run", "static", "ingest"):
        rc = cli.main([command, str(path)])
        assert rc == 2  # uniform refusal boundary: message + exit 2
        assert "[case_data]" in capsys.readouterr().err
    # A survivable config oddity (child spec_exp) is now a warning on
    # the same route -- the refusal that follows is the missing
    # [case_data], not the oddity.
    bad = _write(tmp_path, d02="spec_exp = 0.5")
    rc = cli.main(["run", str(bad)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "warning:" in err and "spec_exp" in err
    assert "[case_data]" in err


def test_cli_single_domain_summary_needs_no_transition_fields(
        tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    import gpuwm.case_data as case_data
    from gpuwm import runtime

    path = tmp_path / "single-experiment.toml"
    path.write_text("[experiment]\nname = \"single\"\n", encoding="utf-8")
    exp = SimpleNamespace(name="single")
    monkeypatch.setattr(
        case_data, "load_experiment_case",
        lambda _path, **_kwargs: (exp, object()))
    monkeypatch.setattr(
        runtime, "run_experiment", lambda *_args, **_kwargs: SimpleNamespace(
            wrfout_paths=(), completed_seconds=60.0, nan_free=True))
    assert cli.main([
        "run", str(path), "--outdir", str(tmp_path / "out"),
        "--no-supervise",
    ]) == 0
    output = capsys.readouterr().out
    assert "microphysics_transition_receipt': None" in output


def test_cli_legacy_path_untouched(monkeypatch, tmp_path):
    """A legacy TOML still dispatches through load_config to its
    registered case (the experiment sniff must not intercept it)."""
    from types import SimpleNamespace
    legacy = tmp_path / "legacy.toml"
    legacy.write_text("[run]\ncase = \"real74_d01\"\n"
                      "run_seconds = 60.0\n")
    seen = []
    monkeypatch.setattr(cli, "load_config",
                        lambda path: SimpleNamespace(case="real74_d01"))
    monkeypatch.setitem(
        cli._REAL_CASES, "real74_d01",
        SimpleNamespace(write_static=lambda cfg, output: seen.append(
            ("static", output)) or output))
    assert cli.main(["static", str(legacy),
                     "--output", str(tmp_path / "s.npz")]) == 0
    assert seen and seen[0][0] == "static"


@pytest.mark.parametrize("spelling", ["nan", "inf", "-inf"])
def test_non_finite_root_dx_is_one_refusal_for_every_spelling(
        tmp_path, spelling):
    """E-05's dx arm.  Fraction(nan) raises ValueError and Fraction(inf)
    raises OverflowError -- one branch apart, and only the first is
    caught -- so the same mistake was a sentence or a traceback depending
    on how the reader spelled it.  The finiteness test now precedes the
    conversion, so all three spellings refuse identically."""
    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="").replace(
        "dx = 12000.0", f"dx = {spelling}")
    with pytest.raises(ValueError, match=r"must be a positive, finite"):
        load_experiment(_write(tmp_path, text=text))
