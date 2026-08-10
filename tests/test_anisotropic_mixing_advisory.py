# tests/test_anisotropic_mixing_advisory.py
"""The closed form of the anisotropic-mixing exposure, and its advisory.

``tests/test_anisotropic_mixing_w_stability.py`` measures the thing on
the card: with ``mix_isotropic = 0`` the horizontal diffusion of ``w`` is
handed the VERTICAL exchange coefficient, which is built and capped on
the layer depth, and past ``K*dt/dx^2 = 1/4`` an explicit Laplacian
amplifies the 2-grid-interval mode instead of damping it.

Here is the same statement in closed form --
``mix_upper_bound * (dz_max/dx)^2`` against 1/4, a ratio ``dt`` cancels
out of -- and the load-time warning built on it.  CPU only, and
deliberately in its own file: the measurement's helper opens a CUDA
device, which marks its whole module ``gpu``, and a criterion this cheap
should still be checked on a machine that has no card.

Generic throughout: the criterion is grid geometry, not a case.
"""

from __future__ import annotations

import math
import textwrap

import pytest

from gpuwm.config import (EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT,
                          anisotropic_w_mixing_ratio)
from gpuwm.core.grid import base_layer_depths
from gpuwm.experiment import load_experiment

#: The grid the GPU measurement runs on, so the two files are talking
#: about one configuration and the closed form can be checked against
#: what the card reported.
DX = 100.0
DZ = 537.0
MIX_UPPER_BOUND = 0.1


def test_the_criterion_flags_the_grid_the_measurement_fails_on():
    ratio = anisotropic_w_mixing_ratio(
        km_opt=3, mix_isotropic=0, mix_upper_bound=MIX_UPPER_BOUND,
        dx=DX, dy=DX, dz_max=DZ)
    assert ratio == pytest.approx(MIX_UPPER_BOUND * (DZ / DX) ** 2)
    assert ratio > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT


def test_the_criterion_is_not_applicable_off_the_exposed_path():
    """``None`` means "no per-axis coefficient exists here", which is a
    different answer from "it exists and is small"."""

    common = dict(mix_upper_bound=MIX_UPPER_BOUND, dx=DX, dy=DX, dz_max=DZ)
    assert anisotropic_w_mixing_ratio(
        km_opt=3, mix_isotropic=1, **common) is None
    for km_opt in (0, 1, 4):
        assert anisotropic_w_mixing_ratio(
            km_opt=km_opt, mix_isotropic=0, **common) is None
    # km_opt = 2 builds the same per-axis lengths, from prognostic TKE.
    assert anisotropic_w_mixing_ratio(
        km_opt=2, mix_isotropic=0, **common) is not None


def test_the_criterion_uses_the_shorter_horizontal_side():
    """The stencil is limited by whichever axis it differences over
    fastest, so a stretched cell is judged on its short side."""

    assert anisotropic_w_mixing_ratio(
        km_opt=3, mix_isotropic=0, mix_upper_bound=MIX_UPPER_BOUND,
        dx=4.0 * DX, dy=DX, dz_max=DZ) == pytest.approx(
            MIX_UPPER_BOUND * (DZ / DX) ** 2)


#: A synthetic single-domain tree carrying an explicit eta ladder, so the
#: loader has real layer depths to read.  Radiation off: the advisory is
#: about mixing, and this shallow lid would otherwise trip the radiation
#: level-count preflight first.
_TREE = """\
[experiment]
name = "synth"
start_time = 1974-04-03T12:00:00
run_seconds = 600.0
restart_interval_s = 0.0

[shared]
nz = {nz}
ztop = 20000.0
p_top = {p_top}
eta_levels = {eta}
km_opt = {km_opt}
ra_lw_physics = 0
ra_sw_physics = 0
bl_pbl_physics = 0
mix_upper_bound = {mub}
mix_isotropic = {iso}

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 40
ny = 40
time_step = 2
dx = {dx}
history_interval_s = 600.0
"""

_SYNTH_NZ = 8
_SYNTH_P_TOP = 50000.0
_SYNTH_ETA = [1.0 - i / _SYNTH_NZ for i in range(_SYNTH_NZ)] + [0.0]


def _synth_dz_max() -> float:
    """The layer depths the loader itself will read off this ladder."""

    return float(base_layer_depths(_SYNTH_ETA, 0, 0.2, _SYNTH_P_TOP).max())


def _dx_for_ratio(ratio: float) -> float:
    """The dx that puts this ladder at exactly ``ratio``.

    Deriving dx from the criterion rather than hard-coding one keeps the
    fixtures on the correct side of the limit even if the ladder or the
    base state is ever rebuilt.
    """

    return _synth_dz_max() * math.sqrt(MIX_UPPER_BOUND / ratio)


def _write_tree(tmp_path, *, dx: float, iso: int = 0, km_opt: int = 3,
                mub: float = MIX_UPPER_BOUND):
    path = tmp_path / "exp.toml"
    path.write_text(textwrap.dedent(_TREE).format(
        nz=_SYNTH_NZ, p_top=_SYNTH_P_TOP, eta=repr(_SYNTH_ETA),
        km_opt=km_opt, mub=repr(mub), iso=iso, dx=repr(dx)))
    return path


def test_the_loader_warns_when_the_ratio_exceeds_the_limit(tmp_path, capsys):
    over = 2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT
    load_experiment(_write_tree(tmp_path, dx=_dx_for_ratio(over)))
    err = capsys.readouterr().err
    assert "warning:" in err
    # The computed number, the criterion it was compared against, and the
    # depths it was computed from, so a reader can check the arithmetic
    # without re-deriving it.
    assert "mix_upper_bound*(dz_max/dx)^2" in err
    assert f"{over:.3g}" in err
    assert str(EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT) in err
    assert f"{_synth_dz_max():.1f} m deep" in err
    assert "mix_isotropic = 1" in err


def test_the_loader_is_silent_below_the_limit(tmp_path, capsys):
    under = 0.5 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT
    load_experiment(_write_tree(tmp_path, dx=_dx_for_ratio(under)))
    assert capsys.readouterr().err == ""


def test_an_isotropic_length_silences_it_at_the_same_grid(tmp_path, capsys):
    """The remedy the warning names has to be the remedy that works: same
    ladder, same dx, one switch."""

    dx = _dx_for_ratio(2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT)
    load_experiment(_write_tree(tmp_path, dx=dx, iso=1))
    assert capsys.readouterr().err == ""


def test_a_lower_cap_silences_it_at_the_same_grid(tmp_path, capsys):
    """So does the other remedy it names, which is what makes the number
    it prints actionable rather than decorative."""

    dx = _dx_for_ratio(2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT)
    load_experiment(_write_tree(tmp_path, dx=dx, mub=0.4 * MIX_UPPER_BOUND))
    assert capsys.readouterr().err == ""


def test_a_horizontal_only_selector_is_never_advised_about(tmp_path, capsys):
    """km_opt = 4 builds one coefficient on the horizontal spacing, so the
    exposure does not exist and neither should the warning."""

    dx = _dx_for_ratio(2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT)
    load_experiment(_write_tree(tmp_path, dx=dx, km_opt=4))
    assert capsys.readouterr().err == ""


def test_the_sentence_separates_the_inverting_tier_from_the_growing_one():
    """Two thresholds, two statements.

    A reader handed one number against one limit cannot tell 0.3 from
    4.2, and only one of those is the tier that aborted a run.
    """

    from gpuwm.config import (EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT,
                              anisotropic_w_mixing_advice)

    common = dict(where="d03", km_opt=3, mix_isotropic=0, dx=DX, dy=DX,
                  dz_max=DZ)
    # mix_upper_bound chosen so the ratio lands in each tier exactly.
    inverting = 0.35 / (DZ / DX) ** 2
    growing = 4.0 / (DZ / DX) ** 2

    ratio, advice = anisotropic_w_mixing_advice(
        mix_upper_bound=inverting, **common)
    assert ratio == pytest.approx(0.35)
    assert "INVERTS" in advice and "AMPLIFIES" not in advice
    assert str(EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT) in advice

    ratio, advice = anisotropic_w_mixing_advice(
        mix_upper_bound=growing, **common)
    assert ratio == pytest.approx(4.0)
    assert "AMPLIFIES" in advice and "INVERTS" not in advice

    # Both tiers still name the criterion, the remedy and the posture.
    for mub in (inverting, growing):
        _, advice = anisotropic_w_mixing_advice(mix_upper_bound=mub, **common)
        assert "mix_isotropic = 1" in advice
        assert "ADVISORY, not a refusal" in advice
        assert str(EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT) in advice


def test_the_sentence_is_absent_below_the_limit_and_off_the_path():
    """The negative control on the wording, not just on the warning."""

    from gpuwm.config import anisotropic_w_mixing_advice

    common = dict(where="d03", mix_upper_bound=MIX_UPPER_BOUND, dx=DX,
                  dy=DX, dz_max=DZ)
    under = 0.1 / (DZ / DX) ** 2
    assert anisotropic_w_mixing_advice(
        km_opt=3, mix_isotropic=0,
        **{**common, "mix_upper_bound": under})[1] is None
    assert anisotropic_w_mixing_advice(
        km_opt=3, mix_isotropic=1, **common) == (None, None)
    assert anisotropic_w_mixing_advice(
        km_opt=4, mix_isotropic=0, **common) == (None, None)


def test_gpuwm_check_repeats_the_advisory_in_its_report(tmp_path):
    """The load-time line is emitted hours before the run that pays for
    it; the preflight report is the door a reader actually opens."""

    from gpuwm.core.preflight import check_advisories

    exp = load_experiment(_write_tree(
        tmp_path, dx=_dx_for_ratio(2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT)))
    advisories = check_advisories(exp)
    mixing = [line for line in advisories
              if "mix_upper_bound*(dz_max/dx)^2" in line]
    assert len(mixing) == 1
    assert mixing[0].startswith("d01 ")
    assert "mix_isotropic = 1" in mixing[0]


def test_gpuwm_check_is_silent_on_the_same_grid_with_one_mixing_length(
        tmp_path):
    """The negative control the report needs: an advisory that fires on
    a legitimate configuration is noise, and noise is what got the
    original warning ignored."""

    from gpuwm.core.preflight import check_advisories

    dx = _dx_for_ratio(2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT)
    exp = load_experiment(_write_tree(tmp_path, dx=dx, iso=1))
    assert [line for line in check_advisories(exp)
            if "mix_upper_bound" in line] == []


# --- the ladder a config declined to write out -----------------------------
#
# The criterion needs layer depths, and until 2026-08-09 a config that
# carried no `eta_levels` was skipped: the exposure came back empty, the
# loader said nothing, `gpuwm check` said nothing, and the repository
# screen counted it as a pass.  km_opt = 3 with mix_isotropic = 0 at
# dx = 100 m is the exact configuration the criterion exists for, so a
# route that cannot see it is a hole in the middle of the check.  The
# depths are now RESOLVED the way the model resolves them.

#: The same tree, minus the two keys that declare a ladder.  Everything
#: else is byte-for-byte `_TREE`, so a difference in behaviour is a
#: difference in the ladder and nothing else.
_TREE_NO_LADDER = "\n".join(
    line for line in _TREE.splitlines()
    if not line.startswith(("p_top =", "eta_levels ="))) + "\n"


def _resolved_dz_max(nz: int = _SYNTH_NZ, ztop: float = 20000.0) -> float:
    """What the resolution should produce, derived independently here."""

    from gpuwm.core.grid import (analytic_base_pressure, base_layer_depths,
                                 make_vertical_coord)

    return float(base_layer_depths(
        make_vertical_coord(nz).znw, 0, 0.2,
        analytic_base_pressure(ztop)).max())


def _write_tree_without_a_ladder(tmp_path, *, dx: float, iso: int = 0,
                                 km_opt: int = 3,
                                 mub: float = MIX_UPPER_BOUND):
    path = tmp_path / "exp_no_ladder.toml"
    path.write_text(textwrap.dedent(_TREE_NO_LADDER).format(
        nz=_SYNTH_NZ, p_top="", eta="", km_opt=km_opt, mub=repr(mub),
        iso=iso, dx=repr(dx)))
    return path


def test_the_resolved_ladder_spans_exactly_the_declared_model_top():
    """The resolution is not a guess with a plausible shape.

    ``analytic_base_pressure`` is the exact inverse of the relation
    ``base_layer_depths`` uses to turn pressure back into height, so the
    resolved column's layers sum to ``ztop`` to FP64.  Without that
    property the depths would be an invented profile, and a criterion
    fed invented depths is worse than one that stays quiet.
    """

    from gpuwm.core.grid import (analytic_base_pressure, base_layer_depths,
                                 make_vertical_coord)

    for nz, ztop in ((8, 20000.0), (40, 20000.0), (60, 16000.0)):
        depths = base_layer_depths(make_vertical_coord(nz).znw, 0, 0.2,
                                   analytic_base_pressure(ztop))
        assert len(depths) == nz
        assert float(depths.sum()) == pytest.approx(ztop, rel=1e-12)


def test_a_model_top_above_the_analytic_ceiling_has_no_pressure():
    """And is reported as such rather than as a number.

    The 50 K-lapse base state runs out of atmosphere near 24.6 km; a
    ``p_top`` fabricated past that would put a fictitious depth into a
    stability criterion.
    """

    from gpuwm.core.grid import analytic_base_pressure

    assert analytic_base_pressure(24000.0) is not None
    assert analytic_base_pressure(30000.0) is None


def test_the_loader_warns_on_a_tree_that_declares_no_eta_ladder(
        tmp_path, capsys):
    """The reproducing case: km_opt = 3, mix_isotropic = 0, dx = 100 m,
    no ``eta_levels``.  It used to load in silence."""

    load_experiment(_write_tree_without_a_ladder(tmp_path, dx=DX))
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "mix_upper_bound*(dz_max/dx)^2" in err
    assert f"{_resolved_dz_max():.1f} m deep" in err
    # And it says the depths were resolved, not declared: a derived
    # number that reads like a measured one is its own defect.
    assert "resolved ladder" in err
    assert "declares no eta_levels" in err


def test_the_resolved_route_still_has_a_below_limit_side(tmp_path, capsys):
    """Resolution is not "warn about everything without a ladder".

    Same tree, same missing ladder, a spacing wide enough that the
    criterion holds -- and the loader is silent.  Without this the new
    branch could be a blanket warning and every test above would still
    pass.
    """

    under = 0.5 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT
    dx = _resolved_dz_max() * math.sqrt(MIX_UPPER_BOUND / under)
    load_experiment(_write_tree_without_a_ladder(tmp_path, dx=dx))
    assert capsys.readouterr().err == ""


def test_the_resolved_route_keeps_both_off_path_controls(tmp_path, capsys):
    """km_opt = 4 and mix_isotropic = 1 stay silent without a ladder too,
    which is what keeps the idealized/legacy tree from lighting up."""

    load_experiment(_write_tree_without_a_ladder(tmp_path, dx=DX, km_opt=4))
    assert capsys.readouterr().err == ""
    load_experiment(_write_tree_without_a_ladder(tmp_path, dx=DX, iso=1))
    assert capsys.readouterr().err == ""


def test_gpuwm_check_repeats_the_resolved_case_too(tmp_path):
    """The preflight door gets the same sentence, provenance included."""

    from gpuwm.core.preflight import check_advisories

    exp = load_experiment(_write_tree_without_a_ladder(tmp_path, dx=DX))
    mixing = [line for line in check_advisories(exp)
              if "mix_upper_bound*(dz_max/dx)^2" in line]
    assert len(mixing) == 1
    assert mixing[0].startswith("d01 ")
    assert "resolved ladder" in mixing[0]


def test_the_declared_ladder_carries_no_provenance_clause(tmp_path, capsys):
    """The clause has to distinguish, so it must be absent when the
    config wrote its own interfaces."""

    load_experiment(_write_tree(
        tmp_path, dx=_dx_for_ratio(2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT)))
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "resolved ladder" not in err


def test_a_coordinate_without_explicit_layers_is_resolved_not_skipped(
        tmp_path, capsys):
    """REPLACES ``..._is_not_guessed_at``, which pinned the opposite.

    1.6.0 ruled that a coordinate with no eta interfaces should be
    passed over -- "silence is a refusal to invent a number, not a
    verdict that the grid is safe" -- and that reasoning is sound about
    INVENTING one.  It is wrong about this case, because the number does
    not have to be invented: ``nz`` and ``ztop`` are required keys, the
    interfaces are the uniform ladder the model itself builds from them,
    and the model top converts to a pressure through the exact inverse
    of the relation the depths are read with.  Nothing is guessed.

    What the old ruling cost is what this test now pins.  Downstream,
    silence is indistinguishable from a pass: the load-time warning, the
    ``gpuwm check`` advisory and the repository-wide screen in
    ``tests/test_shipped_configs_mixing_stability.py`` all went quiet on
    a config that selects the exposed path -- and the screen reported
    it, in writing, as clean.
    """

    dx = _dx_for_ratio(2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT)
    text = textwrap.dedent(_TREE).format(
        nz=_SYNTH_NZ, p_top=_SYNTH_P_TOP, eta=repr(_SYNTH_ETA),
        km_opt=3, mub=repr(MIX_UPPER_BOUND), iso=0, dx=repr(dx))
    text = "\n".join(line for line in text.splitlines()
                     if not line.startswith(("eta_levels", "p_top")))
    path = tmp_path / "exp.toml"
    path.write_text(text)
    load_experiment(path)
    err = capsys.readouterr().err
    assert "warning:" in err
    # The resolved ladder is DEEPER than the declared one this dx was
    # derived against, so the ratio rises rather than merely appearing:
    # same file, same selectors, and the criterion now has depths.
    assert f"{_resolved_dz_max():.1f} m deep" in err
    assert _resolved_dz_max() > _synth_dz_max()
    assert "resolved ladder" in err


# --- and the ladder that will not resolve even then ------------------------
#
# The third outcome, added 2026-08-09.  Resolving the interfaces from
# `nz`/`ztop` needs the model top expressed as a pressure, and the
# analytic base state runs out of atmosphere near 24.6 km; above that
# `analytic_base_pressure` correctly declines to invent one and the
# exposure hands back `dz_max = inf`.  That sentinel then went into
# `anisotropic_w_mixing_ratio`, which returns `None` for a non-finite
# depth -- the SAME value it returns for a config that is not on the
# exposed path at all -- and every door read it as "nothing to say".
#
# It is the opposite of nothing to say: the domain selected the per-axis
# mixing length and the criterion has no number for it.  There is still
# no number to invent, so what fires is a different sentence.

#: A model top past that ceiling.  Kept as a name because three tests and
#: the repository screen all need the same shape.
_ABOVE_THE_ANALYTIC_CEILING = 26000.0


def _write_tree_above_the_ceiling(tmp_path, *, dx: float = DX, iso: int = 0,
                                  km_opt: int = 3):
    """The no-ladder tree with its model top past the analytic ceiling.

    Built by raising ``ztop`` on ``_TREE_NO_LADDER`` and nothing else, so
    a difference in behaviour against
    :func:`test_the_loader_warns_on_a_tree_that_declares_no_eta_ladder`
    is the model top and nothing else.
    """

    path = _write_tree_without_a_ladder(tmp_path, dx=dx, iso=iso,
                                        km_opt=km_opt)
    path.write_text(path.read_text(encoding="utf-8").replace(
        "ztop = 20000.0", f"ztop = {_ABOVE_THE_ANALYTIC_CEILING!r}"),
        encoding="utf-8")
    return path


def test_an_unresolvable_depth_advises_instead_of_returning_nothing():
    """The unit statement: no ratio, and a sentence saying why.

    ``None`` for the ratio is right -- there is no number and fabricating
    one would put a fictitious depth into a stability criterion.  ``None``
    for the ADVICE was the defect: it made the sentinel indistinguishable
    from "not applicable" at every door downstream.
    """

    from gpuwm.config import (UNRESOLVED_ANISOTROPIC_DEPTH_MARK,
                              anisotropic_w_mixing_advice,
                              selects_anisotropic_w_mixing)

    common = dict(where="d03", mix_upper_bound=MIX_UPPER_BOUND, dx=DX, dy=DX,
                  dz_max=math.inf)
    ratio, advice = anisotropic_w_mixing_advice(
        km_opt=3, mix_isotropic=0, ladder="the ladder story", **common)
    assert ratio is None
    assert UNRESOLVED_ANISOTROPIC_DEPTH_MARK in advice
    assert "the ladder story" in advice
    # It names both remedies -- and NOT the one that cannot help, since
    # lowering a cap on an unknown quantity is not an action.
    assert "eta_levels" in advice
    assert "mix_isotropic = 1" in advice
    assert "lower mix_upper_bound" not in advice
    # And it stays an advisory, and reports no threshold verdict it has
    # no number to reach.
    assert "ADVISORY, not a refusal" in advice
    assert "exceeds the explicit horizontal diffusion limit" not in advice
    assert "INVERTS" not in advice and "AMPLIFIES" not in advice

    # km_opt = 2 selects the same per-axis lengths, so it advises too.
    assert anisotropic_w_mixing_advice(
        km_opt=2, mix_isotropic=0, **common)[1] is not None

    # Off the path the missing depth is genuinely nothing to say: the
    # per-axis coefficient the criterion is about does not exist.
    for off_path in (dict(km_opt=3, mix_isotropic=1),
                     dict(km_opt=4, mix_isotropic=0),
                     dict(km_opt=1, mix_isotropic=0)):
        assert anisotropic_w_mixing_advice(**off_path, **common) == (None,
                                                                     None)
        assert not selects_anisotropic_w_mixing(dx=DX, dy=DX, **off_path)
    assert selects_anisotropic_w_mixing(km_opt=3, mix_isotropic=0, dx=DX,
                                        dy=DX)


def test_the_loader_says_when_no_depth_could_be_resolved(tmp_path, capsys):
    """Door one: the config load.  It used to say nothing at all.

    Advisory, like every other arm of this criterion: the load returns a
    tree and the run proceeds.
    """

    from gpuwm.config import UNRESOLVED_ANISOTROPIC_DEPTH_MARK

    experiment = load_experiment(_write_tree_above_the_ceiling(tmp_path))
    assert len(experiment.domains) == 1
    err = capsys.readouterr().err
    assert "warning:" in err
    assert UNRESOLVED_ANISOTROPIC_DEPTH_MARK in err
    # The provenance says WHY it could not be resolved, not merely that
    # it could not: a reader has to know which key to reach for.
    assert "above the analytic base state's representable ceiling" in err
    assert "mix_isotropic = 1" in err
    # No number was invented on the way past.
    assert "mix_upper_bound*(dz_max/dx)^2 = " not in err


def test_gpuwm_check_repeats_the_unresolvable_case_too(tmp_path):
    """Door two: the preflight report, which is where a reader looks."""

    from gpuwm.config import UNRESOLVED_ANISOTROPIC_DEPTH_MARK
    from gpuwm.core.preflight import check_advisories

    exp = load_experiment(_write_tree_above_the_ceiling(tmp_path))
    mixing = [line for line in check_advisories(exp)
              if UNRESOLVED_ANISOTROPIC_DEPTH_MARK in line]
    assert len(mixing) == 1
    assert mixing[0].startswith("d01 ")
    assert "representable ceiling" in mixing[0]


def test_the_unresolvable_top_keeps_both_off_path_controls(tmp_path, capsys):
    """Both doors stay silent off the path at the same model top.

    Without this the new branch could be "warn about every tall lid" and
    every assertion above would still pass.
    """

    from gpuwm.core.preflight import check_advisories

    for kwargs in (dict(km_opt=4), dict(iso=1)):
        exp = load_experiment(_write_tree_above_the_ceiling(
            tmp_path, **kwargs))
        assert capsys.readouterr().err == ""
        assert [line for line in check_advisories(exp)
                if "mix_upper_bound" in line] == []
