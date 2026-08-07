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


def test_a_coordinate_without_explicit_layers_is_not_guessed_at(
        tmp_path, capsys):
    """An idealized ladder carries no eta interfaces, so there are no
    layer depths to read.  Silence there is a refusal to invent a
    number, not a verdict that the grid is safe."""

    dx = _dx_for_ratio(2.0 * EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT)
    text = textwrap.dedent(_TREE).format(
        nz=_SYNTH_NZ, p_top=_SYNTH_P_TOP, eta=repr(_SYNTH_ETA),
        km_opt=3, mub=repr(MIX_UPPER_BOUND), iso=0, dx=repr(dx))
    text = "\n".join(line for line in text.splitlines()
                     if not line.startswith(("eta_levels", "p_top")))
    path = tmp_path / "exp.toml"
    path.write_text(text)
    load_experiment(path)
    assert capsys.readouterr().err == ""
