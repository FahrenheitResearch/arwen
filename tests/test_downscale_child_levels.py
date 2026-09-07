"""``gpuwm downscale --child-levels``: the door onto the deeper child ladder.

Without a front door the remap is engine-proven and unshipped, so this file
tests the flag, not the operator.
"""

import pytest

from gpuwm.downscale import (
    _derive_child_run_config,
    _render_child_toml,
    build_child_eta_levels,
)


def _parent_config():
    return {
        "nx": 100, "ny": 100, "nz": 49, "dx": 3000.0, "dy": 3000.0,
        "ztop": 20000.0, "dt": 15.0, "run_seconds": 3600.0,
        "hybrid_opt": 2, "etac": 0.2, "moist": True, "mp_physics": 8,
        "terrain_opt": 1, "map_proj": 1,
    }


def test_a_child_level_count_alone_is_refused():
    """A bare count would be filled in with a UNIFORM ladder.

    That is a different atmosphere from a stretched parent's, not a finer
    sampling of it, so the count has to come with a shape.
    """
    with pytest.raises(ValueError) as excinfo:
        build_child_eta_levels(96, stretch=None)
    assert "stretch" in str(excinfo.value)


def test_the_child_ladder_is_a_valid_eta_grid():
    eta = build_child_eta_levels(96, stretch=2.5)
    assert len(eta) == 97
    assert eta[0] == 1.0 and eta[-1] == 0.0
    assert all(b < a for a, b in zip(eta, eta[1:]))


def test_the_child_ladder_flag_reaches_the_derived_config():
    """The rendered child.toml carries the child's nz AND its ladder."""
    eta = build_child_eta_levels(96, stretch=2.5)
    merged = _derive_child_run_config(
        _parent_config(), parent={"dx": 3000.0, "dy": 3000.0}, ratio=3,
        child_nx=120, child_ny=120, run_seconds=1800.0,
        output_interval_s=300.0, child_eta_levels=eta)
    assert merged["nz"] == 96
    assert merged["eta_levels"] == eta

    rendered = _render_child_toml(merged)
    assert "nz = 96" in rendered
    assert "eta_levels = [" in rendered
    # The ladder has to survive the round trip through TOML, or the child
    # would be prepared on one grid and integrated on another.
    import tomllib
    parsed = tomllib.loads(rendered)
    written = tuple(parsed["run"]["eta_levels"])
    assert written == eta


def test_a_child_that_names_no_ladder_renders_no_eta_levels():
    """NEGATIVE CONTROL: the ordinary derived config must not gain a key."""
    merged = _derive_child_run_config(
        _parent_config(), parent={"dx": 3000.0, "dy": 3000.0}, ratio=3,
        child_nx=120, child_ny=120, run_seconds=1800.0,
        output_interval_s=300.0)
    assert "eta_levels" not in merged
    assert "eta_levels" not in _render_child_toml(merged)
    assert merged["nz"] == 49


def test_the_derived_child_ladder_is_refused_when_it_does_not_match_nz():
    with pytest.raises(ValueError):
        _derive_child_run_config(
            _parent_config(), parent={"dx": 3000.0, "dy": 3000.0}, ratio=3,
            child_nx=120, child_ny=120, run_seconds=1800.0,
            output_interval_s=300.0,
            child_eta_levels=build_child_eta_levels(
                96, stretch=2.5)[:-1])


def test_the_cli_exposes_the_flag():
    import argparse

    from gpuwm.downscale import register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "downscale", "parent.nc", "--parent-restart", "r.npz",
        "--out", "o", "--child-levels", "96,2.5"])
    assert args.child_levels == "96,2.5"


# ---------------------------------------------------------------------------
# The two doors the ladder has to reach: the auto-sizer, and the route that
# supplies its own child config.  Both were reachable defects on the lane
# that added the flag (adversarial review 2026-09-03, findings 1 and 2).
# ---------------------------------------------------------------------------


def _peak_and_limit(merged, vram_gib=10.0):
    from datetime import datetime, timezone

    from gpuwm.config import RunConfig
    from gpuwm.core.preflight import (EXTERNAL_MARGIN_BYTES, GIB,
                                      estimate_experiment)
    from gpuwm.domain_wizard import card_assumed_free_gib, fit_headroom_bytes
    from gpuwm.experiment import experiment_from_run_config

    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    exp = experiment_from_run_config(RunConfig(**merged), epoch)
    peak = estimate_experiment(exp, vram_gib=vram_gib).peak_envelope_bytes
    budget = int(card_assumed_free_gib(vram_gib) * GIB) - EXTERNAL_MARGIN_BYTES
    return peak, budget - fit_headroom_bytes(budget)


def test_the_auto_sizer_prices_the_child_on_the_childs_own_ladder():
    """``--point`` without ``--child-size`` must size at the CHILD's nz.

    THE DEFECT THIS PINS.  ``_fit_child_size.fits()`` built the config it
    priced with no ladder, so the search sized the domain at the PARENT's
    level count while the real config was derived 250 lines later WITH the
    ladder.  Measured here on a 342x342 4 km child against a 10 GiB card
    (limit 8.312 GiB): nz=49 prices at 6.554 GiB and fits, nz=128 prices at
    10.980 GiB -- 2.67 GiB over the limit and 0.74 GiB over the whole card.
    The sizer returned 342 and the run then died allocating, after the
    entire parent archive had been read.
    """
    from gpuwm.downscale import _derive_child_run_config, _fit_child_size

    parent_config = _parent_config()
    parent = {"dx": 3000.0, "dy": 3000.0}
    eta = build_child_eta_levels(128, stretch=2.5)

    def derived(size, ladder):
        return _derive_child_run_config(
            parent_config, parent=parent, ratio=3, child_nx=size,
            child_ny=size, run_seconds=1800.0, output_interval_s=300.0,
            child_eta_levels=ladder)

    # The premise: the two ladders really are priced differently, so a
    # sizer that ignores the ladder is not merely inelegant.
    shallow_peak, limit = _peak_and_limit(derived(342, None))
    deep_peak, _ = _peak_and_limit(derived(342, eta))
    assert shallow_peak <= limit < deep_peak

    # The fix: the search prices on the ladder the run will actually use.
    size = _fit_child_size(
        {"nx": 100, "ny": 100, "dx": 3000.0, "dy": 3000.0},
        parent_config, j0=50, i0=50, ratio=3,
        run_seconds=1800.0, output_interval_s=300.0, vram_gib=10.0,
        child_eta_levels=eta)
    sized = derived(size, eta)
    assert sized["nz"] == 128
    peak, limit = _peak_and_limit(sized)
    assert peak <= limit

    # NEGATIVE CONTROL and the defect in one: the size the OLD sizer would
    # have returned -- priced at the parent's 49 levels -- does not fit on
    # the 128-level ladder the run would then have been built with.
    unladdered = _fit_child_size(
        {"nx": 100, "ny": 100, "dx": 3000.0, "dy": 3000.0},
        parent_config, j0=50, i0=50, ratio=3, run_seconds=1800.0,
        output_interval_s=300.0, vram_gib=10.0)
    assert unladdered > size
    peak, limit = _peak_and_limit(derived(unladdered, eta))
    assert peak > limit
    # ...while on the ladder it was priced for, it still fits: the fix
    # moves nothing for a run that never asked for its own levels.
    peak, limit = _peak_and_limit(derived(unladdered, None))
    assert peak <= limit


def test_child_levels_with_a_supplied_child_config_is_refused(tmp_path,
                                                              capsys):
    """``--child-config`` supplies its own ladder; the flag was dropped.

    THE DEFECT THIS PINS, silently:
    ``_parse_child_levels(args.child_levels)`` was read only in the
    ``--point`` branch, so this invocation parsed the flag, stored it, and
    ran the child on the TOML's levels with no word said.  The sibling
    ``--tiles`` refuses exactly this pairing twelve lines above the branch
    that ignored this one, and the commit this lane sits on refuses "a key
    it would silently ignore".
    """
    from gpuwm.cli import main as cli_main
    from test_offline_child_tiles import _child_toml, _parent_archive

    namelist = _parent_archive(tmp_path)
    assert cli_main([
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(_child_toml(tmp_path)), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence", "--child-levels", "128,2.5",
        "--out", str(tmp_path / "child-run"), "--dry-run"]) != 0
    err = capsys.readouterr().err
    assert "--child-levels" in err and "--child-config" in err
    # Names the breakage, per the gate law: which key in the supplied file
    # decides the ladder instead.
    assert "eta_levels" in err


def test_a_child_config_route_without_the_flag_still_runs(tmp_path, capsys):
    """NEGATIVE CONTROL: the refusal must not catch the legal invocation."""
    from gpuwm.cli import main as cli_main
    from test_offline_child_tiles import _child_toml, _parent_archive

    namelist = _parent_archive(tmp_path)
    assert cli_main([
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(_child_toml(tmp_path)), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run"]) == 0
