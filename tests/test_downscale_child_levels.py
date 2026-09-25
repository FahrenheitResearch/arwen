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
    # (The fitter returns the extent with the price it was decided on.)
    size, _ = _fit_child_size(
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
    unladdered, _ = _fit_child_size(
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


def _laddered_child_toml(tmp_path, *, nz, stretch):
    """A supplied child config that already declares its own eta ladder."""
    from test_downscale_cli import _PARENT_CONFIG

    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0,
        child_eta_levels=build_child_eta_levels(nz, stretch=stretch))
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "child.toml"
    path.write_text(_render_child_toml(merged), encoding="utf-8",
                    newline="\n")
    return path


def _downscale_plan(capsys):
    import json

    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):])


def test_child_levels_is_resolved_against_a_supplied_child_config(
        tmp_path, capsys):
    """RETIRES the refusal: the flag is RESOLVED against the file.

    THE DEFECT THIS PINS.  ``--child-levels`` beside ``--child-config`` was
    refused outright -- including against a file that declares no
    ``eta_levels`` at all, where there was nothing to disagree with.  The
    named remedy stranded the ladder generator behind the ``--point``
    route, so a caller who supplies a child config and wants a 128-level
    stretched child had to hand-compute stretched eta interfaces into TOML
    to reach the remap the flag's own help calls THE DOOR.  Now one shared
    function resolves the two, and the plan says which ladder won.
    """
    from gpuwm.cli import main as cli_main
    from test_offline_child_tiles import _child_toml, _parent_archive

    namelist = _parent_archive(tmp_path)
    args = [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run"]

    plain = _child_toml(tmp_path / "plain")
    assert cli_main(args + ["--child-config", str(plain),
                            "--child-levels", "128,2.5"]) == 0
    plan = _downscale_plan(capsys)
    assert plan["child_levels_override"] == "128,2.5"
    assert plan["effective_nz"] == 128
    eta = plan["effective_eta_levels"]
    assert len(eta) == 129
    assert eta[0] == 1.0 and eta[-1] == 0.0
    assert all(b < a for a, b in zip(eta, eta[1:]))
    # The plan AND the price were taken on the flag's ladder, not the
    # file's: a plan that reported 128 while pricing 49 would be two
    # answers about one child.
    assert plan["child_grid"]["nz"] == 128
    assert plan["streaming"]["peak_envelope_bytes"] == (
        plan["memory"]["peak_envelope_bytes"])


def test_child_levels_over_a_declared_ladder_warns_and_wins(
        tmp_path, capsys):
    """A file that already names a ladder is WARNED about, not refused."""
    from gpuwm.cli import main as cli_main
    from test_offline_child_tiles import _parent_archive

    namelist = _parent_archive(tmp_path)
    declared = _laddered_child_toml(tmp_path / "declared", nz=64, stretch=1.5)
    assert cli_main([
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(declared), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence", "--child-levels", "128,2.5",
        "--out", str(tmp_path / "child-run"), "--dry-run"]) == 0
    captured = capsys.readouterr()
    import json

    plan = json.loads(captured.out[captured.out.index("{"):])
    assert plan["effective_nz"] == 128
    overrides = [line for line in captured.err.splitlines()
                 if "--child-levels 128,2.5 replaces the" in line]
    assert len(overrides) == 1
    # Names both ladders, so the reader knows what was overridden.
    assert "64-level" in overrides[0] and "128-level" in overrides[0]


def test_both_sibling_flags_resolve_against_one_supplied_config(
        tmp_path, capsys):
    """THE UMBRELLA: one branch dropped BOTH flags, and one fix answers both.

    ``--tiles`` and ``--child-levels`` were refused by the same
    ``--child-config`` branch, twelve lines apart, for the same stated
    reason.  Against a file that declares neither key there was nothing to
    conflict with, so both are now resolved, and both answers reach the
    plan the reviewer reads.
    """
    from gpuwm.cli import main as cli_main
    from test_offline_child_tiles import _child_toml, _parent_archive

    namelist = _parent_archive(tmp_path)
    plain = _child_toml(tmp_path / "plain")
    assert cli_main([
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(plain), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence", "--tiles", "auto",
        "--child-levels", "128,2.5",
        "--out", str(tmp_path / "child-run"), "--dry-run"]) == 0
    plan = _downscale_plan(capsys)
    assert plan["tiles"]["mode"] == "auto"
    assert plan["child_levels_override"] == "128,2.5"
    assert plan["effective_nz"] == 128
    assert plan["child_grid"]["nz"] == 128


def test_the_two_doors_resolve_the_child_config_with_one_function(tmp_path):
    """One function, both doors: the plan and the run describe one grid."""
    from gpuwm.offline_child import resolve_child_run_config
    from test_offline_child_tiles import _child_toml

    plain = _child_toml(tmp_path / "plain")
    review = resolve_child_run_config(plain, child_levels="128,2.5")
    admission = resolve_child_run_config(plain, child_levels="128,2.5")
    assert review == admission
    assert review.nz == 128
    assert len(review.eta_levels) == 129
    # No flag: the file decides, exactly as before.
    assert resolve_child_run_config(plain).eta_levels is None


def test_one_ladder_disagreement_is_one_warning_at_both_doors(
        tmp_path, capsys):
    """ONE warning, however many doors reach the same resolution.

    The sibling of the ``[tiles]`` case: plan review resolves the ladder
    against the file and the runner's admission resolves it again, so the
    same sentence is reached twice on a real run.  The contract is one
    warning naming both ladders.
    """
    from gpuwm.offline_child import resolve_child_run_config

    declared = _laddered_child_toml(tmp_path / "declared", nz=64, stretch=1.5)
    assert resolve_child_run_config(declared, child_levels="128,2.5").nz == 128
    assert resolve_child_run_config(declared, child_levels="128,2.5").nz == 128
    overrides = [line for line in capsys.readouterr().err.splitlines()
                 if "--child-levels 128,2.5 replaces the" in line]
    assert len(overrides) == 1
    assert "64-level" in overrides[0] and "128-level" in overrides[0]


def test_a_malformed_or_bare_spec_still_refuses_on_the_config_route(
        tmp_path, capsys):
    """NEGATIVE CONTROLS: the two refusals that DO name a breakage stand."""
    from gpuwm.cli import main as cli_main
    from test_offline_child_tiles import _child_toml, _parent_archive

    namelist = _parent_archive(tmp_path)
    plain = _child_toml(tmp_path / "plain")
    args = [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(plain), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run"]

    # A bare N: a uniform ladder under a stretched parent is a different
    # atmosphere, not a finer sampling of it.
    assert cli_main(args + ["--child-levels", "128"]) != 0
    assert "stretch" in capsys.readouterr().err

    # Malformed N[,STRETCH].
    assert cli_main(args + ["--child-levels", "1,2,3"]) != 0
    assert "N[,STRETCH]" in capsys.readouterr().err

    assert not (tmp_path / "child-run").exists()


def _radiative_child_toml(tmp_path):
    """A supplied child config whose domain runs RTE+RRTMGP radiation."""
    from test_downscale_cli import _PARENT_CONFIG

    radiative = dict(_PARENT_CONFIG)
    radiative["ra_lw_physics"] = 4
    radiative["ra_sw_physics"] = 4
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        radiative, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "child.toml"
    path.write_text(_render_child_toml(merged), encoding="utf-8",
                    newline="\n")
    return path


def test_a_ladder_the_childs_radiation_cannot_run_refuses_at_plan_review(
        tmp_path, capsys):
    """THE DEFECT THIS PINS: the rule fired only after the archive was read.

    ``RunConfig`` carries no model-top pressure, so the vertical preflight
    ``validate_run_config`` runs reaches the radiation cap-layer arithmetic
    with ``p_top=None`` and skips it.  The rule that catches it lived in the
    state builder, which runs after the fetch, the SINT, the remap and the
    whole preparation.  That was out of reach while a child's nz was pinned
    to its parent's; a supplied config plus ``--child-levels`` walks
    straight into it, so a 128-level ladder under RTE+RRTMGP planned clean
    on ``--dry-run`` and the run died at the first radiative call.  The
    parent tape knows the model top: plan review reads it there and asks the
    same function, with the same sentence, before anything is opened.
    """
    from gpuwm.cli import main as cli_main
    from test_offline_child_tiles import _parent_archive

    namelist = _parent_archive(tmp_path)
    child = _radiative_child_toml(tmp_path / "radiative")
    args = [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run"]

    assert cli_main(args + ["--child-levels", "128,2.5"]) != 0
    err = capsys.readouterr().err
    assert "exceeds a radiation adapter's layer ceiling" in err
    # Named on the parent's OWN model top, read off the tape, not on a
    # p_top the review invented.
    assert "p_top=10000 Pa" in err
    assert "RTE+RRTMGP longwave" in err
    assert not (tmp_path / "child-run").exists()

    # THE CONTROL: the same config and the same route with a ladder that
    # fits plans clean, so what turned the first one away is the ceiling
    # and not the pairing.
    assert cli_main(args + ["--child-levels", "96,2.5"]) == 0
    plan = _downscale_plan(capsys)
    assert plan["effective_nz"] == 96


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


# --- the boundary strips arrive on the preprocess backend --------------


class _DeviceArrayDouble:
    """What a CuPy array does when NumPy tries to adopt it: refuse.

    ``np.asarray`` on a device array raises ``TypeError: Implicit
    conversion to a NumPy array is not allowed``, and the boundary remap
    reads its strips straight off the preprocess backend, which is CUDA
    by default.  The double is the cheapest instrument that shows the
    difference between a value that was taken to the host and one that
    was not, on a machine with no card.
    """

    def __init__(self, array):
        import numpy as np

        self._array = np.asarray(array)

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not "
                        "allowed. Please use `.get()` to construct a "
                        "NumPy array explicitly.")

    def __getitem__(self, key):
        return _DeviceArrayDouble(self._array[key])

    def get(self):
        return self._array

    @property
    def shape(self):
        return self._array.shape


def test_the_boundary_remap_reads_its_strips_off_the_device(monkeypatch):
    """`--child-levels` on the default backend used to die at exit 1.

    Every field the remap loop reads goes through ``_to_host``; ``mu`` is
    read once before that loop and was read with ``np.asarray``, so the
    flag this whole module is the door for crashed on its first boundary
    frame with CuPy's implicit-conversion TypeError -- on the default
    ``--preprocess-backend cuda``, which is the only backend a downscale
    run selects unless it is told otherwise.
    """
    import numpy as np

    from gpuwm import offline_child

    monkeypatch.setattr(
        offline_child, "_to_host",
        lambda value: np.ascontiguousarray(
            value.get() if hasattr(value, "get") else value,
            dtype=np.float32))

    ny, nx, parent_nz = 4, 5, 3
    child_znw = np.array([1.0, 0.7, 0.4, 0.15, 0.0])
    parent_znw = np.array([1.0, 0.6, 0.25, 0.0])
    mub = np.full((ny, nx), 90000.0)
    phb = np.cumsum(
        np.full((parent_nz + 1, ny, nx), 3000.0), axis=0)
    strips = {
        "mu": _DeviceArrayDouble(np.full((1, ny, nx), 500.0)),
        "theta": _DeviceArrayDouble(
            np.full((parent_nz, ny, nx), 300.0 * 90500.0)),
    }

    out, receipts = offline_child._remap_boundary_snapshot_to_child_ladder(
        strips, child_mub=mub, child_phb=phb,
        parent_znw=parent_znw, child_znw=child_znw,
        hybrid_opt=2, etac=0.2, p_top=5000.0,
        moisture_names=frozenset())

    assert out["mu"] is strips["mu"]          # untouched, and still coupled
    assert np.asarray(out["theta"]).shape == (len(child_znw) - 1, ny, nx)
    assert np.isfinite(np.asarray(out["theta"])).all()
    assert [entry.field for entry in receipts] == ["theta"]
