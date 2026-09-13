"""``[tiles]`` reaches the offline child, from the TOML to the stepper.

THE DEFECT THIS PINS.  The experiment TOML the multi-domain front doors
read has accepted ``[tiles]`` since 2.2.0.  The RunConfig TOML -- the
schema ``gpuwm downscale`` hands to the standalone child -- refused the
block outright as an unknown table, so the one route whose domain is most
likely to outgrow the card it is run on was the one route that could not
ask to stream.  Measured at the 2.2.1 cut: a ``[tiles] mode = 'on'``
appended to a derived child config died in ``load_config`` before any
parent frame was opened.

CPU-side only, deliberately.  The streamed integration itself is proven by
``tilestream/test_join.py`` (bit-exact against the resident arm) and by
the 2.2.2 planner-driven GPU leg; what was missing was never the
transport, it was the wiring, and wiring is what these assert.
"""

from datetime import datetime, timedelta
import json

import pytest

from gpuwm.cli import main as cli_main
from gpuwm.config import load_config, load_streaming_options
from gpuwm.core import streaming
from gpuwm.downscale import _derive_child_run_config, _render_child_toml
from gpuwm.offline_child import OfflineChildContractError
from gpuwm.offline_child_run import _CAPABILITIES
from test_downscale_cli import _PARENT_CONFIG
from test_offline_child import _history


def _child_toml(tmp_path, *, tiles_block: str = "") -> "object":
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "child.toml"
    path.write_text(_render_child_toml(merged) + tiles_block,
                    encoding="utf-8", newline="\n")
    return path


def test_the_child_schema_accepts_a_tiles_table(tmp_path):
    """RED BEFORE THE FIX: ``unknown table(s) ['tiles']``."""

    path = _child_toml(tmp_path, tiles_block='[tiles]\nmode = "on"\n')
    cfg = load_config(path)
    # Accepted, and NOT merged into the RunConfig: [tiles] is an execution
    # choice whose whole claim is that it changes nothing, so it must not
    # reach the fields a restart identity binds.
    assert not hasattr(cfg, "tiles")
    assert (cfg.nx, cfg.ny) == (12, 10)

    options = load_streaming_options(path)
    assert options.mode == "on"
    assert options.enabled


def test_a_child_config_without_the_block_is_the_shared_off_object(tmp_path):
    """The OFF contract: not a parsed default, the same object."""

    options = load_streaming_options(_child_toml(tmp_path))
    assert options is streaming.OFF
    assert not options.enabled


def test_a_misspelled_tiles_key_is_refused_by_the_run_config_reader(tmp_path):
    """A knob that silently does nothing is how a run gets the wrong mode.

    ``load_config`` validates the block and discards it, so a caller that
    reads only the RunConfig is not the reason a typo survives admission.
    """

    path = _child_toml(
        tmp_path, tiles_block='[tiles]\nmode = "on"\ntile_x = 128\n')
    with pytest.raises(ValueError, match="tile_x"):
        load_config(path)


def test_a_tiling_under_mode_off_is_refused(tmp_path):
    """A surface that is off must be empty, or a mode flips somewhere else."""

    path = _child_toml(
        tmp_path, tiles_block='[tiles]\ntile_nx = 128\ntile_ny = 128\n')
    with pytest.raises(ValueError, match="must be empty"):
        load_streaming_options(path)


def test_a_pinned_key_under_mode_auto_is_refused(tmp_path):
    """The AUTO twin of the rule above, at the door a user reaches.

    ``auto``'s documented product IS the planner's tiling, so a key that
    pins one of the planner's own answers beside it is a request the mode
    cannot honour -- and it did not honour it: ``nbuffers = 2`` planned 3
    on the tree this was found on, with nothing warning and the receipt
    recording the request beside the outcome.  Refused where the fix is
    obvious rather than silently overridden where it is not, which is the
    ruling the per-domain budget keys already carry.
    """

    path = _child_toml(
        tmp_path, tiles_block='[tiles]\nmode = "auto"\nnbuffers = 2\n')
    with pytest.raises(ValueError, match="nbuffers"):
        load_streaming_options(path)


def test_off_binds_the_dycore_step_itself(tmp_path):
    """No ``[tiles]`` means no branch at all -- the same function object.

    This is the whole OFF contract, and it is what makes "a child that
    configures nothing is unchanged" a fact rather than a claim.
    """
    from gpuwm.core.dycore import step

    cfg = load_config(_child_toml(tmp_path))
    stepper = streaming.make_stepper(
        None, cfg, load_streaming_options(_child_toml(tmp_path)),
        build=None)
    assert stepper is step


def test_the_standalone_builder_is_the_prepared_builder(tmp_path):
    """A root with no tree still gets the proven builder, not a second one."""

    build = streaming.standalone_domain_builder(grid_id=2)
    assert callable(build)
    # The two facts it answers for a domain that is its own root.
    node = streaming._StandaloneNode(
        cfg=streaming._StandaloneNodeCfg(grid_id=2))
    assert node.parent is None
    assert node.cfg.grid_id == 2


def test_the_runner_declares_that_it_honors_tiles():
    """The capability surface Studio's doctor reads, not a comment."""

    assert _CAPABILITIES["tiles"] == "honored"


# ---------------------------------------------------------------------------
# the front door
# ---------------------------------------------------------------------------


def _parent_archive(tmp_path):
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    return namelist


def _plan(capsys):
    import json

    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):])


def test_the_plan_reports_the_mode_the_child_will_actually_use(
        tmp_path, capsys):
    """Read off the config that will be RUN, not off the flag that was typed.

    The whole front-door claim is that a ``[tiles]`` block in a child config
    is honored, so ``--dry-run`` has to be able to say which mode the run
    will take -- including for a config this command did not write -- and
    now DECIDE it, on the card the review holds, with the same function the
    run decides with (``gpuwm.downscale_pricing``).  The ``tiles`` block
    still reports the configured options; the ``streaming`` block beside
    ``memory`` reports the verdict.
    """

    namelist = _parent_archive(tmp_path)
    args = [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run"]

    plain = _child_toml(tmp_path / "plain")
    assert cli_main(args + ["--child-config", str(plain)]) == 0
    plan = _plan(capsys)
    assert plan["tiles"]["mode"] == "off"
    assert plan["streaming"]["mode"] == "resident"
    assert plan["streaming"]["why"] == "[tiles] mode = 'off'"
    assert plan["streaming"]["tile"] is None
    assert plan["memory"]["basis"] == "explicit-size"
    assert plan["memory"]["peak_envelope_bytes"] == (
        plan["streaming"]["peak_envelope_bytes"])

    auto = _child_toml(tmp_path / "auto",
                       tiles_block='[tiles]\nmode = "auto"\n')
    assert cli_main(args + ["--child-config", str(auto)]) == 0
    plan = _plan(capsys)
    assert plan["tiles"]["mode"] == "auto"
    # Decided against the declared 24 GiB card: the configured envelope
    # fits, so the child is resident, and the budget it was judged on is
    # written down.
    assert plan["streaming"]["mode"] == "resident"
    assert "configured resident envelope fits" in plan["streaming"]["why"]
    assert plan["streaming"]["basis"] == "declared"
    assert plan["streaming"]["budget_bytes"] > (
        plan["streaming"]["peak_envelope_bytes"])
    assert plan["streaming"]["tile"] is None

    # A 12x10 child pinned to mode = "on" cannot be tiled at all (the
    # smallest legal compute window is wider than the domain), and that
    # refusal now lands HERE, at plan review, in the planner's own words,
    # instead of after the whole archive has been interpolated.
    streamed = _child_toml(tmp_path / "streamed",
                           tiles_block='[tiles]\nmode = "on"\n')
    assert cli_main(args + ["--child-config", str(streamed)]) != 0
    err = capsys.readouterr().err
    assert "cannot be tiled at all" in err
    assert "RESIDENT" in err
    assert not (tmp_path / "child-run").exists()


def test_tiles_flag_is_resolved_against_a_supplied_child_config(
        tmp_path, capsys):
    """RETIRES the refusal: the flag is RESOLVED against the file.

    THE DEFECT THIS PINS.  ``--tiles`` beside ``--child-config`` was
    refused outright, including against a file that declares no ``[tiles]``
    table at all, on the grounds that the flag writes into a config this
    command derives.  Nothing was being overwritten in that case: the flag
    named a mode the file was silent about, and ``[tiles]`` binds no
    restart identity, so there was no breakage to name.  Now the two are
    resolved by one function both doors call, and the effective mode is
    what plan review prints and what the price is taken on.
    """

    namelist = _parent_archive(tmp_path)
    args = [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run"]

    # (a) The file is silent about [tiles]; the flag decides, and the plan
    # reports the mode the child will actually integrate under.
    plain = _child_toml(tmp_path / "plain")
    assert cli_main(
        args + ["--child-config", str(plain), "--tiles", "auto"]) == 0
    plan = _plan(capsys)
    assert plan["tiles"]["mode"] == "auto"
    assert plan["streaming"]["mode"] == "resident"
    assert "configured resident envelope fits" in plan["streaming"]["why"]

    # (a2) The SAME pairing with mode = 'on' still refuses -- but in the
    # planner's own words, about this 12x10 domain, not because the flag
    # and the file were typed together.  The flag was resolved and priced;
    # what turned it away is the geometry.
    assert cli_main(
        args + ["--child-config", str(plain), "--tiles", "on"]) != 0
    err = capsys.readouterr().err
    assert "cannot be tiled at all" in err
    assert "RESIDENT" in err

    # (b) The file and the flag agree: a no-op, and no override warning.
    auto = _child_toml(tmp_path / "auto",
                       tiles_block='[tiles]\nmode = "auto"\n')
    assert cli_main(
        args + ["--child-config", str(auto), "--tiles", "auto"]) == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out[captured.out.index("{"):])
    assert plan["tiles"]["mode"] == "auto"
    assert "replaces the [tiles] mode" not in captured.err

    # (c) They disagree: the flag wins as the later and more specific
    # statement, and exactly one warning names both modes.
    streamed = _child_toml(tmp_path / "streamed",
                           tiles_block='[tiles]\nmode = "on"\n')
    assert cli_main(
        args + ["--child-config", str(streamed), "--tiles", "auto"]) == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out[captured.out.index("{"):])
    assert plan["tiles"]["mode"] == "auto"
    overrides = [line for line in captured.err.splitlines()
                 if "replaces the [tiles] mode" in line]
    assert len(overrides) == 1
    assert "'on'" in overrides[0] and "'auto'" in overrides[0]

    # (d) The file spells mode = "off" OUT LOUD and the flag disagrees.
    # THE DEFECT THIS PINS.  The resolver asked ``supplied == OFF`` and an
    # explicit ``[tiles]`` mode = "off" -- a legal, documented spelling --
    # compares EQUAL to the shared OFF object the config authority returns
    # for a file with no ``[tiles]`` table at all.  So the user's own
    # written statement was classified as "the config declares nothing",
    # ``--tiles auto`` replaced it with nothing on stderr, and the durable
    # plan document carried no trace of the override.  Silence is the
    # ABSENCE of the table; an explicit off is a declaration, and
    # disagreeing with it earns the same one warning naming both modes.
    explicit_off = _child_toml(tmp_path / "explicit-off",
                               tiles_block='[tiles]\nmode = "off"\n')
    assert cli_main(
        args + ["--child-config", str(explicit_off), "--tiles", "auto"]) == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out[captured.out.index("{"):])
    assert plan["tiles"]["mode"] == "auto"
    overrides = [line for line in captured.err.splitlines()
                 if "replaces the [tiles] mode" in line]
    assert len(overrides) == 1
    assert "'off'" in overrides[0] and "'auto'" in overrides[0]
    # And the override is in the plan a reviewer reads, not only on a
    # stream nobody keeps.
    assert any("replaces the [tiles] mode" in str(record.get("action", ""))
               for record in plan["warnings"])

    # (e) The file pins a TILING under its own declared mode, and the flag
    # names a mode that cannot carry one.  THE DEFECT THIS PINS.  The
    # resolver swapped the mode field on the object the file had already
    # built (``replace(supplied, mode=mode)``), so the only reader of the
    # result was ``StreamingOptions.__post_init__``, whose sentences are
    # addressed to whoever WROTE the block.  A legal ``mode = "on"`` with
    # a pinned 32x32 tiling became unconstructible the moment ``--tiles
    # auto`` changed the mode: the command printed the override warning
    # saying the flag had won and then, on the very next line, exited 2
    # with "[tiles] sets tile_nx, tile_ny while mode = 'auto' ... say
    # which you meant: mode = 'on' to pin the tiling" -- which is exactly
    # what the file already said.  The FLAG imposed that mode, so no edit
    # to the file cleared it: following the printed remedy changed
    # nothing and the invocation refused identically, with no way out at
    # all.  Now the knobs leave with the mode that could carry them, and
    # the one warning a disagreement earns says which they were and both
    # ways of keeping them.
    pinned = _child_toml(
        tmp_path / "pinned",
        tiles_block=('[tiles]\nmode = "on"\ntile_nx = 32\ntile_ny = 32\n'
                     'nbuffers = 2\n'))
    assert cli_main(
        args + ["--child-config", str(pinned), "--tiles", "auto"]) == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out[captured.out.index("{"):])
    assert plan["tiles"]["mode"] == "auto"
    # ``to_json`` omits what is None, so the pinned keys are gone from the
    # plan the price was taken on instead of sitting beside a mode that
    # would have ignored them.
    assert not {"tile_nx", "tile_ny", "nbuffers"} & set(plan["tiles"])
    overrides = [line for line in captured.err.splitlines()
                 if "replaces the [tiles] mode" in line]
    assert len(overrides) == 1
    assert "'on'" in overrides[0] and "'auto'" in overrides[0]
    # ONE warning, and it names the keys it dropped and BOTH ways of
    # keeping them, so the file is never blamed for a mode the flag
    # imposed and the remedy is never what the file already says.
    assert "tile_nx, tile_ny, nbuffers" in overrides[0]
    assert "Drop --tiles to keep the pinned tiling" in overrides[0]
    assert "delete tile_nx, tile_ny, nbuffers from that file" in overrides[0]
    # And the command does not contradict itself inside one door: nothing
    # refuses after the override has been announced.
    assert "SILENTLY IGNORE" not in captured.err
    assert any("replaces the [tiles] mode" in str(record.get("action", ""))
               for record in plan["warnings"])
    # The file on disk is never rewritten, so the tiling it pins is still
    # there for the invocation that drops the flag.
    assert "tile_nx = 32" in pinned.read_text(encoding="utf-8")


def test_the_two_doors_resolve_tiles_with_one_function(tmp_path):
    """One function, both doors: review and run cannot answer differently.

    ``gpuwm downscale``'s plan review and ``offline_child_run``'s admission
    both call ``resolve_child_streaming_options``, so the namespace the
    wizard hands the engine carries the flag rather than a pre-resolved
    object, and the engine reaches the same answer on its own.
    """

    from gpuwm.offline_child import resolve_child_streaming_options

    auto = _child_toml(tmp_path / "auto",
                       tiles_block='[tiles]\nmode = "auto"\n')
    assert resolve_child_streaming_options(auto, None).mode == "auto"
    assert resolve_child_streaming_options(auto, "auto").mode == "auto"
    assert resolve_child_streaming_options(auto, "on").mode == "on"
    plain = _child_toml(tmp_path / "plain")
    assert resolve_child_streaming_options(plain, None) is streaming.OFF
    assert resolve_child_streaming_options(plain, "on").mode == "on"
    # An explicit mode = "off" is a DECLARATION, not silence: it is the
    # one that compares equal to the shared OFF object without being it,
    # so the resolver has to ask by identity or it overrides a written
    # statement without saying so.
    explicit_off = _child_toml(tmp_path / "explicit-off",
                               tiles_block='[tiles]\nmode = "off"\n')
    supplied = load_streaming_options(explicit_off)
    assert supplied == streaming.OFF and supplied is not streaming.OFF
    assert resolve_child_streaming_options(explicit_off, None).mode == "off"
    assert resolve_child_streaming_options(explicit_off, "auto").mode == "auto"
    # A pinned tiling under the mode that owns it, against a flag mode
    # that cannot carry one: BOTH doors get one object back, with the
    # knobs cleared, instead of the ValueError that reached neither door
    # as an answer it could act on.  The file is untouched, so dropping
    # the flag still runs the tiling it pins.
    pinned = _child_toml(
        tmp_path / "pinned",
        tiles_block='[tiles]\nmode = "on"\ntile_nx = 32\ntile_ny = 32\n')
    resolved = resolve_child_streaming_options(pinned, "auto")
    assert resolved.mode == "auto"
    assert (resolved.tile_nx, resolved.tile_ny) == (None, None)
    kept = resolve_child_streaming_options(pinned, None)
    assert (kept.mode, kept.tile_nx, kept.tile_ny) == ("on", 32, 32)


def test_one_tiles_disagreement_is_one_warning_at_both_doors(
        tmp_path, capsys):
    """ONE warning, however many doors reach the same resolution.

    THE DEFECT THIS PINS.  Both doors resolve the flag against the file --
    that is the rule that keeps plan review and admission from answering
    differently -- so on a real run the identical override sentence is
    reached twice in one process.  Printed twice it contradicts the
    published contract ("a disagreement is one warning naming both
    values") and reads on stderr like two separate overrides of two
    separate statements.  The dry-run tests above could not see it: they
    only ever open the first door.
    """

    from gpuwm.offline_child import resolve_child_streaming_options

    streamed = _child_toml(tmp_path / "streamed",
                           tiles_block='[tiles]\nmode = "on"\n')
    # Exactly what a non-dry-run does: plan review resolves, then the
    # runner's admission resolves the same file against the same flag.
    assert resolve_child_streaming_options(streamed, "auto").mode == "auto"
    assert resolve_child_streaming_options(streamed, "auto").mode == "auto"
    overrides = [line for line in capsys.readouterr().err.splitlines()
                 if "replaces the [tiles] mode" in line]
    assert len(overrides) == 1
    assert "'on'" in overrides[0] and "'auto'" in overrides[0]


def test_a_second_invocation_in_one_process_is_told_again(tmp_path, capsys):
    """ONE warning per INVOCATION, not one per process.

    THE DEFECT THIS PINS.  The dedup the test above asserts is module
    state with no owner, so a caller that drives ``gpuwm downscale`` twice
    in one process over the same file and the same flag was told about the
    override once and then had its SECOND, separate override replace the
    same written statement in silence.  That is the same failure the
    explicit ``mode = "off"`` repair exists to prevent, moved one level up
    from the resolver to the command, and it contradicts the published
    contract ("the warning is said once per invocation however many doors
    resolve the same configuration").  ``downscale_main`` now empties the
    set as the command opens, so the set spans one command and the two
    doors inside it, and no more.
    """

    namelist = _parent_archive(tmp_path)
    child = _child_toml(tmp_path / "explicit-off",
                        tiles_block='[tiles]\nmode = "off"\n')

    def _invoke(outdir):
        return cli_main([
            "downscale", str(tmp_path), "--parent-domain", "3",
            "--parent-namelist", str(namelist), "--ratio", "1",
            "--i-parent-start", "4", "--j-parent-start", "4",
            "--accept-parent-cadence", "--child-config", str(child),
            "--tiles", "auto", "--out", str(outdir), "--dry-run"])

    said = []
    for name in ("first", "second"):
        assert _invoke(tmp_path / name) == 0
        said.append([line for line in capsys.readouterr().err.splitlines()
                     if "replaces the [tiles] mode" in line])
    # Each invocation says it once, and says the same sentence: the second
    # override is a second override, not a repeat of the first.
    assert [len(lines) for lines in said] == [1, 1]
    assert said[0] == said[1]
    assert "'off'" in said[0][0] and "'auto'" in said[0][0]


def test_render_child_toml_emits_a_mode_only_block():
    """Only the mode: a derived config must not carry THIS machine's plan."""

    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    text = _render_child_toml(merged, tiles_mode="auto")
    assert '[tiles]\nmode = "auto"' in text
    for pinned in ("tile_nx", "tile_ny", "nbuffers", "halo"):
        assert pinned not in text
    assert _render_child_toml(merged) .count("[tiles]") == 0
