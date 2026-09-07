"""The memory ledger, audited against streaming -- with the controls.

Every capability here is checked twice: once for the answer, once for a
control that fails if the thing under test were switched off or quietly
broken.  A refusal that fires on everything proves nothing, and an equality
that would hold no matter what the code did proves less.

===============================  =========================================
capability                       the control that would catch it being fake
===============================  =========================================
the estimate is mode-blind       priced with mode on and off; the bytes
                                 must be EQUAL, so a later "streaming-aware
                                 estimate" cannot land unnoticed
the drift guard is mode-blind    the three guard quantities, priced both
                                 ways, must be equal -- the guard cannot
                                 fire *because of* streaming, and equally
                                 cannot protect a streamed run
the refusal is at admission      ``refuse_unrouted_streaming`` on a route
                                 with NO builder must raise for ``'on'``;
                                 ``'off'`` and ``'auto'`` must not, or the
                                 refusal is a blanket one and proves nothing
the routes wire the builders     ``gpuwm run`` and the prepared tree route
                                 both stream instead of refusing (read out
                                 of their source, both ends of the seam),
                                 and ``gpuwm go`` admits ``mode='on'``
                                 pre-fetch while still refusing an invalid
                                 [tiles] table on the same side of the fetch
the receipt names the mode       ``off`` must produce an EMPTY entry (the
                                 pre-streaming receipt, byte for byte), and
                                 ``on`` must produce a non-empty one
the two VRAM models              old empirical resident fit cannot strand
                                 an auto run whose configured resident
                                 envelope requires the fitting tile road
===============================  =========================================

CPU only.  Nothing here needs a card; every quantity is shape arithmetic.
The one leg that DOES need a card -- allocating the shared workspaces and
running the drift guard's three comparisons against them, with a negative
control that prices one tree and allocates for another -- lives in
``python -m tilestream.ledger_probe --guard``, because a comparison of two
numbers neither of which touched a device proves the arithmetic and nothing
else.  MEASURED there on an RTX 5090, real74 four-domain tree: arena
3.0876 = 3.0876 GiB, rebuilt 1.9198 = 1.9198, rrtmgp 1.9105 = 1.9105,
resident AND streamed; the negative control (d04 32 cells wider than the
tree that was priced) fired on two of the three legs.

Run it::

    python -m tilestream.test_ledger_gate
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime
from pathlib import Path

from gpuwm.config import load_config
from gpuwm.core import preflight as pf
from gpuwm.core import streaming
from gpuwm.experiment import experiment_from_run_config

ROOT = Path(__file__).resolve().parents[1]
GIB = pf.GIB


def _exp(n: int = 448, mode: str = "off", **stream_kwargs):
    """``configs/real74_d01.toml`` at ``n x n``, with a ``[tiles]`` mode."""
    cfg = dataclasses.replace(
        load_config(ROOT / "configs" / "real74_d01.toml"), nx=n, ny=n)
    exp = experiment_from_run_config(cfg, datetime(1974, 4, 3, 12))
    if mode != "off":
        exp = dataclasses.replace(
            exp, tiles=streaming.StreamingOptions(
                mode=mode, **stream_kwargs))
    return exp


# --------------------------------------------------------------------------
# the estimate, and the drift guard, are both blind to [tiles]
# --------------------------------------------------------------------------

def test_the_preflight_estimate_is_identical_with_streaming_on():
    """``estimate_experiment`` never reads ``exp.tiles``.

    Stated as an equality rather than as prose because the interesting
    consequence runs both ways.  It means the ledger's drift guard cannot
    fire BECAUSE a domain was streamed -- the risk this audit was opened on,
    and it is not real.  It also means the estimate a route refuses on
    prices a resident allocation a streamed run will never make, which is
    the risk that IS real and which this equality is the proof of.
    """
    off = pf.estimate_experiment(_exp(448, "off"))
    on = pf.estimate_experiment(
        _exp(448, "on", tile_nx=224, tile_ny=224, nbuffers=2))
    assert off.alloc_estimate_bytes == on.alloc_estimate_bytes
    assert off.peak_envelope_bytes == on.peak_envelope_bytes
    assert off.resident_bytes == on.resident_bytes
    return (f"estimate identical both ways: "
            f"{off.alloc_estimate_bytes / GIB:.3f} GiB alloc, "
            f"{off.peak_envelope_bytes / GIB:.3f} GiB envelope")


def test_the_drift_guard_quantities_are_identical_with_streaming_on():
    """The three quantities ``prepared_domain_tree_forecast`` compares.

    ``arena.nbytes`` vs ``estimate.scratch_arena_bytes``, the rebuilt-state
    workspace, and the shared RRTMGP chunk workspace.  Both sides of every
    comparison are computed from ``exp.domains`` by the SAME function, so
    the guard is a self-consistency check on the shape registry and nothing
    else.  A streamed domain changes neither side.
    """
    exp_off = _exp(448, "off")
    exp_on = _exp(448, "on", tile_nx=224, tile_ny=224, nbuffers=2)
    for a, b in ((exp_off, exp_on),):
        assert (pf.shared_scratch_arena_bytes(a.domains)
                == pf.shared_scratch_arena_bytes(b.domains))
        assert (pf.shared_dycore_state_workspace_bytes(a.domains)
                == pf.shared_dycore_state_workspace_bytes(b.domains))
    est_off = pf.estimate_experiment(exp_off)
    est_on = pf.estimate_experiment(exp_on)
    assert est_off.workspace_bytes == est_on.workspace_bytes
    return ("guard quantities identical both ways; the guard cannot fire "
            "because of streaming, and cannot protect a streamed run either")


# --------------------------------------------------------------------------
# the admission refusal, and its controls
# --------------------------------------------------------------------------

def test_mode_on_is_refused_at_admission():
    try:
        streaming.refuse_unrouted_streaming(_exp(448, "on"), "test route")
    except streaming.StreamingRefused as error:
        assert "wires no streamed-domain builder" in str(error), str(error)
        return "mode='on' refused at admission, and the message names why"
    raise AssertionError("mode='on' was admitted by a route with no builder")


def test_mode_off_and_auto_are_not_refused():
    """THE CONTROL.  A refusal that fires on everything proves nothing.

    ``off`` is the default every experiment carries, and ``auto`` on a
    route that CONSULTS the seam is a legitimate resident run wherever the
    domain fits; refusing either would make the check above vacuous.
    """
    streaming.refuse_unrouted_streaming(_exp(448, "off"), "test route")
    streaming.refuse_unrouted_streaming(_exp(448, "auto"), "test route")
    streaming.refuse_unrouted_streaming(object(), "test route")
    return "off, auto and a streaming-less object are all admitted"


def test_the_run_route_streams_instead_of_refusing():
    """``gpuwm run`` reads ``[tiles]`` now, so the relay must not refuse it.

    THE REFUSAL THIS REPLACES WAS TRUE WHEN IT WAS WRITTEN.
    ``runtime.run_experiment`` called ``integrate_prepared_case`` with
    ``stepper=None`` on its single-domain arm and ``execute_experiment`` /
    ``walk_spawn_legs`` with no ``steppers=`` on both tree arms, so a
    configured mode was read, validated, echoed into the resolved-config
    report and then dropped -- a resident run with nothing said.  The
    remedy for that was never a permanent refusal; it was the wiring, and
    the refusal's own message named it (``streaming.builders_for_tree``).

    Both arms are wired.  What is checked here is that the two readers of
    that fact agree: the route itself, and run-plan's delivery table,
    which relays the core's sentence at resolve time and would otherwise
    go on refusing -- before the run directory -- a config the route runs.

    ``refuse_unrouted_streaming`` is NOT deleted and is still exercised:
    it is the right answer for any route that genuinely reads nothing,
    and ``test_mode_on_is_refused_at_admission`` above holds its words.
    """
    import inspect

    from gpuwm import runplan, runtime

    src = inspect.getsource(runtime.run_experiment)
    assert 'refuse_unrouted_streaming(exp, "gpuwm run"' not in src, (
        "gpuwm run still refuses [tiles] at its front door while wiring "
        "the builders behind it")
    # The tree arm now delegates after retaining its cold Machine. Check
    # both ends of that call, rather than requiring allocation wiring to
    # remain textually inside the public runner.
    assert "return _run_built_experiment(" in src
    assert "planning_machine=planning_machine" in src
    tree_src = inspect.getsource(runtime._run_built_experiment)
    assert "builders=_streaming.builders_for_tree(model, exp.tiles)" in tree_src
    assert "steppers = _streaming.steppers_for_tree(" in tree_src
    assert "machine=planning_machine" in tree_src
    assert 'resident_estimate=getattr(model.memory_ledger, "estimate", None)' in tree_src
    # The fixed single-domain arm still binds its standalone builder.
    assert "standalone_domain_builder(" in src
    assert "stepper=single_stepper" in src
    # And the relay agrees.  "unrouted" here would refuse at resolve time.
    assert runplan._STREAMING_DELIVERY["experiment"] == "tree", (
        runplan._STREAMING_DELIVERY)
    decision = runplan.streaming_decision(_exp(448, "auto"),
                                          chain="experiment")
    assert decision["refusal"] is None, decision["refusal"]
    assert decision["delivery"] == "tree", decision
    return ("gpuwm run wires builders_for_tree and standalone_domain_builder "
            "instead of refusing, and run-plan's 'experiment' chain relays "
            "a 'tree' delivery with no refusal")


def test_the_tree_route_streams_instead_of_refusing():
    """The tree route's half of the wiring, beside the run route's above.

    THE TEST THIS REPLACES ASSERTED A REFUSAL THAT NO LONGER EXISTS.  It
    read ``refuse_unrouted_streaming(exp`` out of
    ``prepared_domain_tree_forecast.py`` and pinned its source position
    ahead of the allocations -- true on the lane's own base, where the tree
    route had no streamed-domain builder and the honest thing was to refuse
    before spending VRAM.  On the release line the route WIRES the builders
    (``steppers_for_tree`` with ``builders_for_tree``), the file carries a
    deliberate no-refusal comment at the old call site, and its sibling test
    above already pins the same fact for ``gpuwm run`` -- so a surviving
    refusal-order assertion here would be asking the route to refuse the one
    mode it now serves.  What is pinned instead: the refusal really is GONE
    from the route (not merely moved after the allocations, which was the
    original defect), and the wiring that replaced it is present, on both
    ends of the seam.
    """
    text = (ROOT / "gpuwm" / "prepared_domain_tree_forecast.py").read_text(
        encoding="utf-8")
    assert "refuse_unrouted_streaming(exp" not in text, (
        "prepared_domain_tree_forecast.py refuses [tiles] again while also "
        "wiring builders; one of the two is lying to the user")
    assert "NO streaming refusal for [tiles]" in text, (
        "the deliberate no-refusal comment left the tree route; if the "
        "admission design changed, update this gate with it")
    assert "steppers = streaming.steppers_for_tree(" in text
    assert "builders=streaming.builders_for_tree(model, exp.tiles)" in text
    return ("the tree route wires steppers_for_tree/builders_for_tree and "
            "carries the deliberate no-refusal comment; "
            "refuse_unrouted_streaming is absent from the file")


# --------------------------------------------------------------------------
# the receipt
# --------------------------------------------------------------------------

def test_the_off_receipt_is_empty():
    """THE CONTROL for the receipt: a resident run's receipt must not move.

    Every fingerprint and every receipt written before ``[tiles]``
    existed has to stay byte-identical, or the field is a breaking change
    dressed as an addition.
    """
    assert streaming.receipt_entry(streaming.OFF) == {}
    assert streaming.receipt_entry(None, {}) == {}
    assert streaming.identity_payload_entry(
        streaming.StreamingOptions(mode="on", tile_nx=8, tile_ny=8)) == {}
    return "off contributes {} to the receipt, and on contributes {} to identity"


def test_the_on_receipt_names_the_mode_and_the_decision():
    options = streaming.StreamingOptions(
        mode="on", tile_nx=224, tile_ny=224, nbuffers=2)
    cfg = _exp(448).root.run
    decision = streaming.decide(cfg, options)
    entry = streaming.receipt_entry(options, {1: decision})
    assert entry["configured"]["mode"] == "on"
    assert entry["any_streamed"] is True
    text = entry["decisions"]["d01"]["explain"]
    # "[tiles] ON", not "streaming ON": the explanation is interpolated into
    # make_stepper's refusal, and the product has a second door called
    # `gpuwm stream`.  Pinned here so the vocabulary cannot drift back.
    assert text.startswith("[tiles] ON"), text
    assert "tile 224x224" in text, text
    # And a route that forgot the out-parameter still records the mode,
    # rather than writing an empty entry that reads as "resident".
    bare = streaming.receipt_entry(options)
    assert bare and bare["configured"]["mode"] == "on"
    return f"receipt carries: {text}"


# --------------------------------------------------------------------------
# CONFIGURED AUTO ADMISSION: the old disagreement is a regression control
# --------------------------------------------------------------------------


def test_auto_uses_the_configured_resident_admission():
    """The old empirical resident answer cannot strand a fitting tile road.

    No measurement coefficients or card-name thresholds are repinned here.
    The conservative configured envelope chooses the road; the existing
    tile ledger continues to price that road. Actual GPU peak calibration
    remains the separate ledger_probe experiment above.
    """
    from tilestream import autoplan
    results = []
    for n, free_gib in ((704, 23.5), (960, 31.4)):
        exp = _exp(n, "auto")
        machine = autoplan.Machine(int(free_gib * GIB), 256 * GIB)
        estimate = dataclasses.replace(pf.estimate_experiment(exp),
                                       envelope_family="windows")
        old = streaming.decide(exp.root.run, dataclasses.replace(
            exp.tiles, resident_context=None), machine=machine)
        assert not old.stream
        assert estimate.peak_envelope_bytes > machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
        fixed = streaming.decide(exp.root.run, exp.tiles, machine=machine,
                                 resident_estimate=estimate)
        assert fixed.stream
        envelope = streaming.streamed_envelope(exp.root.run, exp.tiles,
                                               machine=machine, decision=fixed)
        assert envelope.peak_vram_bytes <= machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
        results.append(f"{n}x{n}: resident exceeds allowance; tiled forecast fits")
    return "; ".join(results)


def test_gpuwm_go_admits_tiles_and_gates_them_before_the_download():
    """END TO END on the one route a first-time user actually types.

    ``gpuwm go``'s whole design principle is that a refusal belongs on THIS
    side of the fetch: the memory gate and the geography gate both run before
    the download, because the alternative is spending the user's bandwidth on
    forcing data for a run that cannot happen.  A ``[tiles]`` shape no stage
    can honour is the same kind of refusal and sits beside them.

    REWRITTEN FOR 2.5.0, and the refusal this used to assert is why.  It
    demanded ``plan_from_config`` refuse ``mode = "on"`` outright, quoting
    "wires no streamed-domain builder" -- a statement about the route that
    stopped being true when both prepared runners wired
    ``streaming.builders_for_tree``: the authority stage carries a config's
    ``[tiles]`` table into the hash-bound experiment.toml byte for byte,
    the forecast stage reads it and streams, and ``go``'s memory gate
    prices the streamed envelope.  Re-adding that refusal would reject a
    config the chain demonstrably runs (MEASURED 2026-08-16: a real
    ``gpuwm go`` with ``[tiles] mode = "on"``, single 12 km domain,
    report.json tiles decision STREAM), with a message asserting a
    breakage that does not exist -- it would fail exactly when the product
    is fixed; the 2.2.0 admission-gate incident
    (tests/test_streamed_admission.py) is the ruling on which way the
    admission verdict goes.

    What WAS still wrong is the silence: the plan recorded nothing about
    ``[tiles]``, so ``go`` planned six commands that never said the run
    would stream, and the only evidence was one line five stages in, on a
    captured stdout.  So the contract now pinned is four-sided:

    * the CONTROL -- the same config without the block -- plans cleanly
      and records nothing (``plan["tiles"] is None``), or this test would
      pass against a plan that annotated everything;
    * ``mode = "on"`` on the single-domain chain PLANS, to the same
      runner, and the plan records the routing where the banner and the
      dry run read it -- honored visibly, not silently;
    * a ``[tiles]`` table no stage could honour -- an invalid mode -- is
      still REFUSED on this side of the download, as a ``GoRefusal``
      naming the table, not a traceback after a 160 MiB fetch;
    * a stationary tree preserves both requested streamed endpoints on
      ordinary planning call, before the fetch.
    """
    import contextlib
    import io
    import sys
    import tempfile

    sys.argv = ["gpuwm"]
    from gpuwm import go_cli
    from gpuwm.cli import main as cli_main

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "go.toml"
        tree = Path(tmp) / "go_tree.toml"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cli_main([
                "domain", "--point=35.3,-97.5", "--card", "24gb",
                "--ladder", "12", "--source", "gfs",
                "--cycle", "2026-07-29T18", "--hours", "6",
                "--out", str(out), "--physics-profile",
                "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"])
            assert rc == 0, rc
            rc = cli_main([
                "domain", "--point=35.3,-97.5", "--card", "24gb",
                "--ladder", "12-3", "--source", "gfs",
                "--cycle", "2026-07-29T18", "--hours", "6",
                "--out", str(tree), "--physics-profile",
                "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"])
            assert rc == 0, rc
        # THE CONTROL: the same config, no [tiles], must plan -- silently.
        control = go_cli.plan_from_config(out, outdir=Path(tmp) / "a")
        assert control["tiles"] is None, control["tiles"]
        # The headline: mode = "on" is ADMITTED, planned, routed, and SAID.
        streamed = Path(tmp) / "go_stream.toml"
        streamed.write_text(
            out.read_text(encoding="utf-8") + '\n[tiles]\nmode = "on"\n',
            encoding="utf-8")
        plan = go_cli.plan_from_config(streamed, outdir=Path(tmp) / "b")
        assert plan["config"] == streamed, plan["config"]
        tiles = plan["tiles"]
        assert tiles is not None, (
            "gpuwm go planned a [tiles] mode = 'on' config with no record "
            "of it: the run would stream and the plan never says so")
        assert tiles["mode"] == "on", tiles
        assert tiles["asked"] == ["d01"], tiles
        assert "host store" in tiles["sentence"], tiles["sentence"]
        assert plan["runner"] == go_cli.RUNNER_MODULE, plan["runner"]
        # The refusal that still belongs pre-fetch: a table no stage can
        # honour, refused as a GoRefusal that names the [tiles] key.
        bad = Path(tmp) / "go_bad.toml"
        bad.write_text(
            out.read_text(encoding="utf-8") + '\n[tiles]\nmode = "banana"\n',
            encoding="utf-8")
        try:
            go_cli.plan_from_config(bad, outdir=Path(tmp) / "e")
        except go_cli.GoRefusal as error:
            assert "[tiles]" in str(error) and "banana" in str(error), error
        else:
            raise AssertionError(
                "gpuwm go planned a config whose [tiles] table no stage can "
                "honour; that refusal belongs before the download")
        # A stationary tree may stream both endpoints; its plan retains
        # every requested domain before any fetch takes place.
        tree_streamed = Path(tmp) / "go_tree_stream.toml"
        tree_streamed.write_text(
            tree.read_text(encoding="utf-8") + '\n[tiles]\nmode = "on"\n',
            encoding="utf-8")
        tree_plan = go_cli.plan_from_config(
            tree_streamed, outdir=Path(tmp) / "c")
        assert tree_plan["tiles"]["mode"] == "on"
        assert tree_plan["tiles"]["asked"] == ["d01", "d02"]
        return ("gpuwm go plans the control silently, records the mode='on' "
                "routing on the plan, refuses an invalid [tiles] table as a "
                "GoRefusal, and preserves both-streamed tree settings -- all "
                "before the fetch stage runs")


TESTS = [
    test_the_preflight_estimate_is_identical_with_streaming_on,
    test_the_drift_guard_quantities_are_identical_with_streaming_on,
    test_mode_on_is_refused_at_admission,
    test_mode_off_and_auto_are_not_refused,
    test_the_run_route_streams_instead_of_refusing,
    test_gpuwm_go_admits_tiles_and_gates_them_before_the_download,
    test_the_tree_route_streams_instead_of_refusing,
    test_the_off_receipt_is_empty,
    test_the_on_receipt_names_the_mode_and_the_decision,
    test_auto_uses_the_configured_resident_admission,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            note = test()
        except Exception as error:                    # noqa: BLE001
            failed += 1
            print(f"FAIL  {test.__name__}\n      {type(error).__name__}: "
                  f"{error}")
        else:
            print(f"ok    {test.__name__}"
                  + (f"\n      {note}" if note else ""))
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
