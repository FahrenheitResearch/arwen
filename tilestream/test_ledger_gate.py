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
the two VRAM models              the band where they disagree, walked and
                                 printed -- an OPEN DEFECT, pinned so that
                                 reconciling them turns this file red
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
    # The tree arm and the single-domain arm, both.
    assert "builders=_streaming.builders_for_tree(model, exp.tiles)" in src
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
# THE OPEN DEFECT: two models, one question
# --------------------------------------------------------------------------

def _band(free_bytes: int, step: int = 16, limit: int = 2048):
    """Sizes where autoplan says RESIDENT and preflight REFUSES anyway."""
    from tilestream import autoplan as A

    ap_budget = int(free_bytes * (1.0 - A.VRAM_HEADROOM))
    lo = hi = None
    n = step
    while n <= limit:
        exp = _exp(n)
        cfg = exp.root.run
        cells = cfg.nx * cfg.ny * cfg.nz
        resident = A.footprint_for(cfg).resident_bytes(cells)
        envelope = pf.estimate_experiment(exp).peak_envelope_bytes
        streams = resident > ap_budget
        refuses = envelope > free_bytes
        if (not streams) and refuses:
            lo = n if lo is None else lo
            hi = n
        elif lo is not None and streams:
            break
        n += step
    return lo, hi


def test_the_two_vram_models_disagree_and_auto_answers_the_wrong_one():
    """OPEN DEFECT, pinned so that fixing it turns this file red.

    ``[tiles] mode = "auto"`` asks :mod:`tilestream.autoplan` whether the
    domain fits.  The route that runs it, and ``gpuwm go``'s memory gate
    before that, ask :func:`gpuwm.core.preflight.estimate_experiment`.  The
    two models price the full(real74) rung at 541 B/cell and ~887 B/cell
    respectively, so on every card there is a band of domain sizes where
    autoplan says "resident is fine" -- streaming therefore never fires --
    and preflight then refuses the run outright.  A user in that band who
    turned streaming on to make their domain fit is refused anyway, for a
    reason that has nothing to do with the mode they turned on.

    WHICH MODEL TO MOVE IS A MEASUREMENT, and it has been taken -- see
    ``tilestream/ledger_probe.py``'s docstring.  On an RTX 5090 at 352^2 x 49
    full physics the CuPy pool ALONE holds 6.656 GiB against autoplan's
    7.019 GiB prediction for the whole device, leaving 0.36 GiB for a CUDA
    context autoplan's own constant puts at 0.39.  Autoplan is the optimistic
    one; preflight's refusal in this band is the defensible number, and it is
    ``auto``'s decision NOT to stream that is wrong.  Two rungs on one card is
    a direction, not a re-fit: reconciling them wants autoplan's 29-point
    measurement redone before either constant moves.

    THE BAND PINS BELOW ARE THE 2.5.1 LINE'S, re-derived 2026-08-20 on
    the Windows cut box (the envelope's platform form).  They moved for
    the second time in three days, and both moves are the same shape:
    preflight's envelope fell, so it refuses from a LARGER n, so every
    band's LOW edge rose while every HIGH edge stayed exactly where it
    was.  That asymmetry is the evidence the move is preflight's alone --
    the high edge is autoplan's "resident stops fitting" crossing, and
    autoplan has not been touched.

    * 2026-08-18, the memgate-3080 landing: the x1.75-with-pool-constants
      Windows envelope became the RTX 3080-measured WDDM form and the
      size-independent term fell from ~8.8 GiB to ~2.9 GiB.  Lows rose to
      416/640/752.
    * 2026-08-20, the measured-VRAM-reserve landing: the fit gate stopped
      paying for the same bytes twice -- the machine-peak envelope
      carries the CUDA context and the kernel backing store, and the
      budget had subtracted them again -- and the context stopped being
      one card's 2026-07-26 reading applied to every card.  MEASURED
      here now: envelope 6.14 GiB at 224^2 x 49 against a 2.90 GiB alloc
      estimate (was 6.39), 11.13 GiB at 416^2 (was 12.39), 14.75 GiB at
      512^2 (was 16.72), with the size-independent non-pool term at 2.74
      GiB.  Lows rose to 448/704/832.

    The disagreement is therefore NARROWER again on this line, but it is
    not reconciled, and the defect stands.

    WHEN THE TWO MODELS ARE RECONCILED, DELETE THIS TEST.  It asserts the
    presence of the defect, which is the only way an arithmetic-only audit
    can keep a silent regression from re-opening it.
    """
    bands = {}
    for name, free_gib in (("5070", 11.9), ("4090", 23.5), ("5090", 31.4)):
        lo, hi = _band(int(free_gib * GIB))
        bands[name] = (lo, hi)
        assert lo is not None, (
            f"no disagreement band on the {name} -- if the models were "
            "reconciled, delete this test; if the ladder changed, re-derive")
    assert bands["5070"] == (448, 512), bands
    assert bands["4090"] == (704, 816), bands
    assert bands["5090"] == (832, 976), bands
    return "; ".join(f"{k}: n in [{v[0]}, {v[1]}]" for k, v in bands.items())


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
    * the one ``[tiles]`` shape the chain genuinely cannot run -- a tree
      whose coupling edge has both ends streamed -- is refused before the
      fetch by the core's own edge sentence, relayed at load time.  And
      ``gpuwm go`` proper still refuses the tree as a tree first.
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
        # THE SHAPE THAT CANNOT RUN: both ends of a coupling edge streamed.
        # Refused at load, relayed by the door, before the fetch -- through
        # the tree arm (run-plan's), since go proper never reaches [tiles]
        # on a tree at all:
        tree_streamed = Path(tmp) / "go_tree_stream.toml"
        tree_streamed.write_text(
            tree.read_text(encoding="utf-8") + '\n[tiles]\nmode = "on"\n',
            encoding="utf-8")
        try:
            go_cli.plan_from_config(
                tree_streamed, outdir=Path(tmp) / "c", allow_tree=True)
        except go_cli.GoRefusal as error:
            assert "BOTH ends streamed" in str(error), str(error)
        else:
            raise AssertionError(
                "a tree with every coupling edge streamed was admitted; "
                "the coupler refuses that shape at the first FORCE, after "
                "the fetch and both preparations")
        try:
            go_cli.plan_from_config(tree_streamed, outdir=Path(tmp) / "d")
        except go_cli.GoRefusal as error:
            assert "single-domain runner" in str(error), str(error)
        else:
            raise AssertionError("go proper admitted a domain tree")
        return ("gpuwm go plans the control silently, records the mode='on' "
                "routing on the plan, refuses an invalid [tiles] table as a "
                "GoRefusal, and refuses the both-ends-streamed tree -- all "
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
    test_the_two_vram_models_disagree_and_auto_answers_the_wrong_one,
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
