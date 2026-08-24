"""The admission gate has to price the run the config asks for.

THE DEFECT, MEASURED AT THE 2.2.0 CUT.  ``gpuwm go`` refused a 550x550x49
config with ``[tiles] mode = "on"`` and 200x200 tiles -- the exact shape
streaming exists for -- because every memory term in
:mod:`gpuwm.core.preflight` itemizes a domain RESIDENT in VRAM.  The
refusal read "the forecast is the memory-binding phase at 15.36 GiB peak
envelope ... EXCEEDS the 13.09 GiB budget by 2.27 GiB" on a card with
15.24 GiB free, and the remedy it printed was ``--no-memory-gate``.

``gpuwm check`` said the quiet part out loud on the same file: "[tiles]
mode = 'on' is configured, and every memory number in this report prices
the RESIDENT allocation -- this estimator has no model of a streamed
domain, so a refusal here is a statement about a resident run."

So the one configuration class the release's headline feature exists to
enable was refused BY DEFAULT and reachable only behind a flag, which is
the "fixed means default" law failing in the most direct way available.
These tests are the front door: they drive ``memory_gate`` with a probe
standing in for the card, and they fail if the streamed config stops being
admitted or if a genuinely-too-big one stops being refused.
"""
from __future__ import annotations

import argparse
import json
import textwrap

import pytest

from gpuwm import go_cli
from gpuwm.core import preflight


GIB = 1024 ** 3

#: The card the 2.2.0 user-door leg ran on: an RTX 5070 Ti with nothing
#: else resident.  The resident envelope for the domain below is 15.36 GiB,
#: which is what made the old gate refuse against this exact number.
CARD_FREE_BYTES = int(15.24 * GIB)

_TILES = """\
[tiles]
mode = "on"
tile_nx = 200
tile_ny = 200
"""


#: The smallest domain on the card above that the RESIDENT model still
#: refuses, and the control below's subject.
#:
#: IT USED TO BE 550, and the reason it moved is a fix, not a drift.  The
#: measured-VRAM-reserve landing of 2026-08-20 found the fit gate paying
#: for the same bytes twice: the machine-peak envelope already carries
#: the CUDA context and the kernel backing store, and it was compared
#: against a budget that had subtracted those same bytes AGAIN as part of
#: the allocation reserve.  Charging each measured byte once moved
#: 550^2 x 49 from 15.36 GiB against a 13.09 GiB budget to 15.03 GiB
#: against 14.74 -- it FITS now, and refusing it was the defect.
#: MEASURED here on this card: 550^2 admits resident at 15.03 GiB, 576^2
#: refuses at 16.09 GiB against the 14.74 GiB budget, and 576^2 with the
#: same [tiles] table is admitted at 6.71 GiB.
#:
#: A control has to be a run the gate genuinely cannot admit.  Leaving it
#: pointed at 550^2 would have asserted the old double charge, which is
#: pinning a bug rather than a contract.
_OVER_BUDGET_NX = 576


def _config(tmp_path, *, nx=550, ny=550, tiles=_TILES, name="exp",
            source="gfs"):
    """One GFS-shaped experiment TOML with a single root domain.

    Deliberately written as a TOML and loaded through the product's own
    loader rather than assembled as objects: the finding was about "the
    cannot-fit-resident config a user actually types", and a fixture that
    built an ``ExperimentConfig`` in Python would skip the half of the
    path where ``[tiles]`` has to survive.
    """
    path = tmp_path / f"{name}.toml"
    path.write_text(textwrap.dedent(f"""\
        [experiment]
        name = "synth"
        start_time = 2024-05-03T12:00:00
        run_seconds = 3600.0
        restart_interval_s = 0.0

        [fetch]
        source = "{source}"
        cycle = "2024-05-03T12"
        hours = 1

        [shared]
        nz = 49
        ztop = 20000.0
        moist = true
        moist_cq = true
        mp_physics = 10
        ra_lw_physics = 4
        ra_sw_physics = 4
        sf_sfclay_physics = 91
        sf_surface_physics = 2
        bl_pbl_physics = 1
        cu_physics = 1
        nwp_diagnostics = 1

        """) + tiles + f"""
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = {nx}
ny = {ny}
time_step = 20
dx = 3000.0
history_interval_s = 3600.0
""", encoding="utf-8")
    return path


def _plan(config_path):
    return {"config": str(config_path), "source": "gfs", "cadence": None}


@pytest.fixture()
def card(monkeypatch):
    """Stand in for the device probe, so the gate is testable off a card.

    Patched at ``gpuwm.core.preflight`` because that is where
    ``memory_gate`` imports it from; patching the name in ``go_cli`` would
    leave the real subprocess running and the test would pass or fail on
    whatever card the runner happens to have.
    """
    def _probe(*_args, **_kwargs):
        return {"free_bytes": CARD_FREE_BYTES, "total_bytes": int(16 * GIB),
                "name": "test card", "local_memory_bytes_per_thread": 0}

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", _probe)
    return _probe


def test_the_streamed_config_a_user_types_is_admitted_with_no_flags(
        tmp_path, card):
    """THE REGRESSION.  550sq/200-tile on a 16 GiB card, plain `gpuwm go`.

    The assertion that matters is ``refuse is False``: this is the run the
    user asked for and the gate has no business stopping it.  The verdict
    is asserted to name the STREAMED shape as well, because a gate that
    admitted the run while still describing a resident allocation would
    pass this test and lie in the terminal.
    """
    gate = go_cli.memory_gate(_plan(_config(tmp_path)))

    assert gate["refuse"] is False, gate["verdict"]
    assert gate["phases"].streamed_forecast
    assert "streamed forecast" in gate["verdict"]
    assert "tile buffer(s) of 236x236" in gate["verdict"]
    # And it is genuinely cheaper, not merely relabelled: the streamed
    # envelope has to be a fraction of the resident one it replaced.  The
    # HOLD is under half of it; the reported figure is the radiation peak,
    # which carries a rung-sized transient the resident arithmetic prices
    # elsewhere, and is still the smaller number.
    phases = gate["phases"]
    assert (phases.streamed.vram_bytes
            < 0.5 * phases.resident_forecast_envelope_bytes)
    assert (phases.forecast_envelope_bytes
            < 0.7 * phases.resident_forecast_envelope_bytes)


def test_the_same_config_without_tiles_is_still_refused(tmp_path, card):
    """THE CONTROL, and the reason the test above is not vacuous.

    A domain this card genuinely cannot hold resident is still refused
    when the ``[tiles]`` table is removed, and admitted byte for byte
    when it is there.  Without the pair, a gate that had simply stopped
    refusing anything would pass the regression test above.

    The subject is ``_OVER_BUDGET_NX``, not the 550 the headline test
    uses, and that constant carries the measurement: charging the CUDA
    context and the kernel backing store ONCE instead of twice made
    550^2 x 49 fit this card, so 550 stopped being a control the day the
    double charge was fixed.
    """
    over = {"nx": _OVER_BUDGET_NX, "ny": _OVER_BUDGET_NX}

    gate = go_cli.memory_gate(
        _plan(_config(tmp_path, tiles="", name="resident", **over)))
    assert gate["refuse"] is True, gate["verdict"]
    assert gate["phases"].streamed_forecast is False
    assert "EXCEEDS" in gate["verdict"]

    # ... and [tiles] is what makes the difference, on the same domain.
    streamed = go_cli.memory_gate(
        _plan(_config(tmp_path, name="streamed", **over)))
    assert streamed["refuse"] is False, streamed["verdict"]
    assert streamed["phases"].streamed_forecast


def test_the_headline_domain_now_fits_this_card_resident(tmp_path, card):
    """The double charge is gone, and this is the byte that proves it.

    550^2 x 49 full physics on 15.24 GiB free was the 2.2.0 finding's own
    subject and it was REFUSED resident, at 15.36 GiB against a 13.09 GiB
    budget.  The budget had subtracted the CUDA context and the kernel
    backing store that the envelope already carried, so 2.91 GiB of a
    card was spent twice.  Charged once, the run fits and the gate admits
    it with no flags.

    Pinned as its own statement rather than left implicit in the control
    above, because it is a USER-VISIBLE admission change: a run the
    previous release refused now starts.
    """
    gate = go_cli.memory_gate(_plan(_config(tmp_path, tiles="")))

    assert gate["refuse"] is False, gate["verdict"]
    assert gate["phases"].streamed_forecast is False
    assert gate["phases"].forecast_envelope_bytes < CARD_FREE_BYTES


def test_a_domain_too_big_even_streamed_is_refused_in_streamed_numbers(
        tmp_path, card):
    """Streaming is not a licence to admit everything.

    A tile whose compute window cannot fit the card is still an OOM, and
    the refusal has to be stated in the numbers the user can act on -- the
    tile and its window -- rather than in a resident figure that describes
    no run.
    """
    huge = textwrap.dedent("""\
        [tiles]
        mode = "on"
        tile_nx = 2048
        tile_ny = 2048
        """)
    config = _config(tmp_path, nx=4096, ny=4096, tiles=huge, name="huge")
    gate = go_cli.memory_gate(_plan(config))

    assert gate["refuse"] is True
    assert gate["phases"].streamed_forecast
    assert "tile buffer(s) of 2084x2084" in gate["verdict"]


def test_the_check_advisory_no_longer_claims_there_is_no_streamed_model(
        tmp_path):
    """F5: the stale sentence that contradicted the shipped feature.

    ``gpuwm check`` told every user of a ``[tiles]`` config that "'on' is
    refused by the forecast routes, which wire no streamed-domain builder"
    -- describing the release's own headline feature as not working, a
    release after the wiring landed.
    """
    exp = preflight._load_experiment_any(_config(tmp_path))
    advisory = preflight.streaming_advisory(exp)

    assert advisory is not None
    assert "wire no streamed-domain builder" not in advisory
    assert "has no model of a streamed domain" not in advisory
    assert "streams this domain" in advisory


# ---------------------------------------------------------------------------
# The SAME question at the other door: `gpuwm check`
# ---------------------------------------------------------------------------
#
# ``gpuwm go``'s gate learned the streamed envelope at 2.2.0 and the report
# door did not, on two separate seams:
#
# * the phase estimate was DISCARDED whole -- streamed forecast term
#   included -- whenever the source's ingest lane could not be priced, so
#   every config forced to a source outside SOURCE_ANALYSIS_LEVELS fell
#   back to the resident envelope and was refused at exit 4 / exit 1;
# * the phase estimate was asked for without a ``machine``, so
#   ``mode = "auto"`` was priced against whatever card happened to be in
#   the machine running the report rather than the one it was told to size
#   for -- or, with no CuPy at all, silently reverted to resident.
#
# The pair below is the contract: the envelope the report prints, the exit
# code it returns and the gate leg it evaluates all describe the run the
# config asks for, AND a config that does not fit even streamed is still
# refused, in streamed numbers.

#: Straddles the streamed envelope of the 550sq/200-tile config above
#: (9.45 GiB at the radiation peak against the 10.79 GiB envelope budget
#: this declares) while leaving the resident one (14.30 GiB) far outside
#: it.  ``--budget-gib`` alone, with no ``--vram-gib``: naming a capacity
#: would let the report recognise the card under the desk and price the
#: local profile, which makes the numbers a property of the machine
#: running the tests.
#:
#: IT USED TO BE 6, and it moved for a fix rather than a drift: the
#: streamed envelope carried only what the tiling HOLDS, so 6 straddled
#: 6.71 GiB against an 8.79 GiB budget.  With the measured RRTMGP
#: transient in the figure the same config reaches 9.45 GiB and 6 no
#: longer admits it -- correctly, because the run meets that peak at its
#: first radiation call.  See the radiation-transient section below.
_FITS_STREAMED_GIB = 8

#: Below the streamed envelope too (5.79 GiB envelope budget).  The gate
#: has to keep refusing here or it is not a gate.
_FITS_NEITHER_GIB = 3


def _run_check(argv):
    """`gpuwm check` through its own registrar, as the CLI builds it."""
    parser = argparse.ArgumentParser(prog="gpuwm")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight.register_cli(sub)
    args = parser.parse_args(argv)
    return args.func(args)


def _check(capsys, config, *flags):
    rc = _run_check(["check", str(config), "--json", *flags])
    return rc, json.loads(capsys.readouterr().out)


def _streamed(config, **kwargs):
    exp = preflight._load_experiment_any(config)
    return preflight.streamed_forecast_envelope(exp, **kwargs)


def test_check_prices_the_streamed_forecast_when_ingest_is_unpriced(
        tmp_path, capsys):
    """THE REGRESSION, at the door that reports rather than runs.

    A source this estimator does not model costs the reader the INGEST
    phase, which is a real gap and is said out loud.  It must not also
    cost them the streamed forecast term, which has nothing to do with
    the source: discarding the whole phase estimate to drop one of its
    two halves is what put a resident envelope on a streamed config and
    refused it.
    """
    config = _config(tmp_path, source="hrrr")
    rc, payload = _check(capsys, config, "--budget-gib",
                         str(_FITS_STREAMED_GIB))

    streamed = _streamed(config)
    assert payload["peak_envelope_bytes"] == int(streamed.peak_vram_bytes)
    # ...and it is genuinely the smaller figure, not a relabelled one:
    # the tiling holds under half what the resident domain would, and the
    # radiation peak on top of it is still below the resident envelope.
    assert (payload["observed_peak_envelope_bytes"]
            > 2 * payload["streamed"]["vram_bytes"])
    assert (payload["peak_envelope_bytes"]
            < payload["observed_peak_envelope_bytes"])
    assert payload["observed_peak_envelope_exceeds_budget"] is False
    assert rc == 0, payload["phase_verdict"]

    # The ingest SECTION is what an unpriced source removes, and its
    # absence is stated rather than silent.
    assert payload["ingest"] is None
    assert "NOT PRICED" in payload["ingest_not_priced_reason"]
    assert "hrrr" in payload["ingest_not_priced_reason"]
    # The verdict still names the phase and the tiling that produced it.
    assert "streamed forecast" in payload["phase_verdict"]
    assert "tile buffer(s) of 236x236" in payload["phase_verdict"]


def test_check_alloc_gate_admits_a_streamed_config_that_fits(
        tmp_path, capsys):
    """The N0 alloc leg priced a resident allocation on a streamed run.

    Fed the itemized resident estimate, ``alloc_estimate_le_wddm_budget``
    failed at exit 1 for exactly the configurations streaming exists to
    enable -- a harder refusal than the envelope's exit 4, and one no
    amount of correct envelope reporting could have unstuck.
    """
    config = _config(tmp_path, source="hrrr")
    rc, payload = _check(capsys, config, "--budget-gib",
                         str(_FITS_STREAMED_GIB))

    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is True
    assert rc == 0


def test_check_alloc_gate_still_refuses_what_even_streaming_cannot_fit(
        tmp_path, capsys):
    """THE CONTROL.  A gate that stopped refusing is not a gate.

    Same config, a budget below the STREAMED envelope as well: refused,
    and refused in the streamed numbers, because a reader told "trim it"
    has to be trimming the thing that is actually too big.
    """
    config = _config(tmp_path, source="hrrr")
    rc, payload = _check(capsys, config, "--budget-gib",
                         str(_FITS_NEITHER_GIB))

    streamed = _streamed(config)
    assert payload["peak_envelope_bytes"] == int(streamed.peak_vram_bytes)
    assert payload["peak_envelope_bytes"] > payload["envelope_budget_bytes"]
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is False
    assert rc == 1


def test_check_prices_an_auto_tiling_against_the_card_it_was_told_about(
        tmp_path, capsys):
    """``mode = "auto"`` is the planner's decision, and the planner needs a
    card.  Asked without one it consults ``Machine.detect``, which reads
    whatever is under the desk -- so a report sizing a 6 GiB target
    described the tiling of the 16 GiB card printing it, and on a box with
    no CuPy at all quietly reverted to the resident envelope.

    The declared free figure the report itself publishes is the card it was
    told about; the tiling it prints has to be that card's.
    """
    from tilestream import autoplan

    from gpuwm.core.streaming import _host_total_bytes

    config = _config(tmp_path, tiles='[tiles]\nmode = "auto"\n',
                     source="hrrr", name="auto")
    rc, payload = _check(capsys, config, "--budget-gib",
                         str(_FITS_STREAMED_GIB))

    machine = autoplan.Machine(
        vram_bytes=int(payload["measured_free_bytes"]),
        host_bytes=int(_host_total_bytes()), name="declared",
        host_source="probe")
    expected = _streamed(config, machine=machine)
    assert expected is not None, "fixture no longer reaches the planner"
    assert payload["peak_envelope_bytes"] == int(expected.peak_vram_bytes)
    assert rc == 0


def test_the_report_names_one_tiling_not_two(tmp_path, capsys):
    """The advisory and the verdict describe the same run.

    ``streaming_advisory`` derived its own envelope, and under
    ``mode = "auto"`` that derivation reached ``Machine.detect`` -- the
    card under the desk -- while the verdict planned against the card
    ``--budget-gib`` named.  One report then opened with "2 tile
    buffer(s) of 220x174 ... 5.80 GiB" and said "2 tile buffer(s) of
    311x146 ... 6.17 GiB" four lines later, which teaches a reader that
    one of the two is noise.
    """
    config = _config(tmp_path, tiles='[tiles]\nmode = "auto"\n',
                     source="hrrr", name="auto-one-answer")
    _rc, payload = _check(capsys, config, "--budget-gib",
                          str(_FITS_STREAMED_GIB))

    advisory = next(line for line in payload["advisories"]
                    if "[tiles]" in line)
    window = (f"{payload['streamed']['window_nx']}x"
              f"{payload['streamed']['window_ny']}")
    assert window in advisory, advisory
    assert window in payload["phase_verdict"]


def test_check_on_a_config_that_does_not_stream_is_byte_identical(
        tmp_path, capsys):
    """THE REGRESSION FENCE.  Nothing above may touch resident arithmetic.

    Pinned as integers rather than as a relation, because the failure this
    guards against is a silent drift of a few bytes in a number users
    compare against their card.
    """
    config = _config(tmp_path, tiles="", source="hrrr", name="resident")
    rc, payload = _check(capsys, config, "--budget-gib",
                         str(_FITS_STREAMED_GIB))

    assert payload["peak_envelope_bytes"] == 15350499065
    assert payload["observed_peak_envelope_bytes"] == 15350499065
    assert payload["alloc_estimate_bytes"] == 12184233721
    assert payload["reserve_bytes"] == 3531792356
    assert payload["budget_bytes"] == _FITS_STREAMED_GIB * GIB
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is False
    assert rc == 1

    # ...and they are the itemizer's own numbers, not a transcription.
    exp = preflight._load_experiment_any(config)
    estimate = preflight.estimate_experiment(exp, vram_gib=None)
    assert payload["alloc_estimate_bytes"] == estimate.alloc_estimate_bytes


def test_an_unconfigured_run_is_priced_exactly_as_it_always_was(tmp_path):
    """The mode's existence must cost a resident run nothing.

    No ``[tiles]`` table means no planner, no envelope and the same
    forecast term as before -- asserted as an identity against the
    resident number rather than as a range.
    """
    exp = preflight._load_experiment_any(_config(tmp_path, tiles=""))
    phases = preflight.estimate_phases(exp, source="gfs")

    assert phases.streamed is None
    assert (phases.forecast_envelope_bytes
            == phases.resident_forecast_envelope_bytes
            == phases.forecast.peak_envelope_bytes)


# ---------------------------------------------------------------------------
# The RADIATION TRANSIENT: what the card holds when RRTMGP actually fires
# ---------------------------------------------------------------------------
#
# ``StreamedEnvelope.vram_bytes`` prices what a streamed forecast HOLDS --
# the CUDA context, the rung's per-process fixed cost and the tile buffers.
# A radiation step then allocates, uses and frees a working set on top of
# that, MEASURED at +2.74 GiB on the three-domain 9/3/1 km run and carried
# as ``tilestream.autoplan.RADIATION_TRANSIENT_BYTES``.  It recurs every
# radiation period and lasts 60-75 s of it, and the first call is
# ``itimestep == 1``.
#
# THE OMISSION.  The planner already reserves it before it picks a tile
# (``autoplan.budget_for``), and the streamed-init road already adds it
# before it picks a road (``psdf._streamed_peak_bytes``).  The ESTIMATE
# surfaces -- the wizard's fit, ``gpuwm check``'s streamed alloc leg,
# ``run-plan --estimate`` and ``gpuwm go``'s memory gate -- priced the
# steady state alone.  A PINNED tiling makes that the whole story: it
# consults no planner, so nothing anywhere on its path had the number at
# all.  A card between the 6.71 GiB steady state and the 9.45 GiB
# radiation peak of the fixture above was admitted by every one of those
# surfaces and then met the transient at the first radiation call.

#: Free VRAM above every OTHER phase of the fixture -- its 6.71 GiB
#: streamed steady state and its 8.34 GiB ingest phase -- and below the
#: 9.45 GiB the forecast reaches when radiation fires.  This is the card
#: the omission admits and the run dies on, and the FORECAST is what has
#: to bind here or the leg is measuring the ingest term instead.
_FITS_STEADY_NOT_PEAK_BYTES = int(9.0 * GIB)

#: ``--budget-gib`` whose 8.79 GiB envelope budget straddles the same
#: pair: the 6.71 GiB steady figure fits it and the 9.45 GiB radiation
#: peak does not.  This is the value ``_FITS_STREAMED_GIB`` held before
#: the transient reached these surfaces, which is the same statement from
#: the other side -- it was admitting a run this card cannot finish.
#: ``_FITS_STREAMED_GIB`` itself is the arm that must still be ADMITTED,
#: because a fix that refused everything would pass the legs below.
_BUDGET_FITS_STEADY_NOT_PEAK_GIB = 6

#: A pinned tiling at a rung that runs no radiation.  Its transient is
#: zero by measurement, so every figure on its path must be byte-identical
#: to what it was before this reservation reached these surfaces.
_MOIST_SHARED = """\
nz = 49
ztop = 20000.0
moist = true
moist_cq = true
mp_physics = 10
"""


def _moist_config(tmp_path, *, nx=550, ny=550, name="moist"):
    """The same domain and the same pinned tiling at the ``moist`` rung."""
    path = tmp_path / f"{name}.toml"
    path.write_text(textwrap.dedent("""\
        [experiment]
        name = "synth"
        start_time = 2024-05-03T12:00:00
        run_seconds = 3600.0
        restart_interval_s = 0.0

        [fetch]
        source = "gfs"
        cycle = "2024-05-03T12"
        hours = 1

        [shared]
        """) + _MOIST_SHARED + "\n" + _TILES + f"""
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = {nx}
ny = {ny}
time_step = 20
dx = 3000.0
history_interval_s = 3600.0
""", encoding="utf-8")
    return path


def test_the_streamed_envelope_carries_the_measured_radiation_transient(
        tmp_path):
    """THE OMISSION, at the object every estimate surface reads.

    Both halves traceable to the constant they came from, and the peak
    stated as its own field rather than left for four callers to add on.
    """
    from tilestream import autoplan

    config = _config(tmp_path)
    exp = preflight._load_experiment_any(config)
    env = _streamed(config)

    assert env is not None, "fixture no longer streams"
    fp = autoplan.footprint_for(exp.domains[0].run)
    assert env.rung == "full"
    assert (env.radiation_transient_bytes
            == fp.radiation_transient_bytes
            == autoplan.RADIATION_TRANSIENT_BYTES["full"])
    assert env.peak_vram_bytes == (env.vram_bytes
                                   + env.radiation_transient_bytes)
    # 6.71 GiB held, 2.74 GiB more when RRTMGP fires.
    assert env.vram_bytes / GIB == pytest.approx(6.71, abs=0.05)
    assert env.peak_vram_bytes / GIB == pytest.approx(9.45, abs=0.05)


def test_a_rung_that_runs_no_radiation_prices_no_transient(tmp_path):
    """THE FENCE.  Zero where it was measured zero, and the same bytes.

    Every dry and moist plan has to be byte-identical to what it was
    before the reservation reached these surfaces, or the fix is a tax on
    configurations the measurement says nothing about.
    """
    config = _moist_config(tmp_path)
    env = _streamed(config)

    assert env is not None, "fixture no longer streams"
    assert env.rung == "moist"
    assert env.radiation_transient_bytes == 0
    assert env.peak_vram_bytes == env.vram_bytes


def test_go_refuses_a_card_that_only_fits_the_steady_state(
        tmp_path, monkeypatch):
    """THE REGRESSION at the door that spends the user's bandwidth.

    9.0 GiB free holds the 6.71 GiB the tiling steadily holds and does
    not hold the 9.45 GiB it reaches when radiation fires.  Admitting it
    is a download, a preparation and a forecast that stops completing
    steps at the first radiation call.
    """
    def _probe(*_args, **_kwargs):
        return {"free_bytes": _FITS_STEADY_NOT_PEAK_BYTES,
                "total_bytes": int(10.0 * GIB), "name": "test card",
                "local_memory_bytes_per_thread": 0}

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", _probe)
    gate = go_cli.memory_gate(_plan(_config(tmp_path)))

    assert gate["phases"].streamed_forecast
    assert gate["refuse"] is True, gate["verdict"]
    # THE FORECAST is what binds, not the ingest phase this card also
    # nearly meets: a leg that refused on ingest would pass with the
    # transient still missing.
    assert gate["phases"].binding_phase == "forecast"
    assert gate["phases"].forecast_envelope_bytes > _FITS_STEADY_NOT_PEAK_BYTES


def test_go_still_admits_the_same_config_on_a_card_that_fits_the_peak(
        tmp_path, card):
    """THE CONTROL.  15.24 GiB free holds the peak, and still admits.

    Without this the leg above is satisfied by a gate that has simply
    started refusing streamed runs.
    """
    gate = go_cli.memory_gate(_plan(_config(tmp_path)))

    assert gate["refuse"] is False, gate["verdict"]
    assert gate["phases"].streamed_forecast
    assert gate["phases"].forecast_envelope_bytes < CARD_FREE_BYTES


def test_check_prices_the_radiation_transient_into_the_streamed_envelope(
        tmp_path, capsys):
    """``gpuwm check``'s reported envelope is the peak, and says so.

    The steady figure stays reportable beside it -- the two answer
    different questions, and a reader comparing a receipt against an NVML
    steady-state reading needs the first one -- but the number every
    verdict and every gate compares is the peak.
    """
    config = _config(tmp_path, source="hrrr")
    _rc, payload = _check(capsys, config, "--budget-gib",
                          str(_FITS_STREAMED_GIB))
    env = _streamed(config)

    assert payload["peak_envelope_bytes"] == int(env.peak_vram_bytes)
    assert payload["streamed"]["vram_bytes"] == int(env.vram_bytes)
    assert (payload["streamed"]["radiation_transient_bytes"]
            == int(env.radiation_transient_bytes))
    assert (payload["streamed"]["peak_vram_bytes"]
            == int(env.peak_vram_bytes))


def test_check_alloc_gate_refuses_a_budget_that_only_fits_the_steady_state(
        tmp_path, capsys):
    """THE REGRESSION at the report door, on the gate that exits 1.

    The same pair as the ``go`` leg above, declared rather than probed:
    the streamed alloc leg weighed 6.71 GiB against an 8.79 GiB envelope
    budget and passed, on a card that meets 9.45 GiB at the first
    radiation call.
    """
    config = _config(tmp_path, source="hrrr")
    rc, payload = _check(capsys, config, "--budget-gib",
                         str(_BUDGET_FITS_STEADY_NOT_PEAK_GIB))
    env = _streamed(config)

    assert int(env.vram_bytes) <= payload["envelope_budget_bytes"], (
        "fixture no longer straddles the budget")
    assert int(env.peak_vram_bytes) > payload["envelope_budget_bytes"]
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is False
    assert rc == 1


def test_check_alloc_gate_still_admits_a_budget_that_fits_the_peak(
        tmp_path, capsys):
    """THE CONTROL for the leg above, one budget step up."""
    config = _config(tmp_path, source="hrrr")
    rc, payload = _check(capsys, config, "--budget-gib",
                         str(_FITS_STREAMED_GIB))

    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is True
    assert rc == 0


def test_run_plan_estimate_quotes_the_radiation_peak(tmp_path):
    """The figure a front end renders verbatim is the peak, not the hold.

    ``run-plan --estimate`` is subprocessed by a front end and drawn as
    "this run needs N GiB"; quoting the steady state there sizes a card
    the run then meets the transient on.
    """
    import gpuwm.runplan as runplan_module

    geog = tmp_path / "GEOG"
    geog.mkdir(exist_ok=True)
    config = _config(tmp_path)
    plan = runplan_module.build_plan(
        {"schema": runplan_module.PLAN_SCHEMA, "name": "transient",
         "route": "prepared", "config": {"path": str(config)},
         "run_options": {"geog_root": str(geog)},
         "output_root": str(tmp_path / "run")},
        source="plan.json", base_dir=tmp_path, sha256="0" * 64)
    document = json.loads(json.dumps(runplan_module.estimate_plan(plan)))
    env = _streamed(config)

    vram = document["vram"]
    assert vram["envelope_basis"] == "streamed"
    assert vram["peak_envelope_bytes"] == int(env.peak_vram_bytes)
    assert vram["estimate_bytes"] == int(env.peak_vram_bytes)
    assert vram["streamed"]["vram_bytes"] == int(env.vram_bytes)
    assert (vram["streamed"]["radiation_transient_bytes"]
            == int(env.radiation_transient_bytes))


def test_the_wizard_fit_seam_weighs_the_radiation_peak(tmp_path):
    """The door that EMITS a config sizes on the same figure.

    The wizard compares ``estimate_phases(...).peak_envelope_bytes``
    against ``sizing_budget_bytes`` at every candidate layout and again
    on the emitted file.  Fed the steady state it certifies a layout that
    its own follow-up ``gpuwm check`` would refuse -- and worse, one the
    user then runs.
    """
    from gpuwm.domain_wizard import sizing_budget_bytes

    config = _config(tmp_path)
    exp = preflight._load_experiment_any(config)
    phases = preflight.estimate_phases(exp, source="gfs")
    env = _streamed(config)

    assert phases.streamed_forecast
    assert phases.forecast_envelope_bytes == int(env.peak_vram_bytes)
    budget = sizing_budget_bytes(
        exp, free_bytes=_FITS_STEADY_NOT_PEAK_BYTES, vram_gib=None,
        forcing_interval_seconds=21600.0)
    assert int(env.vram_bytes) <= budget, "fixture no longer straddles"
    assert phases.peak_envelope_bytes > budget
