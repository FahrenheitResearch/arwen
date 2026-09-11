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
from dataclasses import replace
import json
import math
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


def _total_from_terms(env, *, radiation_storage=True):
    """Re-add the terms the refusal prints, the way the model states them.

    The itemized prepared model (:mod:`gpuwm.core.prepared_tile_memory`)
    prices a streamed process as: ``nbuffers`` buffers of the compute
    window, each the sum of its named categories, plus the retained
    template and the k-tables, held at no less than the loader's
    initialization peak plus the k-tables; that pool under the allocator
    headroom; plus the CUDA context, the local-memory reservation and the
    unmodelled floor.  This helper performs exactly that arithmetic from
    ``env.terms`` -- the lines ``gpuwm go`` and ``gpuwm check`` print under
    a refusal -- so every figure below is derived from the priced terms
    rather than copied from a run.

    ``radiation_storage=False`` leaves the per-buffer RTE named storage
    out, which is what the process holds BETWEEN radiation calls: the
    model retains that storage per stream inside ``vram_bytes`` instead of
    reserving a separate transient, so the pair (without, with) is the
    steady-versus-peak straddle the fixtures below are built on.
    """
    terms = dict(env.terms)
    per_buffer = int(terms["buffer/total_bytes"])
    if not radiation_storage:
        per_buffer -= int(terms["buffer/radiation_named_storage_bytes"])
    pool = (int(env.nbuffers) * per_buffer
            + int(terms["fixed/template_resident_bytes"])
            + int(terms["fixed/k_tables_bytes"]))
    pool = max(pool, int(terms["fixed/loader_pool_peak_bytes"])
               + int(terms["fixed/k_tables_bytes"]))
    return (math.ceil(preflight.ALLOCATOR_HEADROOM * pool)
            + int(terms["fixed/cuda_context_bytes"])
            + int(terms["fixed/local_memory_bytes"])
            + int(terms["fixed/unmodelled_bytes"]))


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
    # figure is the itemized model's own total for two 236x236 buffers --
    # re-added here from the terms the refusal would print -- and it is
    # below the resident envelope it replaced.  It used to be asserted
    # under half of the resident figure; that fraction belonged to the
    # affine measured-rung model, which priced no radiation storage in
    # the hold.  The itemized model retains the RTE peak per buffer, so
    # the streamed run costs what its two windows cost and no fraction
    # of the resident number is a contract.
    phases = gate["phases"]
    assert phases.streamed.vram_bytes == _total_from_terms(phases.streamed)
    assert phases.forecast_envelope_bytes == phases.streamed.peak_vram_bytes
    assert (phases.forecast_envelope_bytes
            < phases.resident_forecast_envelope_bytes)


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
#: (11.80 GiB under the itemized model, 2026-09-10, against the 12.80 GiB
#: envelope budget this declares) while leaving the resident one (14.31
#: GiB) outside it.  ``--budget-gib`` alone, with no ``--vram-gib``:
#: naming a capacity would let the report recognise the card under the
#: desk and price the local profile, which makes the numbers a property
#: of the machine running the tests.
#:
#: IT USED TO BE 6, THEN 8, and each move was a fix rather than a drift.
#: At 6 the streamed envelope carried only what the affine rung model
#: said the tiling HOLDS (6.71 GiB against an 8.79 GiB budget); at 8 the
#: measured RRTMGP transient was added on top (9.45 GiB against 10.79).
#: The itemized prepared model prices each buffer at its own window --
#: state, physics, scratch, forcing, the step transient, the RTE named
#: storage retained per stream and the column workspace -- and the two
#: 236x236 buffers of this fixture come to 11.80 GiB with the floors and
#: the allocator headroom, which 8 no longer admits.  The tests below do
#: not pin that figure: they re-add it from the printed terms.
_FITS_STREAMED_GIB = 10

#: Below the streamed envelope too (5.80 GiB envelope budget).  The gate
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
    # ...and it is genuinely the smaller figure, not a relabelled one: the
    # streamed term is the itemized model's total for the two windows,
    # re-added from the terms the report prints, and it is below the
    # resident envelope.  (It was once asserted under half the resident
    # figure; that fraction was the affine rung model's, which priced no
    # radiation storage into the hold.)
    assert (payload["streamed"]["vram_bytes"]
            == int(streamed.vram_bytes)
            == _total_from_terms(streamed))
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
        tmp_path, capsys, monkeypatch):
    """THE REGRESSION FENCE.  Nothing above may touch resident arithmetic.

    Pinned as integers rather than as a relation, because the failure this
    guards against is a silent drift of a few bytes in a number users
    compare against their card.
    """
    config = _config(tmp_path, tiles="", source="hrrr", name="resident")
    rc, payload = _check(capsys, config, "--budget-gib",
                         str(_FITS_STREAMED_GIB))

    # USTM (3669990e8c) adds one 550x550 FP32 surface plane;
    # EOS (6b11e4c994) adds dc3f/dc4f/dphb_resid, three 49-value
    # vectors on this flat grid: 1,210,588 real resident bytes.
    # The CFL diagnostic is off and contributes exactly zero here.
    # Resolved radiation selection (267900003) also prices the requested
    # 4/4 modules when the legacy ra_physics alias is zero. The RTE ceiling
    # is 5152 B/thread versus Morrison's 5120: 32 * 170 * 1536 = 8,355,840
    # non-pool bytes on the reference card. No itemized allocation moves.
    assert payload["peak_envelope_bytes"] == 15360247081
    assert payload["observed_peak_envelope_bytes"] == 15360247081
    assert payload["alloc_estimate_bytes"] == 12185625897
    assert payload["reserve_bytes"] == 3540189961
    assert payload["budget_bytes"] == _FITS_STREAMED_GIB * GIB
    assert payload["gates"]["alloc_estimate_le_wddm_budget"] is False
    assert rc == 1

    # ...and they are the itemizer's own numbers, not a transcription.
    exp = preflight._load_experiment_any(config)
    estimate = preflight.estimate_experiment(exp, vram_gib=None)
    assert payload["alloc_estimate_bytes"] == estimate.alloc_estimate_bytes

    # Retain the old pins as an executable attribution control. Before
    # 267900003, domain_kernel_modules checked only ra_physics == 4 and
    # omitted these four modules for an explicitly declared LW/SW pair.
    assert exp.root.run.ra_physics == 0
    assert preflight.radiation_scheme_ids(exp.root.run) == (4, 4)
    frames = preflight.kernel_local_frame_bytes(exp)
    radiation_modules = {"rrtmgp_cloud", "rrtmgp_gas", "rrtmgp_mcica",
                         "rrtmgp_rte"}
    assert (set(preflight._radiation_44_kernel_modules(exp.root.run))
            == radiation_modules)
    assert frames["rrtmgp_rte"] == max(frames.values()) == 5152
    with monkeypatch.context() as old_accounting:
        # Remove only the newly reachable module accounting, preserving all
        # physics selectors, arrays, workspace prices and admission logic.
        old_accounting.setattr(preflight, "_radiation_44_kernel_modules",
                               lambda run: ())
        old_frames = preflight.kernel_local_frame_bytes(exp)
        assert old_frames == {name: size for name, size in frames.items()
                              if name not in radiation_modules}
        assert old_frames["morrison"] == max(old_frames.values()) == 5120
        old_estimate = preflight.estimate_experiment(exp, vram_gib=None)
        old_rc, old_payload = _check(capsys, config, "--budget-gib",
                                     str(_FITS_STREAMED_GIB))

    # The actual command regains every original pin, with the same refusal.
    assert old_payload["peak_envelope_bytes"] == 15351891241
    assert old_payload["observed_peak_envelope_bytes"] == 15351891241
    assert old_payload["alloc_estimate_bytes"] == 12185625897
    assert old_payload["reserve_bytes"] == 3531834121
    assert old_rc == rc == 1
    delta = (5152 - 5120) * 170 * 1536
    assert delta == 8355840
    assert (estimate.non_pool_device_bytes - old_estimate.non_pool_device_bytes
            == delta)
    assert replace(old_estimate,
                   non_pool_device_bytes=estimate.non_pool_device_bytes) == estimate
    for key in ("peak_envelope_bytes", "observed_peak_envelope_bytes",
                "reserve_bytes"):
        assert payload[key] - old_payload[key] == delta


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
# The RADIATION PEAK: what the card holds when RRTMGP actually fires
# ---------------------------------------------------------------------------
#
# ``StreamedEnvelope.vram_bytes`` prices what a streamed forecast HOLDS.
# Under the affine measured-rung model that was the CUDA context, the
# rung's per-process fixed cost and the tile buffers, and a radiation
# step then allocated, used and freed a working set on top of it --
# MEASURED at +2.74 GiB on the three-domain 9/3/1 km run and carried as
# ``tilestream.autoplan.RADIATION_TRANSIENT_BYTES``, added to the hold as
# ``radiation_transient_bytes`` to make ``peak_vram_bytes``.
#
# THE ITEMIZED MODEL PRICES THE PEAK INSIDE THE BUFFER.  A prepared
# forecast is priced by :mod:`gpuwm.core.prepared_tile_memory`, which
# itemizes each buffer at its own window and RETAINS the RTE named storage
# per stream (``buffer/radiation_named_storage_bytes``: the standalone
# workspace of the unfused LW/SW call at that window's column count), so
# what the card holds at the instant radiation fires is already in
# ``vram_bytes`` and the separate transient is zero for these fixtures.
# The straddle that mattered -- a card or budget that holds the process
# BETWEEN radiation calls and not AT one -- is still the defect the
# omission of 2.5 admitted (a download, a preparation and a forecast that
# dies at ``itimestep == 1``), and it is built below from the model's own
# terms: the total WITHOUT the per-buffer radiation storage on one side,
# the priced total WITH it on the other.
#
# THE OMISSION, as it was found: the planner reserved the transient before
# it picked a tile (``autoplan.budget_for``) and the streamed-init road
# added it before it picked a road, but the ESTIMATE surfaces -- the
# wizard's fit, ``gpuwm check``'s streamed alloc leg, ``run-plan
# --estimate`` and ``gpuwm go``'s memory gate -- priced the steady state
# alone.  A PINNED tiling made that the whole story: it consults no
# planner, so nothing anywhere on its path had the number at all.

#: Free VRAM above what the fixture holds BETWEEN radiation calls (its
#: two buffers without their RTE named storage: 7.80 GiB under the
#: itemized model, 2026-09-10, and above its 8.34 GiB ingest phase) and
#: below the 11.80 GiB it holds when radiation fires.  This is the card
#: the omission admits and the run dies on, and the FORECAST is what has
#: to bind here or the leg is measuring the ingest term instead.  The
#: tests do not pin either figure: they re-add both from the terms.
_FITS_STEADY_NOT_PEAK_BYTES = int(9.0 * GIB)

#: ``--budget-gib`` whose 8.80 GiB envelope budget straddles the same
#: pair: the hold without radiation storage fits it and the priced peak
#: does not.  This is the value ``_FITS_STREAMED_GIB`` held before the
#: radiation peak reached these surfaces, which is the same statement from
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


def test_the_streamed_envelope_carries_the_radiation_peak_inside_its_buffers(
        tmp_path):
    """THE OMISSION, at the object every estimate surface reads.

    Every term traceable to the arithmetic it came from.  The itemized
    prepared model retains the RTE named storage per buffer, so the
    radiation peak is a line of the buffer the refusal prints -- equal to
    the standalone workspace at that window's column count -- and the
    separate transient is zero; the peak every surface compares is then
    the hold itself.  The total is re-added from the printed terms, and
    the hold without that storage is strictly less, which is the straddle
    the legs below stand on.
    """
    from gpuwm.core.prepared_tile_memory import standalone_rte_storage_bytes

    config = _config(tmp_path)
    exp = preflight._load_experiment_any(config)
    env = _streamed(config)

    assert env is not None, "fixture no longer streams"
    assert env.rung == "full"
    terms = dict(env.terms)
    # The radiation peak is a named term of the buffer, not a reservation
    # beside it: the standalone unfused LW/SW storage at the window's own
    # column count, retained for every stream.
    radiation = int(terms["buffer/radiation_named_storage_bytes"])
    assert radiation > 0
    assert radiation == sum(standalone_rte_storage_bytes(
        int(exp.root.run.nz), env.window_nx * env.window_ny,
        int(exp.column_chunk), exp.vertical.p_top).values())
    assert terms["radiation_transient_bytes"] == env.radiation_transient_bytes == 0
    assert env.peak_vram_bytes == env.vram_bytes
    # A buffer is the sum of the categories printed for it, and the total
    # is the model's arithmetic over those terms.
    categories = ("state", "physics", "scratch", "lbc", "diagnostic",
                  "step_transient", "radiation_named_storage",
                  "column_workspace")
    assert terms["buffer/total_bytes"] == sum(
        int(terms[f"buffer/{name}_bytes"]) for name in categories)
    assert terms["buffers_bytes"] == env.nbuffers * terms["buffer/total_bytes"]
    assert env.vram_bytes == terms["vram_bytes"] == _total_from_terms(env)
    assert _total_from_terms(env, radiation_storage=False) < env.vram_bytes


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

    9.0 GiB free holds what the tiling holds between radiation calls
    (its buffers without their RTE named storage) and does not hold what
    it reaches when radiation fires.  Admitting it is a download, a
    preparation and a forecast that stops completing steps at the first
    radiation call.
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
    the streamed alloc leg once weighed the hold alone against an 8.80
    GiB envelope budget and passed, on a card that meets the radiation
    peak at the first call.  The straddle is re-added from the model's
    terms: the hold without the per-buffer RTE storage fits the budget,
    the priced peak does not.
    """
    config = _config(tmp_path, source="hrrr")
    rc, payload = _check(capsys, config, "--budget-gib",
                         str(_BUDGET_FITS_STEADY_NOT_PEAK_GIB))
    env = _streamed(config)

    assert (_total_from_terms(env, radiation_storage=False)
            <= payload["envelope_budget_bytes"]), (
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
    # The straddle from the model's own terms: the hold between radiation
    # calls fits this budget, the priced peak does not.
    assert (_total_from_terms(env, radiation_storage=False) <= budget), (
        "fixture no longer straddles")
    assert phases.peak_envelope_bytes > budget
