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


def _config(tmp_path, *, nx=550, ny=550, tiles=_TILES, name="exp"):
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
        source = "gfs"
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
    # envelope has to be a fraction of the resident one it replaced.
    phases = gate["phases"]
    assert (phases.forecast_envelope_bytes
            < 0.6 * phases.resident_forecast_envelope_bytes)


def test_the_same_config_without_tiles_is_still_refused(tmp_path, card):
    """THE CONTROL, and the reason the test above is not vacuous.

    Byte for byte the same domain with the ``[tiles]`` table removed still
    does not fit this card, and must still be refused.  Without this, a
    gate that had simply stopped refusing anything would pass the
    regression test above.
    """
    gate = go_cli.memory_gate(_plan(_config(tmp_path, tiles="")))

    assert gate["refuse"] is True
    assert gate["phases"].streamed_forecast is False
    assert "EXCEEDS" in gate["verdict"]


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
