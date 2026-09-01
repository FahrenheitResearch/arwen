"""``StreamingRefused`` reaches the user as a refusal, not a traceback.

Named breakage: :class:`gpuwm.core.streaming.StreamingRefused` subclasses
``RuntimeError``, and the CLI boundary's ``RuntimeError`` clause re-raised
anything it did not recognise -- so a ``[tiles] mode = "on"`` config over
a nested tree, which ``build_experiment`` refuses at config validation
(``gpuwm/experiment.py``, via ``refuse_streamed_nests``), escaped every
door except run-plan as a ~20-line Python traceback with the remedy
paragraph buried at the bottom.  Reproduced on the published 2.5.8 and
2.6.0 wheels.  The boundary now prints the refusal on the same contract
as every documented refusal: the message, exit 2, no traceback, remedy
intact.
"""

from __future__ import annotations

import textwrap

from gpuwm import cli


def _nested_tiles_on_config(tmp_path):
    """A nested tree whose ``[tiles] mode = "on"`` the loader refuses."""
    path = tmp_path / "nested_tiles_on.toml"
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

        [tiles]
        mode = "on"
        tile_nx = 96
        tile_ny = 96

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = 120
        ny = 120
        time_step = 20
        dx = 3000.0
        history_interval_s = 3600.0

        [[domain]]
        grid_id = 2
        parent_id = 1
        i_parent_start = 30
        j_parent_start = 30
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        nx = 90
        ny = 90
        history_interval_s = 3600.0
        """), encoding="utf-8")
    return path


def test_check_prints_the_streaming_refusal_and_exits_2(tmp_path, capsys):
    path = _nested_tiles_on_config(tmp_path)
    code = cli.main(["check", str(path)])
    err = capsys.readouterr().err
    assert code == 2, err
    # The core's own sentence, verbatim subject matter: the refusal names
    # the concrete breakage (a coupling edge with both ends streamed).
    assert "BOTH ends streamed" in err
    # The remedy paragraph survives at the boundary instead of being
    # buried under stack frames.
    assert "remedy" in err
    # A refusal is one message, never a traceback.
    assert "Traceback (most recent call last)" not in err


def test_the_refusal_text_is_the_loaders_own(tmp_path, capsys):
    """The door relays the message; it does not restate it."""
    import pytest

    from gpuwm.core.streaming import StreamingRefused
    from gpuwm.domain_wizard import experiment_from_text

    path = _nested_tiles_on_config(tmp_path)

    with pytest.raises(StreamingRefused) as caught:
        experiment_from_text(path.read_text(encoding="utf-8"),
                             source=str(path))
    code = cli.main(["check", str(path)])
    err = capsys.readouterr().err
    assert code == 2
    # The first (user-facing) layer of the loader's sentence is what the
    # door prints -- gpuwm.explain.layered's boundary picks it, exactly
    # as the ValueError refusals are printed.
    from gpuwm.explain import split

    assert split(str(caught.value))[0] in err
