"""``[output]`` as a config surface: both TOML routes, and the tree.

The grammar itself is ``tests/test_history_selection.py``.  This file is
about REACHABILITY -- that the block is admitted by the loaders every
front door shares, refused key by key when it is misspelled, resolvable
per domain, and absent from the restart identity.
"""
from __future__ import annotations

import textwrap

import pytest

from gpuwm.config import load_config, load_history_selection
from gpuwm.experiment import load_experiment
from gpuwm.io import history_selection as hs


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# ------------------------------------------------- the RunConfig TOML route

_RUN_CONFIG = """
    [grid]
    nx = 40
    ny = 30
    nz = 20
    dx = 3000.0
    dy = 3000.0
    ztop = 20000.0

    [dynamics]
    moist = true

    [run]
    dt = 15.0
    run_seconds = 60.0
    output_interval_s = 60.0
"""


def test_runconfig_toml_admits_the_output_table(tmp_path):
    path = _write(tmp_path, "run.toml", _RUN_CONFIG + """
    [output]
    preset = "minimal"
    """)
    load_config(path)                     # no unknown-table refusal
    assert load_history_selection(path).preset == "minimal"


def test_runconfig_toml_without_output_reads_the_full_default(tmp_path):
    path = _write(tmp_path, "run.toml", _RUN_CONFIG)
    assert load_history_selection(path) is hs.FULL


def test_a_misspelled_output_key_refuses_at_load_config(tmp_path):
    """A typo may not silently write the full inventory under your name."""
    path = _write(tmp_path, "run.toml", _RUN_CONFIG + """
    [output]
    history_dropp = ["QCLOUD"]
    """)
    with pytest.raises(ValueError) as excinfo:
        load_config(path)
    assert "history_dropp" in str(excinfo.value)
    assert "history_drop" in str(excinfo.value)


# ------------------------------------------------ the experiment TOML route

_EXPERIMENT = """
    [experiment]
    name = "output-surface"
    start_time = 2020-05-01T00:00:00
    run_seconds = 120.0
    restart_interval_s = 120.0

    [projection]
    map_proj = "lambert"
    ref_lat = 35.0
    ref_lon = -97.0
    truelat1 = 30.0
    truelat2 = 40.0
    stand_lon = -97.0

    [shared]
    nz = 20
    ztop = 20000.0
    moist = true
    mp_physics = 0
    ra_physics = 0
    bl_pbl_physics = 0
    sf_surface_physics = 0
    cu_physics = 0

    [[domain]]
    grid_id = 1
    parent_id = 0
    i_parent_start = 1
    j_parent_start = 1
    parent_grid_ratio = 1
    parent_time_step_ratio = 1
    history_interval_s = 60.0
    e_we = 61
    e_sn = 61
    dx = 3000.0
    dy = 3000.0
    time_step = 15

    [[domain]]
    grid_id = 2
    parent_id = 1
    i_parent_start = 11
    j_parent_start = 11
    parent_grid_ratio = 3
    parent_time_step_ratio = 3
    history_interval_s = 60.0
    e_we = 31
    e_sn = 31
"""


def test_experiment_toml_without_output_is_full_on_every_domain(tmp_path):
    exp = load_experiment(_write(tmp_path, "exp.toml", _EXPERIMENT))
    assert exp.output is hs.FULL
    for dc in exp.domains:
        assert dc.output is None
        assert hs.resolve(exp.output, dc.output).writes_everything


def test_tree_wide_output_applies_to_every_domain(tmp_path):
    path = _write(tmp_path, "exp.toml", _EXPERIMENT + """
    [output]
    history_drop = ["QCLOUD", "QRAIN"]
    """)
    exp = load_experiment(path)
    assert exp.output.history_drop == ("QCLOUD", "QRAIN")
    for dc in exp.domains:
        selection = hs.resolve(exp.output, dc.output)
        assert selection.dropped(("QCLOUD", "T2")) == ("QCLOUD",)


def test_a_domain_overrides_the_tree_wide_table_entirely(tmp_path):
    """Keep the parent whole; trim the child that writes the bytes."""
    text = _EXPERIMENT.replace(
        "    e_we = 31\n    e_sn = 31\n",
        '    e_we = 31\n    e_sn = 31\n    output = { preset = "minimal" }\n')
    exp = load_experiment(_write(tmp_path, "exp.toml", text))
    d01, d02 = exp.domains
    assert hs.resolve(exp.output, d01.output).writes_everything
    assert hs.resolve(exp.output, d02.output).preset == "minimal"


def test_a_domain_output_that_is_not_a_table_refuses(tmp_path):
    text = _EXPERIMENT.replace(
        "    e_we = 31\n    e_sn = 31\n",
        '    e_we = 31\n    e_sn = 31\n    output = "minimal"\n')
    with pytest.raises(ValueError) as excinfo:
        load_experiment(_write(tmp_path, "exp.toml", text))
    assert "inline TABLE" in str(excinfo.value)
    assert "grid_id = 2" in str(excinfo.value)


def test_a_domain_output_refusal_names_the_domain(tmp_path):
    text = _EXPERIMENT.replace(
        "    e_we = 31\n    e_sn = 31\n",
        '    e_we = 31\n    e_sn = 31\n'
        '    output = { history_drop = ["NOSUCHVAR"] }\n')
    with pytest.raises(ValueError) as excinfo:
        load_experiment(_write(tmp_path, "exp.toml", text))
    message = str(excinfo.value)
    assert "NOSUCHVAR" in message
    assert "grid_id = 2" in message


def test_the_plan_time_dependency_warning_prints_at_config_resolution(
        tmp_path, capsys):
    """Config resolution, not render time, is where a lost product is said."""
    path = _write(tmp_path, "exp.toml", _EXPERIMENT + """
    [output]
    history_drop = ["REFL_10CM"]
    """)
    capsys.readouterr()
    load_experiment(path)
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "REFL_10CM" in err
    assert "refl" in err
    assert "composite_reflectivity" in err
    # Once per domain, each naming its own domain.
    assert "d01" in err and "d02" in err


def test_a_full_default_experiment_warns_about_nothing(tmp_path, capsys):
    capsys.readouterr()
    load_experiment(_write(tmp_path, "exp.toml", _EXPERIMENT))
    assert "[output]" not in capsys.readouterr().err


# ------------------------------------------------------- restart identity

def test_the_selection_is_absent_from_the_restart_identity(tmp_path):
    """A trimmed run must resume a full run's checkpoints, and back.

    The history tape and the checkpoint stream are different files:
    ``gpuwm.io.restart`` writes model state, never the history frame.
    Binding the selection would refuse exactly the operation the surface
    exists for -- a run that filled its disk resuming with a trimmed
    tape.
    """
    from gpuwm.core.model import restart_identity_payload

    full = load_experiment(_write(tmp_path, "full.toml", _EXPERIMENT))
    trimmed_text = _EXPERIMENT + """
    [output]
    preset = "minimal"
    """
    trimmed_text = trimmed_text.replace(
        "    e_we = 31\n    e_sn = 31\n",
        '    e_we = 31\n    e_sn = 31\n    output = { preset = "severe" }\n')
    trimmed = load_experiment(_write(tmp_path, "trim.toml", trimmed_text))
    assert restart_identity_payload(full) == restart_identity_payload(trimmed)


def test_the_checkpoint_writer_has_no_seam_to_the_history_selection():
    """The two streams are separate files, and separate code.

    A checkpoint is written from MODEL STATE by ``gpuwm.io.restart``;
    the history tape is written from an assembled frame by
    ``gpuwm.io.wrfout``.  If the selection ever reached the checkpoint
    writer, a trimmed run would write a checkpoint it could not resume
    from -- silently, because a restart reads back exactly what it
    wrote.  The absence of the seam is the guarantee, so it is asserted
    directly rather than inferred from a round trip that would pass
    either way.
    """
    from pathlib import Path

    import gpuwm.io.restart as restart_module

    source = Path(restart_module.__file__).read_text(encoding="utf-8")
    assert "history_selection" not in source
    assert "HistorySelection" not in source
