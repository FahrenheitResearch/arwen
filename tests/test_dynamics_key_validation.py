"""A key this schema does not know is a wrong answer, not a warning.

The fleet's hostile-input node filed two shapes of the same defect and
both reached ``forecast validity PASS`` at exit 0:

* a *misspelled* dynamics key (``damp_coef`` for ``dampcoef``) was
  dropped with one stderr line, and the run silently used the built-in
  default instead of the value the user typed;
* an *out-of-range* dynamics coefficient (``epssm = 5.0``, ``dampcoef =
  -5.0``) was carried into the solver unexamined and moved 2 m
  temperature by 2.2 K in two simulated hours.

Warn-not-block is the right posture for a run that will almost certainly
work.  Neither of these is that: both deliver a confident number that is
not the number the configuration asked for.  So both refuse, and the
refusal names the key.

Every test here is a regression bar over one of those two shapes; each
one fails against v1.4.0.
"""

from __future__ import annotations

import textwrap

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.experiment import load_experiment

from test_experiment import BASE, _write  # noqa: F401 - shared fixture text


# --------------------------------------------------------------------
# Shape 1: a key the schema does not know
# --------------------------------------------------------------------

def test_misspelled_shared_key_refuses_and_names_itself(tmp_path):
    """``damp_coef`` is a typo for ``dampcoef``; silence would run the
    default 0.2 and call the answer valid."""

    with pytest.raises(ValueError) as caught:
        load_experiment(_write(tmp_path, shared="damp_coef = 0.4"))
    message = str(caught.value)
    assert "damp_coef" in message          # names the key it refused
    assert "[shared]" in message           # names the table
    assert "dampcoef" in message           # names the key it meant


@pytest.mark.parametrize("table,line", [
    ("experiment", "restart_intervals = 0.0"),
    ("shared", "epsm = 0.5"),
    ("domain", "parent_grid_ratios = 3"),
])
def test_unknown_key_refuses_in_every_schema_table(tmp_path, table, line):
    kwargs = {"experiment": "restart_interval_s = 0.0"}
    if table == "experiment":
        kwargs["experiment"] = "restart_interval_s = 0.0\n" + line
    elif table == "shared":
        kwargs["shared"] = line
    else:
        kwargs["d02"] = line
    with pytest.raises(ValueError, match=r"does not have a key"):
        load_experiment(_write(tmp_path, **kwargs))


def test_unknown_top_level_table_refuses(tmp_path):
    """``[dynamics]`` is the LEGACY config's table name.  Pasting it into
    an experiment config used to drop every key in it in one line."""

    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="") + "\n[dynamics]\nepssm = 0.9\n"
    with pytest.raises(ValueError) as caught:
        load_experiment(_write(tmp_path, text=text))
    assert "dynamics" in str(caught.value)


def test_the_refusal_is_one_sentence_with_the_rest_behind_explain(tmp_path):
    """Message-layer contract: action half short, mechanism half layered."""

    from gpuwm.explain import split

    with pytest.raises(ValueError) as caught:
        load_experiment(_write(tmp_path, shared="damp_coef = 0.4"))
    action, why = split(str(caught.value))
    assert why, "the known-key list belongs behind --explain"
    assert action.count("\n") <= 2, action


# --------------------------------------------------------------------
# Shape 2: a known key carrying a value outside the range it is defined on
# --------------------------------------------------------------------

def _run(**overrides) -> RunConfig:
    base = dict(nx=8, ny=8, nz=8, dx=1000.0, dy=1000.0, ztop=12000.0,
                dt=6.0, run_seconds=60.0)
    base.update(overrides)
    return validate_run_config(RunConfig(**base))


def test_the_reference_dynamics_values_still_pass():
    """The guard must not narrow the configurations that already run:
    these are the values ``gpuwm domain`` itself emits."""

    _run(epssm=0.5, smdiv=0.1, emdiv=0.01, damp_opt=3, zdamp=5000.0,
         dampcoef=0.2, w_damping=1, c_s=0.25, time_step_sound=4,
         spec_exp=0.33)
    _run()  # and the bare defaults


@pytest.mark.parametrize("key,value", [
    ("epssm", 5.0),        # a weight in (0, 1]; the fleet's exhibit
    ("epssm", 0.0),
    ("epssm", float("nan")),
    ("dampcoef", -5.0),    # anti-damping: the layer injects energy
    ("dampcoef", float("inf")),
    ("zdamp", -1.0),       # a damping layer of negative depth
    ("zdamp", 0.0),
    ("smdiv", -0.1),
    ("emdiv", -0.01),
    ("c_s", -0.25),
    ("ztop", -20000.0),    # reached PASS in v1.4.0
    ("ztop", 0.0),
    ("time_step_sound", 0),
    ("time_step_sound", -4),
    ("spec_exp", -1.0),
    ("diff_6th_factor", -0.12),
    ("p_surf", 0.0),
    ("base_temp", -290.0),
])
def test_out_of_range_dynamics_value_refuses(key, value):
    with pytest.raises(ValueError) as caught:
        _run(**{key: value})
    assert key in str(caught.value)


@pytest.mark.parametrize("value", [1, 2, 4, -1])
def test_unimplemented_damp_opt_refuses(value):
    """``damp_opt`` 1 and 2 are real WRF choices this tree does not
    implement: every damping site tests ``== 3``, so asking for them
    silently disables the damping layer the user asked for."""

    with pytest.raises(ValueError, match=r"damp_opt"):
        _run(damp_opt=value, zdamp=5000.0, dampcoef=0.2)


@pytest.mark.parametrize("value", [2, -1])
def test_unimplemented_w_damping_refuses(value):
    with pytest.raises(ValueError, match=r"w_damping"):
        _run(w_damping=value)


def test_zdamp_deeper_than_the_model_top_refuses():
    """``_damp_factors`` divides by ``zdamp`` about ``ztop - zdamp``; a
    layer taller than the column damps the whole troposphere."""

    with pytest.raises(ValueError, match=r"zdamp"):
        _run(damp_opt=3, ztop=12000.0, zdamp=20000.0)


def test_the_guard_is_reached_through_the_experiment_path(tmp_path):
    """Both loaders funnel into ``validate_run_config``; the fleet hit
    the experiment one."""

    with pytest.raises(ValueError, match=r"epssm"):
        load_experiment(_write(tmp_path, shared="epssm = 5.0"))


def test_the_guard_is_reached_through_the_legacy_loader(tmp_path):
    from gpuwm.config import load_config

    path = tmp_path / "legacy.toml"
    path.write_text(textwrap.dedent("""\
        [grid]
        nx = 8
        ny = 8
        nz = 8
        dx = 1000.0
        dy = 1000.0
        ztop = 12000.0
        [run]
        dt = 6.0
        run_seconds = 60.0
        [dynamics]
        dampcoef = -5.0
        """))
    with pytest.raises(ValueError, match=r"dampcoef"):
        load_config(path)


# --------------------------------------------------------------------
# The commonest error of all: a path that is not a config
# --------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["missing", "directory", "empty"])
def test_a_wrong_kind_of_config_path_refuses_in_one_sentence(tmp_path, kind):
    """v1.4.0 answered each of these with a different Python traceback
    at exit 1, where every other refusal is a sentence at exit 2."""

    import gpuwm.cli as cli

    if kind == "missing":
        target = tmp_path / "does" / "not" / "exist.toml"
    elif kind == "directory":
        target = tmp_path / "adir"
        target.mkdir()
    else:
        target = tmp_path / "empty.toml"
        target.write_bytes(b"")

    with pytest.raises(ValueError) as caught:
        load_experiment(target)
    assert str(target) in str(caught.value)

    # And through the shipped CLI: rc 2, no traceback.
    assert cli.main(["check", str(target), "--budget-gib", "2",
                     "--vram-gib", "6"]) == 2
