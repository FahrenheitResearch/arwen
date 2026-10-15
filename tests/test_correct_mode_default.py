"""Correct mode is the engine default (2.9.0, fixed means default).

CPU tests of the switch (gpuwm/correct_mode.py): the default is correct, exact
is selectable and authors every key off, a strict WRF process
(GPUWM_WRF_EXACT=1) forces exact and records it, a run receipt names each
domain's mode, a key outside its scheme moves no echo or digest, and a 2.8.9
checkpoint resumes only in exact mode, with a notice that says how.

THE BREAKAGE THIS PREVENTS.  Without these, the default could fall back to
exact (a bare default run would show the WB4/WB5 defects again: "fixed" would
mean opt-in), a strict 0 ULP proof could run corrected arithmetic under its
strict label, or the flip could move the digest of every run that never reads
the keys and miss every prepared cache and checkpoint after the upgrade.

The device budget tests (each correction fails in exact and closes in correct)
are tests/test_correct_mode_water_budget.py, on the GPU shard.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gpuwm import correct_mode
from gpuwm.config import RunConfig, validate_run_config

ROOT = Path(__file__).resolve().parents[1]

# -------------------------------------------------------------- the switch --
def test_exact_mode_authors_every_key_off():
    """Exact authors every key False, so it is the matched arithmetic
    whatever the RunConfig defaults are (they are on from 2.9.0)."""
    res = correct_mode.resolve("exact")
    assert res.settings == {"thompson_cloud_surface_booking": False,
                            "ntiedtke_dry_mass_coupling": False,
                            "wsm6_conservative_surface": False,
                            "wdm6_conservative_surface": False}
    assert res.applied == ()
    assert "forced_by" not in res.receipt()


def test_correct_mode_sets_every_registered_key():
    res = correct_mode.resolve("correct")
    assert res.settings == {"thompson_cloud_surface_booking": True,
                            "ntiedtke_dry_mass_coupling": True,
                            "wsm6_conservative_surface": True,
                            "wdm6_conservative_surface": True}
    assert res.receipt()["applied_findings"] == ["WB1", "WB4", "WB5"]


def test_correct_skip_drops_one_correction():
    res = correct_mode.resolve("correct", ["WB5"])
    assert res.settings == {"thompson_cloud_surface_booking": True,
                            "ntiedtke_dry_mass_coupling": True,
                            "wsm6_conservative_surface": False,
                            "wdm6_conservative_surface": False}
    assert res.receipt()["physics.correct_skip"] == ["WB5"]


def test_the_default_is_correct_in_a_default_process():
    """2.9.0 (fixed means default): an unnamed mode is correct; a strict
    WRF process (GPUWM_WRF_EXACT=1) is covered by the subprocess test."""
    if correct_mode.STRICT:  # pragma: no cover - strict runner
        pytest.skip("strict process; see the strict subprocess test")
    assert correct_mode.PHYSICS_MODE_DEFAULT == "correct"
    assert correct_mode.resolve() == correct_mode.resolve("correct")
    assert correct_mode.resolve(None, ["WB4"]).applied == ("WB1", "WB5")


def test_strict_wrf_mode_forces_exact_and_records_it():
    """GPUWM_WRF_EXACT=1 is a 0 ULP proof of WRF's arithmetic: the default
    resolves to exact with forced_by, the RunConfig keys default off, and a
    named correct mode or an on key is refused by name."""
    import json
    import os
    import subprocess
    import sys
    # Strict mode installs its NVRTC hook at ``import gpuwm``
    # (gpuwm/wrf_exact.py install, sealed by the assembled-legacy proofs),
    # and that hook imports cupy.cuda.compiler.  On a host without CuPy (the
    # public CI cpu job, node-1's Stage 1) the probe hands it a stand-in
    # compiler whose two hooked entry points are never called: this test
    # reads the mode resolution only, and no kernel is compiled.
    stub = (
        "import importlib.util, sys, types\n"
        "if importlib.util.find_spec('cupy') is None:\n"
        "    class _Program:\n"
        "        def compile(self, options=(), log_stream=None):\n"
        "            raise RuntimeError('no CuPy on this host')\n"
        "    compiler = types.ModuleType('cupy.cuda.compiler')\n"
        "    compiler._compile_with_cache_cuda = _Program().compile\n"
        "    compiler._NVRTCProgram = _Program\n"
        "    cuda = types.ModuleType('cupy.cuda'); cuda.compiler = compiler\n"
        "    cupy = types.ModuleType('cupy'); cupy.cuda = cuda\n"
        "    sys.modules.update({'cupy': cupy, 'cupy.cuda': cuda,\n"
        "                        'cupy.cuda.compiler': compiler})\n")
    probe = stub + (
        "import json\n"
        "from gpuwm import correct_mode as m\n"
        "from gpuwm.config import RunConfig, validate_run_config\n"
        "out = {'default': m.resolve().receipt(),\n"
        "       'key_default': RunConfig.wsm6_conservative_surface}\n"
        "try:\n"
        "    m.resolve('correct'); out['correct'] = 'accepted'\n"
        "except ValueError as e:\n"
        "    out['correct'] = str(e)\n"
        "cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,\n"
        "                dt=1.0, run_seconds=10.0, moist=True, mp_physics=6,\n"
        "                wsm6_conservative_surface=True)\n"
        "try:\n"
        "    validate_run_config(cfg); out['key_on'] = 'accepted'\n"
        "except ValueError as e:\n"
        "    out['key_on'] = str(e)\n"
        "print(json.dumps(out))\n")
    env = dict(os.environ, GPUWM_WRF_EXACT="1", GPUWM_NO_LOCAL_GPU="1")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, env=env, cwd=str(ROOT), timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["default"]["physics.mode"] == "exact"
    assert out["default"]["forced_by"] == "GPUWM_WRF_EXACT=1"
    assert out["default"]["applied_findings"] == []
    assert out["key_default"] is False
    assert "strict WRF process" in out["correct"]
    assert "strict WRF process" in out["key_on"]


@pytest.mark.parametrize("mode", ["Correct", "", "on", "default", 1])
def test_unknown_mode_is_refused(mode):
    with pytest.raises(ValueError, match="physics mode"):
        correct_mode.resolve(mode)


def test_unknown_or_repeated_skip_is_refused():
    with pytest.raises(ValueError, match="not a correction"):
        correct_mode.resolve("correct", ["WB9"])
    with pytest.raises(ValueError, match="twice"):
        correct_mode.resolve("correct", ["WB5", "WB5"])
    with pytest.raises(ValueError, match="exact"):
        correct_mode.resolve("exact", ["WB5"])


_TINY = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
             dt=1.0, run_seconds=10.0, moist=True, mp_physics=6)


def test_runconfig_defaults_are_correct_mode_and_exact_is_selectable():
    """2.9.0 declared default change (plan call 1): the RunConfig keys
    default on; naming exact gives back the matched arithmetic."""
    cfg = validate_run_config(RunConfig(**_TINY))
    assert cfg.wsm6_conservative_surface is correct_mode.CORRECT_KEYS_DEFAULT
    assert cfg.ntiedtke_dry_mass_coupling is correct_mode.CORRECT_KEYS_DEFAULT
    exact = validate_run_config(
        RunConfig(**_TINY, **correct_mode.resolve("exact").settings))
    assert exact.wsm6_conservative_surface is False
    assert exact.ntiedtke_dry_mass_coupling is False
    assert correct_mode.resolution_of(exact).mode == "exact"
    if not correct_mode.STRICT:
        assert correct_mode.resolution_of(cfg).receipt() == (
            correct_mode.resolve("correct").receipt())


def test_wb5_spans_wsm6_and_wdm6_and_a_one_key_arm_reads_as_itself():
    """WB5 is one finding in two schemes (mp_wdm6.F carries WSM6's
    sedimentation routine).  Each key is read by its own scheme only, and a
    configuration that turns one of them off reads back as what it is."""
    assert correct_mode.CORRECTIONS_BY_FINDING["WB5"].keys == (
        "wsm6_conservative_surface", "wdm6_conservative_surface")
    assert correct_mode.key_in_scope("wdm6_conservative_surface",
                                      {"mp_physics": 16})
    assert not correct_mode.key_in_scope("wdm6_conservative_surface",
                                          {"mp_physics": 6})
    arm = RunConfig(**{**_TINY, "mp_physics": 16},
                    **{**correct_mode.resolve("correct").settings,
                       "wsm6_conservative_surface": False})
    receipt = correct_mode.resolution_of(arm).receipt()
    assert receipt["applied_findings"] == ["WB1", "WB4", "WB5"]
    assert receipt["keys"] == {"thompson_cloud_surface_booking": True,
                               "ntiedtke_dry_mass_coupling": True,
                               "wdm6_conservative_surface": True}


def test_run_receipt_names_each_domains_mode():
    exact = RunConfig(**_TINY, **correct_mode.resolve("exact").settings)
    arm = RunConfig(**_TINY, **correct_mode.resolve("correct", ["WB4"]).settings)
    receipt = correct_mode.run_receipt({2: arm, 1: exact})
    assert list(receipt["domains"]) == ["1", "2"]
    assert receipt["domains"]["1"]["physics.mode"] == "exact"
    assert receipt["domains"]["2"]["applied_findings"] == ["WB1", "WB5"]
    assert receipt["domains"]["2"]["physics.correct_skip"] == ["WB4"]
    assert receipt["default"] == correct_mode.PHYSICS_MODE_DEFAULT


def test_run_receipt_says_which_corrections_reached_the_run():
    """A correct-mode domain lists every key on, but only the keys its
    schemes read are in effect: WSM6 without New Tiedtke is WB5 alone, a
    Thompson domain is WB1 alone, Morrison is none (the matched arithmetic),
    and an exact domain is none.  Breakage prevented: a receipt that reads
    as if WB4 and WB5 changed a Thompson-only run."""
    correct = correct_mode.resolve("correct").settings
    cases = (
        ({"mp_physics": 6}, ["WB5"], ["wsm6_conservative_surface"]),
        ({"mp_physics": 16}, ["WB5"], ["wdm6_conservative_surface"]),
        ({"mp_physics": 8}, ["WB1"], ["thompson_cloud_surface_booking"]),
        ({"mp_physics": 10}, [], []),
        ({"mp_physics": 6, "cu_physics": 16}, ["WB4", "WB5"],
         ["ntiedtke_dry_mass_coupling", "wsm6_conservative_surface"]),
    )
    for schemes, findings, keys in cases:
        cfg = RunConfig(**{**_TINY, **schemes}, **correct)
        block = correct_mode.run_receipt({1: cfg})["domains"]["1"]
        assert block["physics.mode"] == "correct"
        assert block["applied_findings"] == ["WB1", "WB4", "WB5"]
        assert block["in_effect"] == {"findings": findings, "keys": keys}, schemes
    exact = RunConfig(**_TINY, **correct_mode.resolve("exact").settings)
    assert correct_mode.run_receipt({1: exact})["domains"]["1"]["in_effect"] == {
        "findings": [], "keys": []}


def test_a_key_outside_its_scheme_does_not_move_the_echo_or_digest():
    """A default run whose schemes never read a correction key (no WSM6, no
    New Tiedtke) echoes and digests as 2.8.9 did; under WSM6 the on key
    binds, and an exact WSM6 run echoes as 2.8.9 did."""
    from gpuwm.io.restart import configuration_echo
    thompson = RunConfig(**{**_TINY, "mp_physics": 8})
    assert "wsm6_conservative_surface" not in configuration_echo(thompson)
    assert "ntiedtke_dry_mass_coupling" not in configuration_echo(thompson)
    wsm6 = RunConfig(**_TINY)
    echo = configuration_echo(wsm6)
    assert echo.get("wsm6_conservative_surface") is (
        True if correct_mode.CORRECT_KEYS_DEFAULT else None)
    assert "ntiedtke_dry_mass_coupling" not in echo
    exact = RunConfig(**_TINY, **correct_mode.resolve("exact").settings)
    assert "wsm6_conservative_surface" not in configuration_echo(exact)


def test_a_289_checkpoint_resumes_only_in_exact_mode_and_says_how():
    """Absent on restore = exact: a header without the key (every 2.8.9
    WSM6 checkpoint) meets the 2.9.0 default and is refused with the notice
    naming the remedy; naming exact resumes it."""
    if correct_mode.STRICT:  # pragma: no cover - strict runner
        pytest.skip("strict process forces exact; nothing flips")
    from gpuwm.io.restart import (RestartMismatchError, _require_config_match,
                                  configuration_echo)
    exact = RunConfig(**_TINY, **correct_mode.resolve("exact").settings)
    stored = configuration_echo(exact)
    assert "wsm6_conservative_surface" not in stored
    with pytest.raises(RestartMismatchError) as refused:
        _require_config_match(stored, RunConfig(**_TINY), "ckpt.nc")
    assert "wsm6_conservative_surface" in str(refused.value)
    assert "exact mode" in str(refused.value)
    _require_config_match(stored, exact, "ckpt.nc")
    # A Thompson checkpoint is untouched by the flip: the keys reach nothing.
    thompson = RunConfig(**{**_TINY, "mp_physics": 8})
    _require_config_match(
        configuration_echo(RunConfig(**{**_TINY, "mp_physics": 8},
                                     **correct_mode.resolve("exact").settings)),
        thompson, "ckpt.nc")


def test_runconfig_refuses_a_truthy_non_bool():
    with pytest.raises(ValueError, match="wsm6_conservative_surface"):
        validate_run_config(replace(RunConfig(**_TINY), wsm6_conservative_surface="false"))


def test_a_wrf_namelist_can_name_exact_mode(tmp_path):
    """The namelist door: a stock WRF namelist has no spelling for the
    correct-mode keys, so the selector comment carries them as booleans.
    Without it an imported namelist (a WRF parity arm) could only run the
    2.9.0 default, correct mode, against WRF's arithmetic."""
    import tomllib

    from gpuwm.experiment import build_experiment
    from gpuwm.namelist_import import import_namelists
    from gpuwm.physics_source_defaults import (
        exact_mode_selectors, read_physics_selector_comment,
        with_physics_selector_comment)

    fixture = ROOT / "tests" / "fixtures" / "source_requests"
    text = (fixture / "hrrr_wrf.nl.c18c").read_text()
    carried = with_physics_selector_comment(text, exact_mode_selectors())
    assert read_physics_selector_comment(carried) == exact_mode_selectors()
    namelist = tmp_path / "namelist.input"
    namelist.write_text(carried)
    resolved, _ = import_namelists(fixture / "hrrr_namelist.wps.c18", namelist)
    run = build_experiment(tomllib.loads(resolved), source="exact").root.run
    assert correct_mode.resolution_of(run).mode == "exact"
    for key in correct_mode.CORRECTION_KEYS:
        assert getattr(run, key) is False
    with pytest.raises(ValueError, match="correct-mode key"):
        read_physics_selector_comment(
            '! gpuwm-physics-selectors-v1: {"wsm6_conservative_surface": "false"}')


def test_wb1_is_output_only_and_moves_no_thompson_identity():
    """WB1 books CLOUDNC and changes no prognostic word, so its key is an
    output-only switch: it binds no checkpoint, digest, fingerprint or
    prepared cache, and a default (correct) Thompson run echoes exactly as
    an exact one.  Without this every 2.8.9 Thompson checkpoint and prepared
    tree would be refused or missed for a field that changes nothing."""
    import dataclasses

    from gpuwm.checkpoint_identity import CONFIG_DIAGNOSTIC_FIELDS
    from gpuwm.io.restart import (_configuration_digest_values,
                                  _require_config_match, configuration_echo)
    assert set(correct_mode.OUTPUT_ONLY_KEYS) <= CONFIG_DIAGNOSTIC_FIELDS
    thompson = RunConfig(**{**_TINY, "mp_physics": 8})
    exact = RunConfig(**{**_TINY, "mp_physics": 8},
                      **correct_mode.resolve("exact").settings)
    assert configuration_echo(thompson) == configuration_echo(exact)
    assert (_configuration_digest_values(dataclasses.asdict(thompson))
            == _configuration_digest_values(dataclasses.asdict(exact)))
    _require_config_match(configuration_echo(exact), thompson, "ckpt.nc")
    if not correct_mode.STRICT:
        assert thompson.thompson_cloud_surface_booking is True
        assert "WB1" in correct_mode.resolution_of(thompson).applied


def test_wb1_slots_are_priced_and_carried_only_when_booking():
    """The preflight prices the two CLOUDNC slots exactly when correct-mode
    Thompson books them, and classifies them as carried state (restart-
    serialized, never on the shared arena); exact mode prices nothing new."""
    from gpuwm.core import preflight as pf
    from gpuwm.ensemble.batch_physics import thompson_column_scratch_fields
    on = RunConfig(**{**_TINY, "mp_physics": 8},
                   thompson_cloud_surface_booking=True)
    off = RunConfig(**{**_TINY, "mp_physics": 8},
                    thompson_cloud_surface_booking=False)
    shape = (on.ny, on.nx)
    assert pf.scratch_slot_registry(on)["mp_cloudnc"] == shape
    assert pf.scratch_slot_registry(on)["mp_cloudncv"] == shape
    assert "mp_cloudnc" not in pf.scratch_slot_registry(off)
    assert pf.scratch_slot_lifetime("mp_cloudnc").kind == "carrying"
    names = {f.name for f in thompson_column_scratch_fields(
        nz=on.nz, ny=on.ny, nx=on.nx, cloud_booking=True)}
    assert {"mp_cloudnc", "mp_cloudncv"} <= names
    assert not {"mp_cloudnc", "mp_cloudncv"} & {
        f.name for f in thompson_column_scratch_fields(
            nz=on.nz, ny=on.ny, nx=on.nx)}
    from gpuwm.io.restart import SERIALIZED_SCRATCH_SLOTS
    assert {"mp_cloudnc", "mp_cloudncv"} <= SERIALIZED_SCRATCH_SLOTS
