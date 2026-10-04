"""Historical hydraulic SPP: whole native stages, bounds and no-op identity."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.ruc_spp import hydraulic_spp, pattern_inputs, validate_spp_mode
from tools.ruc_spp_wrf_oracle.validate import assess, read_csv, replay

ORACLE = Path(__file__).parents[1] / "gpuwm/data/ruc/spp_oracle"
MANIFEST = json.loads((ORACLE / "manifest.json").read_text(encoding="utf-8"))
RECORDS = [record for record in MANIFEST["records"] if record["variant"] == "hydraulic_overlay"]


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["stage"] + "-" + r["label"])
def test_whole_stage_matches_native_overlay_and_column_permutation(record):
    path = ORACLE / record["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    oracle = read_csv(path)
    actual = replay(record["stage"], oracle, record["mode"])
    assess(record["stage"], oracle, actual)
    order = [3, 0, 2, 1]
    reordered = replay(record["stage"], oracle, record["mode"], order=order)
    assess(record["stage"], oracle, reordered, order=order)
    for name in actual:
        np.testing.assert_array_equal(reordered[name], actual[name][..., order], err_msg=name)


@pytest.mark.parametrize("stage", ["soil", "snowsoil"])
def test_disabled_and_zero_pattern_preserve_every_native_field(stage):
    original = read_csv(ORACLE / f"current-{stage}-off.csv")
    for label in ("off", "zero"):
        oracle = read_csv(ORACLE / f"hydraulic_overlay-{stage}-{label}.csv")
        for name in original:
            if name != "rstochcol":
                np.testing.assert_array_equal(oracle[name].view(np.uint32), original[name].view(np.uint32))
    off = replay(stage, original, 0)
    zero = replay(stage, read_csv(ORACLE / f"hydraulic_overlay-{stage}-zero.csv"))
    changed = replay(stage, read_csv(ORACLE / f"hydraulic_overlay-{stage}-plus03.csv"))
    for name in off:
        np.testing.assert_array_equal(off[name].view(np.uint32), zero[name].view(np.uint32), err_msg=name)
    assert np.any(changed["soilmois"] != zero["soilmois"])
    assert np.any(changed["fieldcol_sf"] != 0)


def test_disabled_inputs_are_not_examined_and_enabled_broadcast_is_read_only():
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError("disabled pattern inspected")
    assert pattern_inputs(0, Poison(), Poison(), (9, 2, 3)) == (None, None)
    pattern = np.broadcast_to(np.asarray([[.3, -.3, 0]], np.float32), (9, 2, 3))
    original = pattern.copy()
    hydro = np.full(pattern.shape, 1e-5, np.float32)
    diagnostic = np.zeros_like(hydro)
    hydraulic_spp(hydro, pattern, diagnostic)
    np.testing.assert_array_equal(pattern, original)
    np.testing.assert_array_equal(hydro, np.float32(1e-5) * (np.float32(1) + original))
    np.testing.assert_array_equal(diagnostic, np.float32(1e-5) * original)


@pytest.mark.parametrize("bad", [None, np.zeros((9, 2), np.float64), np.zeros((6, 2), np.float32),
                                np.full((9, 2), np.nan, np.float32), np.full((9, 2), -1.01, np.float32)])
def test_bad_patterns_fail_before_hydraulic_state_changes(bad):
    hydro = np.ones((9, 2), np.float32)
    before = hydro.copy()
    with pytest.raises(ValueError):
        hydraulic_spp(hydro, bad)
    np.testing.assert_array_equal(hydro, before)


@pytest.mark.parametrize("mode", [True, False, -1, 2, 1.0, "1"])
def test_mode_requires_exact_integer_zero_or_one(mode):
    with pytest.raises(ValueError):
        validate_spp_mode(mode)


def test_mutable_diagnostic_aliases_and_read_only_output_are_refused():
    pattern = np.zeros((9, 2), np.float32)
    hydro = np.ones_like(pattern)
    with pytest.raises(ValueError, match="alias"):
        hydraulic_spp(hydro, pattern, pattern)
    with pytest.raises(ValueError, match="alias"):
        hydraulic_spp(hydro, pattern, hydro)
    with pytest.raises(ValueError, match="alias"):
        hydraulic_spp(pattern, pattern)
    output = np.zeros_like(pattern)
    output.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        hydraulic_spp(hydro, pattern, output)
