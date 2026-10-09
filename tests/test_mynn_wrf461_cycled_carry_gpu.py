"""Cycled stock MYNN against WRF 4.6.1 with only the lost carry repaired.

The referee keeps QKE, QC_BL and CLDFRA_BL on a cycled start. Everything
else is stock. Raw words, including signed zeros, must match in both modes.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
import _mynn_families as F

CASES = ("empty", "below", "equal", "above", "high", "mixed")


@pytest.fixture(scope="session")
def mynn_extended_oracles(tmp_path_factory):
    """Build the real referees when no lane-local fixture directory is set."""
    built = None

    def path(name):
        nonlocal built
        if root := os.environ.get("MYNN_CARRY_FIXTURES"):
            return Path(root) / name
        if built is None:
            source = os.environ.get("WRF_SOURCE_ROOT")
            if source is None:
                pytest.skip("set WRF_SOURCE_ROOT to pinned WRF v4.6.1, or MYNN_CARRY_FIXTURES")
            built = tmp_path_factory.mktemp("mynn-carry-referee")
            script = Path(__file__).parents[1] / "tools/mynn_pbl_wrf461_oracle/build_cycled.sh"
            with (built / "build.log").open("w") as log:
                subprocess.run(["bash", str(script), source, str(built)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        return built / "fixtures" / name

    return path


def _driver(values, **kwargs):
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda
    return mynn_bl_driver_cuda(
        {name: cp.asarray(np.ascontiguousarray(value))
         for name, value in values.items()}, bl_mynn_version="wrf_461",
        **kwargs)


@pytest.mark.gpu
@requires_gpu
def test_cycled_stock_start_accepts_carried_fields():
    values = F.step_inputs(F.load(1)[1])
    values["qke"].fill(np.float32(0.9))
    values["qc_bl"].fill(np.float32(3e-4))
    values["cldfra_bl"].fill(np.float32(0.6))
    _driver(values, initflag=1, delt=20.0, cycling=True, flag_qs=True)


def _load(case, mixlength, path):
    with gzip.open(path, "rb") as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream, "ascii")))
    steps = {}
    for step in sorted({int(row["step"]) for row in rows}):
        steps[step] = [[row for row in rows
                        if int(row["step"]) == step and row["case"] == family]
                       for family in F.FAMILIES]
        assert all(len(block) == F.NZ for block in steps[step])
    assert len(steps) == 24
    first = F.step_inputs(steps[1])
    threshold = np.float32(0.0002)
    expected = {"empty": np.float32(0),
                "below": np.nextafter(threshold, np.float32(0)),
                "equal": threshold,
                "above": np.nextafter(threshold, np.float32(1)),
                "high": np.float32(0.9), "mixed": threshold}[case]
    assert first["qke"][:, 0].max().view(np.uint32) == expected.view(np.uint32)
    if case != "empty":
        assert (first["qc_bl"] > 0).all()
        assert (first["cldfra_bl"] > 0).all()
    return steps


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mixlength", (1, 2))
@pytest.mark.parametrize("replay", (True, False), ids=("replay", "free"))
def test_cycled_stock_matches_repaired_fortran(case, mixlength, replay, mynn_extended_oracles):
    import cupy as cp
    results = F.integrate(_driver, mixlength, replay=replay, cycling=True,
                          steps=_load(case, mixlength, mynn_extended_oracles(
                              f"driver-carry-{case}-{mixlength}.csv.gz")),
                          to_host=cp.asnumpy)
    table = F.bit_mismatch_table(results)
    different = sum(sum(counts) for counts in table.values())
    words = sum(value.size for _, _, want in results for value in want.values())
    hashes = [{"step": step,
               "actual": {name: hashlib.sha256(value.tobytes()).hexdigest()
                          for name, value in got.items() if name in want},
               "expected": {name: hashlib.sha256(value.tobytes()).hexdigest()
                            for name, value in want.items()}}
              for step, got, want in results]
    if root := os.environ.get("MYNN_CARRY_RECEIPTS"):
        mode = "replay" if replay else "free"
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f"carry-{case}-{mixlength}-{mode}.json").write_text(
            json.dumps({"case": case, "mixlength": mixlength, "mode": mode,
                        "exact": os.environ.get("GPUWM_WRF_EXACT", "0"),
                        "words": words, "different_words": different,
                        "fields": table, "hashes": hashes}, sort_keys=True))
    assert different == 0, (table, F.max_ulp_table(results))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mixlength", (1, 2))
def test_cold_forecast_full_field_hashes_unchanged(tmp_path, monkeypatch, mixlength):
    """Compare a real 40-step RK forecast to a pre-repair checkpoint.

    The baseline is generated on the same card and arithmetic mode before
    applying the carry repair. No archived model output is required in Git.
    """
    if not (root := os.environ.get("MYNN_COLD_BASELINE")):
        pytest.skip("set MYNN_COLD_BASELINE to pre-repair cold RK checkpoints")
    import test_mynn_pbl_runtime as fixture
    from gpuwm.core.dycore import run_steps
    from gpuwm.io.restart import write_restart
    ctor = fixture.RunConfig
    monkeypatch.setattr(fixture, "RunConfig", lambda **kw: ctor(
        **kw, bl_mynn_version="wrf_461", bl_mynn_mixlength=mixlength,
        cycling=False))
    state, cfg, _ = fixture._build()
    run_steps(state, cfg, 40)
    actual_path = write_restart(tmp_path / "cold.npz", state, cfg)
    rows = {}
    with np.load(Path(root) / str(mixlength) / "reference.npz",
                 allow_pickle=False) as before, np.load(actual_path,
                 allow_pickle=False) as after:
        assert set(before.files) == set(after.files)
        assert np.any(before["fields/qke"] > np.float32(0.01))
        assert np.any(before["fields/exch_h"] > np.float32(0))
        for name in before.files:
            if name == "__gpuwm_restart_header__":
                continue
            left, right = before[name], after[name]
            assert left.shape == right.shape and left.dtype == right.dtype
            assert np.isfinite(left).all() and np.isfinite(right).all(), name
            rows[name] = {"shape": list(left.shape), "dtype": str(left.dtype),
                          "bytes": left.nbytes,
                          "before": hashlib.sha256(left.tobytes()).hexdigest(),
                          "after": hashlib.sha256(right.tobytes()).hexdigest()}
    different = [name for name, row in rows.items()
                 if row["before"] != row["after"]]
    if receipt_root := os.environ.get("MYNN_CARRY_RECEIPTS"):
        (Path(receipt_root) / f"cold-forecast-{mixlength}.json").write_text(
            json.dumps({"steps": 40, "different_fields": different,
                        "full_field_hashes": rows}, sort_keys=True))
    assert different == [], different
