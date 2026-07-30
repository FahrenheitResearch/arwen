"""Real-forecast proofs for the legacy-RRTMG wiring (charter items).

Two proof families, both on the real d01 case with
``ra_rrtmg_variant = "rrtmg_legacy"`` (configs/real74_d01_rrtmg_legacy.toml):

* the BOOBY-TRAP trio on a short in-process forecast: every host/NumPy
  compute leaf of the ported chains (dossier section 10) tripwired ->
  the forecast completes without firing (the CUDA engines carry the
  radiation step); the device-kernel layer tripwired -> the tripwire
  FIRES (the kernels are genuinely on the executed path); the batched
  engine entries tripwired -> the forecast fails.  No dead tripwires:
  a control proves the trap fires when called, and every patched name
  must exist.
* the RESTART ROUND-TRIP A/B/C at the CLI level (fresh processes):
  30 min -> restart -> 30 min more == uninterrupted 60 min, byte-equal
  on every serialized array (radt = 12 min -> five radiation calls in
  the window, so the restart crosses live legacy-radiation state).

Granular per-leaf coverage (each _taugbN, each SW stage) lives in the
synthetic-grid gates of tests/test_rrtmg_legacy_wiring.py; here the
module entry points are trapped on the REAL forecast path.  ``taumol``
is the sole caller of the TAUGB_IMPLS table, so trapping it covers the
per-band leaves at this level.
"""

from __future__ import annotations

import os

import dataclasses
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))

requires_bundle = pytest.mark.skipif(
    not (BUNDLE / "era5_grib/era5_19740403.grb").is_file(),
    reason="read-only 1974 reference bundle is absent")

cp = pytest.importorskip("cupy")
try:
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                                    # pragma: no cover
    _HAS_GPU = False
requires_gpu = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA GPU")

LEGACY_TOML = REPO_ROOT / "configs" / "real74_d01_rrtmg_legacy.toml"


def _legacy_cfg(run_seconds):
    from gpuwm.config import load_config, validate_run_config

    cfg = load_config(LEGACY_TOML)
    return validate_run_config(dataclasses.replace(
        cfg, run_seconds=float(run_seconds)))


def _trip(name):
    def _t(*args, **kwargs):
        raise AssertionError(f"tripwire fired: {name}")
    _t.__name__ = f"tripwire_{name}"
    return _t


def _numpy_leaves():
    from gpuwm.core import rrtmg_lw as lw
    from gpuwm.core import rrtmg_mcica as mc
    from gpuwm.core import rrtmg_sw as sw

    leaves = [(lw, n) for n in (
        "inatm", "cldprmc", "setcoef", "taumol", "rtrnmc", "rrtmg_lw")]
    leaves += [(sw, n) for n in (
        "inatm_sw", "setcoef_sw", "taumol_sw", "cldprmc_sw", "reftra_sw",
        "vrtqdr_sw", "spcvmc_sw", "rrtmg_sw")]
    leaves += [(mc, n) for n in (
        "generate_lw_subcolumns", "generate_sw_subcolumns")]
    return leaves


@requires_bundle
@requires_gpu
@pytest.mark.gpu
def test_booby_trap_numpy_leaves_do_not_fire(tmp_path, monkeypatch):
    """Charter proof (a): a short real legacy forecast steps with EVERY
    host/NumPy compute leaf replaced by a raising tripwire -- and none
    fires."""
    from gpuwm.verify.cases import real74_d01 as case

    with pytest.raises(AssertionError, match="tripwire fired: control"):
        _trip("control")()
    for mod, name in _numpy_leaves():
        assert hasattr(mod, name), f"dead tripwire target {name}"
        monkeypatch.setattr(mod, name, _trip(f"{mod.__name__}.{name}"))

    cfg = _legacy_cfg(900.0)   # radt=12 min -> radiation at steps 1, 13
    summary = case.run_config(cfg, tmp_path / "trap-none")
    assert summary.completed_seconds == 900.0
    assert summary.nan_free


@requires_bundle
@requires_gpu
@pytest.mark.gpu
def test_booby_trap_withheld_device_kernels_fire(tmp_path, monkeypatch):
    """Charter proof (b): withholding the device-kernel layer makes the
    tripwire fire -- the kernels are genuinely on the executed path."""
    from gpuwm.core import rrtmg_lw as lw
    from gpuwm.core import rrtmg_sw as sw
    from gpuwm.verify.cases import real74_d01 as case

    monkeypatch.setattr(lw, "_gpu_kernel",
                        _trip("rrtmg_lw._gpu_kernel"))
    monkeypatch.setattr(sw.CudaSW, "_k",
                        _trip("rrtmg_sw.CudaSW._k"))
    cfg = _legacy_cfg(900.0)
    with pytest.raises(Exception, match="tripwire fired"):
        case.run_config(cfg, tmp_path / "trap-kernels")


@requires_bundle
@requires_gpu
@pytest.mark.gpu
def test_booby_trap_trapped_engine_entries_fail(tmp_path, monkeypatch):
    """Charter proof (c): trapping the batched engine entries makes the
    forecast fail."""
    from gpuwm.core import rrtmg_lw as lw
    from gpuwm.core import rrtmg_sw as sw
    from gpuwm.verify.cases import real74_d01 as case

    monkeypatch.setattr(lw, "gpu_rrtmg_lw_batched",
                        _trip("gpu_rrtmg_lw_batched"))
    monkeypatch.setattr(sw.CudaSW, "rrtmg_sw_batched",
                        _trip("rrtmg_sw_batched"))
    cfg = _legacy_cfg(900.0)
    with pytest.raises(Exception, match="tripwire fired"):
        case.run_config(cfg, tmp_path / "trap-engines")


def _write_case_config(tmp_path, name, run_seconds, restart_interval):
    text = LEGACY_TOML.read_text(encoding="utf-8")
    text = text.replace("run_seconds = 43200.0",
                        f"run_seconds = {run_seconds}")
    text += f"restart_interval_s = {restart_interval}\n"
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@requires_bundle
@requires_gpu
@pytest.mark.gpu
def test_legacy_restart_continuation_is_bit_identical(tmp_path):
    """Charter proof: legacy restart identity round-trips -- write +
    restore-in-a-fresh-process + continue == uninterrupted, byte-equal
    on every serialized array, with the legacy algorithm identities in
    the header (radt = 12 min puts five radiation calls inside the
    60-minute window, so live legacy radiation state crosses the
    restart boundary)."""
    import test_restart as tr
    from gpuwm.io import restart

    rst_mid = restart.restart_filename(datetime(1974, 4, 3, 12, 30))
    rst_end = restart.restart_filename(datetime(1974, 4, 3, 13, 0))

    cfg_a = _write_case_config(tmp_path, "legacy-a.toml", 1800.0, 1800.0)
    cfg_b = _write_case_config(tmp_path, "legacy-b.toml", 3600.0, 3600.0)
    cfg_c = _write_case_config(tmp_path, "legacy-c.toml", 3600.0, 1800.0)
    out_a = tmp_path / "run-a"
    out_b = tmp_path / "run-b"
    out_c = tmp_path / "run-c"

    tr._cli_run(cfg_a, out_a)
    tr._cli_run(cfg_c, out_c)
    result_b = tr._cli_run(cfg_b, out_b, restart_file=out_a / rst_mid)
    assert "'restarted': True" in result_b.stdout

    # Write-time identity, then THE continuation gate.
    tr._assert_restart_equal(out_a / rst_mid, out_c / rst_mid,
                             compare_trackers=True)
    tr._assert_restart_equal(out_b / rst_end, out_c / rst_end,
                             compare_trackers=True)

    # The header carries the legacy identities, not the RTE+RRTMGP ones.
    header = restart.read_restart_header(out_c / rst_end)
    blob = str(header)
    assert "wrf-v4.6.1-rrtmg-legacy-lw-v1" in blob
    assert "wrf-v4.6.1-rrtmg-legacy-sw-v1" in blob
    assert "rte-rrtmgp-v1" not in blob
