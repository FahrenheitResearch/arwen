"""TEMPO column oracle: the GPU runtime is TEMPO's Fortran, word for word.

TEMPO_ORACLE_DIR names the directory tools/tempo_ref/oracle/build_reference.sh
wrote (unmodified TEMPO 9adf9ef built with its own gfortran flags):
columns_in.bin and columns_catalog.json (127 columns over convective, stable
night, snow and ice, ocean and high-terrain regimes plus edge cases, at dt 20,
60 and 150 s), ref_out.bin (TEMPO's tempo_run on them) and tempo_blob.bin
(TEMPO's tempo_init state).  TEMPO_TABLE_DIR, when set, also runs the
engine's own init (the Rust tempo_init on the pinned tables).  Skips without
the artifacts.

Breakage this prevents: any change to the TEMPO kernels, the runtime's pack
and unpack, or the init that moves one output word of one step away from
TEMPO's Fortran, in the forecast mode (free running) or per step (replay).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

ORACLE = os.environ.get("TEMPO_ORACLE_DIR")
TABLES = os.environ.get("TEMPO_TABLE_DIR")
TOOL = Path(__file__).resolve().parents[1] / "tools" / "tempo_ref" / "oracle" / "run_woof_oracle.py"


def _tool():
    spec = importlib.util.spec_from_file_location("tempo_run_woof_oracle", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(tmp_path, source):
    d = Path(ORACLE)
    tool = _tool()
    return tool, tool.parse_args([
        str(d / "columns_in.bin"), str(d / "columns_catalog.json"), str(d / "ref_out.bin"),
        *source, "--out", str(tmp_path / "summary.json"), "--mode", "both"])


def _check(result):
    for mode, m in result["modes"].items():
        assert m["mismatched"] == 0, (mode, m["max_ulp"], m["first_miss"])
        assert m["values"] == result["values_per_mode"]
        # TEMPO's undefined frozen_fraction on early-return columns is the only
        # excused cell, and WOOF holds +0 there (run_woof_oracle.py)
        assert m["tempo_undefined_frozen_fraction_cells"] < 0.001 * m["values"]


@pytest.mark.skipif(not ORACLE, reason="TEMPO_ORACLE_DIR not set")
def test_column_oracle_fortran_init_blob_is_bitwise(tmp_path):
    tool, args = _args(tmp_path, ["--blob", str(Path(ORACLE) / "tempo_blob.bin")])
    _check(tool.run(args))


@pytest.mark.skipif(not (ORACLE and TABLES), reason="TEMPO_ORACLE_DIR / TEMPO_TABLE_DIR not set")
def test_column_oracle_engine_init_is_bitwise(tmp_path):
    tool, args = _args(tmp_path, ["--tables", TABLES])
    _check(tool.run(args))
