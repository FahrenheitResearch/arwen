"""WRF v4.6.1 Ferrier-Aligo (mp_physics=5) column oracle: fixtures and status.

The oracle (tools/ferrier_hires_wrf461_oracle) runs byte-unmodified
``phys/module_mp_fer_hires.F`` over 72 columns in nine regimes and commits
its outputs under tests/data/oracles/ferrier_hires. These tests bind the
fixtures, the comparator and default admission. The GPU column check
is tools/ferrier_hires_wrf461_oracle/check_port.py.

CPU only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools" / "ferrier_hires_wrf461_oracle"
FIXTURES = ROOT / "tests" / "data" / "oracles" / "ferrier_hires"
NZ = 50


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_ferrier_{name}", HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pinned_sums() -> dict[str, str]:
    sums = {}
    for line in (FIXTURES / "SHA256SUMS").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            sums[parts[1]] = parts[0]
    return sums


def test_fixtures_match_the_hashes_the_oracle_run_recorded():
    sums = _pinned_sums()
    for name, ref in (("columns_inputs.csv", "columns_inputs.csv"),
                      ("levels.csv", "O0/levels.csv"),
                      ("surface.csv", "O0/surface.csv")):
        got = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        assert got == sums[ref], name
    # The WRF-flags build (-O2 -ftree-vectorize) wrote the same bits.
    assert sums["O2vec/levels.csv"] == sums["O0/levels.csv"]
    assert sums["O2vec/surface.csv"] == sums["O0/surface.csv"]


def test_the_generator_reproduces_the_committed_inputs(tmp_path):
    _load("make_columns").main(tmp_path)
    assert (tmp_path / "columns_inputs.csv").read_bytes() == \
        (FIXTURES / "columns_inputs.csv").read_bytes()


def test_the_column_set_spans_the_stated_regimes():
    cols = [line.split(",") for line in
            (FIXTURES / "columns_inputs.csv").read_text().splitlines()
            if line.startswith("C,")]
    assert len(cols) == 72
    regimes = Counter(c[2] for c in cols)
    assert regimes == {r: 8 for r in (
        "convective", "stable_night", "snow", "ocean", "high_terrain",
        "freezing_rain", "cold_anvil", "edge", "mixed")}
    assert {c[3] for c in cols} == {"1", "2"}          # both RHgrd paths
    assert len({c[5] for c in cols}) == 4               # four time steps
    levels = (FIXTURES / "levels.csv").read_text().splitlines()
    assert len(levels) == 1 + 72 * 2 * NZ               # steps 1 and 10
    surface = (FIXTURES / "surface.csv").read_text().splitlines()
    assert len(surface) == 1 + 72 * 2


def test_the_comparator_sees_a_single_bit(tmp_path):
    cmp = _load("compare_ferrier")
    ref = tmp_path / "ref"
    bad = tmp_path / "bad"
    for d in (ref, bad):
        d.mkdir()
        for name in ("levels.csv", "surface.csv"):
            shutil.copy(FIXTURES / name, d / name)
    inputs = FIXTURES / "columns_inputs.csv"
    assert cmp.compare(ref, bad, inputs)
    lines = (bad / "levels.csv").read_text().splitlines()
    fields = lines[1].split(",")
    fields[3] = f"{int(fields[3], 16) ^ 1:08X}"
    lines[1] = ",".join(fields)
    (bad / "levels.csv").write_text("\n".join(lines) + "\n")
    assert not cmp.compare(ref, bad, inputs)


# ---------------------------------------------------------------------------
# Admission and the real importer.

def test_woof_admits_ferrier_without_an_opt_in():
    from gpuwm import config as c
    from gpuwm.ingest import wrfinput
    from gpuwm.namelist_import import _MP_MAP
    assert 5 in c.MP_PHYSICS_ACCEPTED
    assert _MP_MAP[5] == (5, "Ferrier-Aligo", "Ferrier-Aligo")
    assert 5 in wrfinput.SUPPORTED_MICROPHYSICS
    assert 5 not in wrfinput.UNSUPPORTED_MICROPHYSICS
    cfg = c.RunConfig(nx=8, ny=6, nz=8, dx=12000., dy=12000.,
        ztop=10000., dt=6., run_seconds=60., moist=True, mp_physics=5,
        bl_pbl_physics=2, sf_sfclay_physics=2)
    c.validate_run_config(cfg)

def test_real_namelist_importer_keeps_mp5_and_eta_pair(tmp_path):
    from test_namelist_import import INPUT_TEXT, WPS_TEXT, _pair
    from gpuwm.namelist_import import import_namelists
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 5, 5")
    inp = inp.replace("bl_pbl_physics = 11, 11", "bl_pbl_physics = 2, 2")
    inp = inp.replace("sf_sfclay_physics = 91, 91", "sf_sfclay_physics = 2, 2")
    toml, report = import_namelists(*_pair(tmp_path, inp=inp))
    assert "mp_physics = 5" in toml
    assert "bl_pbl_physics = 2" in toml
    assert "sf_sfclay_physics = 2" in toml


def test_mechanical_check_fails_without_receipts(tmp_path):
    import sys
    sys.path.insert(0, str(HARNESS))
    checker = _load('check_port')
    result = checker.check(FIXTURES, tmp_path, None)
    assert not result['pass']
    assert not result['forecast']
    assert not result['planted_bit_caught']
    assert all(not x['pass'] for x in result['strict'].values())


def test_a_claimed_forecast_without_its_artifacts_cannot_pass(tmp_path):
    import sys
    import json
    sys.path.insert(0, str(HARNESS))
    checker = _load('check_port')
    receipt = tmp_path/'claimed.json'
    receipt.write_text(json.dumps({'completed': True, 'dx':12000,
        'mp_physics':5,'bl_pbl_physics':2,'sf_sfclay_physics':2,
        'field_hashes':{'fake':'abc'},'rust_sheet_sha256':'fake'}))
    result=checker.check(FIXTURES,tmp_path,receipt)
    assert not result['forecast']
    assert not result['pass']
    assert result['forecast_error']
