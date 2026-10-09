"""The WRF-input door's cold-start base state against stock WRF 4.6.1.

Oracle: ``tests/oracles/wrf461_start_base_columns.npz``, 160 columns of the
stock-WRF export of the g400 crop (2024-05-21 18Z, HRRR start, RAP
boundaries; wrfinput as written by the 2.8.7 exporter) and the MUB, PB and
PHB words the unmodified WRF 4.6.1 wrf.exe (gfortran 13.3, glibc 2.39,
64 MPI ranks) wrote into its minute-0 history from that file.  The columns
are 120 random ones, the 20 highest-terrain ones and 40 where the
once-rounded binary64 exponential misses glibc's expf (so the p_surf EXP
is pinned too).  ``file_*`` are the export's own base words, kept to show
the oracle discriminates: the door restored those, and they differ from
WRF's in 136 of the 160 MUB columns.

WRF's start_domain_em (start_em.F:554-680, input_from_file, not a restart)
throws the file's base state away and rebuilds it from the terrain and the
reference-profile constants.  The door kept the file's.
"""
from pathlib import Path

import numpy as np
import pytest

ORACLE = Path(__file__).parent / "oracles" / "wrf461_start_base_columns.npz"


@pytest.fixture(scope="module")
def oracle():
    with np.load(ORACLE) as data:
        return {name: data[name] for name in data.files}


def _raw(oracle):
    raw = {name: oracle[name] for name in (
        "C3H", "C4H", "C3F", "C4F", "C1H", "C2H", "DNW", "P_TOP",
        "P00", "T00", "TLP", "TISO", "TLP_STRAT", "P_STRAT")}
    raw["HGT"] = oracle["HGT"][None, :]          # (ny=1, nx=160)
    return raw


def test_oracle_discriminates_the_file_base_from_wrfs(oracle):
    assert np.count_nonzero(oracle["file_MUB"] != oracle["wrf_MUB"]) > 100
    assert np.count_nonzero(oracle["file_PHB"] != oracle["wrf_PHB"]) > 1000


def test_cold_start_base_state_is_wrf461_word_for_word(oracle):
    from gpuwm.ingest.wrfinput import wrf_start_base_state

    start = wrf_start_base_state(_raw(oracle), 2)
    for name in ("MUB", "PB", "PHB"):
        got = np.asarray(start[name], np.float32)
        want = np.asarray(oracle[f"wrf_{name}"], np.float32)
        got = got.reshape(want.shape)
        differing = np.count_nonzero(got.view(np.uint32) != want.view(np.uint32))
        assert differing == 0, f"{name}: {differing} words differ from WRF 4.6.1"


def test_door_base_is_the_cold_start_not_the_file(oracle):
    """``wrf_coordinate_and_base`` hands load_base WRF's rebuilt words."""
    from types import SimpleNamespace
    from gpuwm.ingest import wrfinput

    raw = _raw(oracle)
    nz = raw["C3H"].size
    for name, size in (("ZNW", nz + 1), ("C1F", nz + 1), ("C2F", nz + 1),
                       ("ZNU", nz), ("RDNW", nz), ("DN", nz),
                       ("RDN", nz), ("FNP", nz), ("FNM", nz)):
        raw[name] = np.zeros(size, np.float32)
    raw.update({
        "MUB": oracle["file_MUB"][None, :],
        "PB": oracle["file_PB"][:, None, :],
        "PHB": oracle["file_PHB"][:, None, :],
        "ALB": oracle["file_ALB"][:, None, :],
        "T_INIT": oracle["file_T_INIT"][:, None, :],
    })
    restored = SimpleNamespace(raw=raw, global_attributes={
        "HYBRID_OPT": 2, "ETAC": 0.2, "HYPSOMETRIC_OPT": 2})
    _, base = wrfinput.wrf_coordinate_and_base(restored, 2)
    assert np.array_equal(np.asarray(base.mub, np.float32).reshape(-1),
                          oracle["wrf_MUB"])
    assert np.array_equal(np.asarray(base.pb, np.float32).reshape(nz, -1),
                          oracle["wrf_PB"])
    assert np.array_equal(np.asarray(base.phb, np.float32).reshape(nz + 1, -1),
                          oracle["wrf_PHB"])


def test_expf_array_is_glibc_on_the_near_midpoint_columns(oracle):
    from gpuwm.core.noahmp_libm import expf
    from gpuwm.ingest.wrfinput import _glibc_expf_array

    x = np.linspace(-0.6, 0.0, 20001, dtype=np.float32)
    got = _glibc_expf_array(x)
    want = np.array([expf(v) for v in x], np.float32)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_missing_reference_constants_are_refused_as_wrf_refuses(oracle):
    from gpuwm.ingest.wrfinput import wrf_start_base_state

    raw = _raw(oracle)
    del raw["T00"]
    with pytest.raises(ValueError, match="T00"):
        wrf_start_base_state(raw, 2)
