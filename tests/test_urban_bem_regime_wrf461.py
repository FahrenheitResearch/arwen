"""BEP+BEM (``sf_urban_physics = 3``) against WRF v4.6.1 over weather regimes.

``tools/urban_bem_regime_oracle`` compiles byte-unmodified WRF v4.6.1
``module_sf_bep_bem.F`` and ``module_sf_bem.F`` (with WRF's own
``urban_param_init`` and ``urban_var_init``) at ``-O0`` and drives ``BEP_BEM``
over every regime of ``REGIMES.csv`` crossed with every urban class of the
table, plus eight edge columns (rural land, open water, FRC 0.01 and 1.0,
gridded morphology, gridded towers up to the 15th height bin, a 40 m and a
2 m first level).  The fixtures under ``tests/data/oracles/urban/bem_regime461``
(see ``PROVENANCE.md`` there) are:

==================  =====================================================
``nlcd-summer``     URBPARM.TBL, 14 warm-season regimes x 3 classes, 60 s
``nlcd-winter``     URBPARM.TBL, 12 cold-season regimes x 3 classes, 30 s
``lcz-summer``      URBPARM_LCZ.TBL, 14 regimes x 11 LCZ classes, 60 s
``lcz-winter``      URBPARM_LCZ.TBL, 12 regimes x 11 LCZ classes, 30 s
``gr1pv-summer``    green roof type 1, rooftop PV, AC schedule, irrigation
``gr2-winter``      green roof type 2, irrigation every hour
``long-summer``     ``nlcd-summer`` for 90 calls at 20 s (30 minutes)
``stop-summer``     ``nlcd-summer`` for 40 calls at 60 s: WRF STOPs in call
                    37 (``upward_rad``: a roof temperature below 100 K in
                    the 28 m/s windstorm column, a growing two-step
                    oscillation of WRF's explicit roof surface update)
==================  =====================================================

The engine carries its OWN state from WRF's initial state through every call,
and every output of every call plus every state array after the last call
must be WRF's word for word.  Where WRF stops the engine must stop in the
same call, on the same column, naming WRF's check.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
import _urban_bem_regime as R
# Named import: the GPU shard manifest pins this device entry
# (tests/test_gpu_shard_manifest.py _TRANSITIVE_DEVICE_HELPERS).
from _urban_bem_regime import measure

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "urban_bem_regime_oracle"

#: What each fixture holds: (columns, calls, step [s], winter, lcz table,
#: green roof flag).  The urban column total over the eight is 574.
SHAPES = {
    "nlcd-summer": (50, 6, 60.0, 0, 0, 0),
    "nlcd-winter": (44, 6, 30.0, 1, 0, 0),
    "lcz-summer": (162, 6, 60.0, 0, 1, 0),
    "lcz-winter": (140, 6, 30.0, 1, 1, 0),
    "gr1pv-summer": (50, 6, 60.0, 0, 0, 1),
    "gr2-winter": (44, 6, 60.0, 1, 0, 1),
    "long-summer": (50, 90, 20.0, 0, 0, 0),
    "stop-summer": (50, 40, 60.0, 0, 0, 0),
}


def test_fixture_shapes_are_the_ones_provenance_names():
    urban = 0
    for variant, (ncol, nsteps, dt, winter, lcz, gr) in SHAPES.items():
        fx = R.load_fixture(variant)
        assert int(fx["ncol"][0]) == ncol, variant
        assert int(fx["nsteps"][0]) == nsteps, variant
        assert float(fx["step1/dt"][0]) == dt, variant
        assert int(fx["season_winter"][0]) == winter, variant
        assert int(fx["use_wudapt_lcz"][0]) == lcz, variant
        assert int(fx["tbl/gr_flag_tbl"][0]) == gr, variant
        urban += int((fx["state0/frc_urb2d"] > 0).sum())
    assert urban == 574


def test_regime_table_is_the_one_the_fixtures_were_built_from():
    """The fixtures carry the regime ids; the CSV carries their names."""
    for winter, count in ((False, 14), (True, 12)):
        names = R.read_regimes(TOOL / "REGIMES.csv", winter)
        assert sorted(names) == list(range(1, count + 1))
    fx = R.load_fixture("lcz-winter")
    assert set(fx["col/regime"].tolist()) == set(range(1, 13))
    assert set(fx["col/class"][fx["col/edge"] == 0].tolist()) == set(range(1, 12))
    assert sorted(fx["col/edge"][fx["col/edge"] > 0].tolist()) == list(range(1, 9))


def test_only_the_stop_fixture_records_a_wrf_stop():
    for variant in SHAPES:
        fx = R.load_fixture(variant)
        assert R.wrf_stop_call(fx) == (37 if variant == "stop-summer" else None)


def test_raw_dump_hashes_are_recorded():
    """Each npz is its raw dump, repacked; the dump hashes are in the folder."""
    listed = (R.FIXTURE_DIR / "data-sha256sums.txt").read_text().split()
    assert len(listed) == 2 * len(SHAPES)
    for variant in SHAPES:
        assert f"{variant}/dump/data.bin" in listed


@pytest.mark.parametrize("variant", sorted(SHAPES))
def test_option3_cold_start_is_wrf461_urban_var_init(variant):
    """The engine's cold start is WRF v4.6.1's ``urban_var_init``, word for word.

    v4.6.1 and v4.7.1 differ in module_sf_urban.F only inside the
    single-layer model's IMP_SCHEME=2 retention, which option 3 never runs.
    """
    from gpuwm.core import urban_bem as ub
    from gpuwm.core.urban_state import (option_spec, resolve_dimensions,
                                        urban_var_init_host)
    from gpuwm.core.urban_tables import UrbanCategories, load_urban_params

    fx = R.load_fixture(variant)
    if int(fx["tbl/gr_flag_tbl"][0]):
        pytest.skip("derived green-roof tables are not shipped tables")
    lcz = int(fx["use_wudapt_lcz"][0])
    nz = int(fx["nz"][0])
    out = urban_var_init_host(
        option=3, use_wudapt_lcz=lcz, params=load_urban_params(3, lcz),
        categories=UrbanCategories(isurban=13, natural=14,
                                   lcz=tuple(range(51, 62))),
        ivgtyp=R.to_gpuwm(fx["init/ivgtyp"]),
        tsk=R.to_gpuwm(fx["init/tsk"]), tslb=R.to_gpuwm(fx["init/tslb"]),
        tmn=R.to_gpuwm(fx["init/tmn"]), smois=R.to_gpuwm(fx["init/smois"]),
        frc_urb2d=R.to_gpuwm(fx["init/frc_urb2d_in"]),
        num_urban_hi=int(fx["num_urban_hi"][0]), nz=nz,
        dims=resolve_dimensions(3, ub), spec=option_spec(3, ub))
    for name in R.STATE_ARRAYS + ("frc_urb2d", "utype_urb2d"):
        want = R.to_gpuwm(fx[f"state0/{name}"])
        got = out[name]
        assert got.dtype == want.dtype, name
        np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32),
                                      err_msg=name)


@pytest.mark.parametrize("variant", sorted(SHAPES))
def test_the_vendored_tables_pack_to_the_words_wrf461_read(variant):
    from gpuwm.core import urban_bem as ub
    from gpuwm.core.urban_tables import load_urban_params

    fx = R.load_fixture(variant)
    if int(fx["tbl/gr_flag_tbl"][0]):
        pytest.skip("derived green-roof tables are not shipped tables")
    lcz = int(fx["use_wudapt_lcz"][0])
    ours = ub.pack_bem_tables(ub.tables_from_params(load_urban_params(3, lcz)))
    wrf = ub.pack_bem_tables(R.table_inputs(fx))
    for got, want in zip(ours, wrf):
        np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("variant", sorted(SHAPES))
def test_bep_bem_regimes_match_wrf461_bitwise(variant):
    """0 ULP on every output of every call and every final state word.

    Measured on an RTX PRO 6000 Blackwell (sm_120, NVRTC 12.9) under
    ``GPUWM_WRF_EXACT=1`` and under default arithmetic: the unit compiles
    the same source with the same options either way.
    """
    fx = R.load_fixture(variant)
    rep = measure(fx)
    differ = {k: v for k, v in rep["fields"].items() if v["words_differ"]}
    assert not differ, differ
    assert rep["words"] > 0
    stop = R.wrf_stop_call(fx)
    if stop is None:
        assert rep["engine_stop"] is None
    else:
        call, message = rep["engine_stop"]
        assert call == stop
        assert message is not None, "WRF stopped; the engine ran on"
        assert "column 26" in message and "upward_rad" in message, message


def test_harness_files_are_present():
    for name in ("run_bep_bem_regime.F90", "build.sh", "build_u2604.sh",
                 "fetch_sources.sh", "SOURCES.sha256", "REGIMES.csv",
                 "compare.py", "to_npz.py", "libm_shim.c",
                 "run_gpu_compare.sh"):
        assert (TOOL / name).is_file(), name
    pins = (TOOL / "SOURCES.sha256").read_text().split()
    assert "392ae96ad16000e7592112c12f2fe7909bec1760c67c5ef33ec876048a745c48" in pins
    digest = hashlib.sha256((TOOL / "REGIMES.csv").read_bytes()).hexdigest()
    assert digest in (R.FIXTURE_DIR / "PROVENANCE.md").read_text()
