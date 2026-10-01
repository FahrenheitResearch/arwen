"""The URBPARM parser is WRF v4.7.1's ``urban_param_init``, bit for bit.

The reference is the WRF routine itself, compiled at -O0 from the pinned tree
and run by ``tools/urban_wrf471_oracle/run_init.F90``; its module state after
the call is in ``gpuwm/data/urban/oracle/infra/init/opt<N>_lcz<M>/``.  Every
table array, every derived value (``HGT/R/RW/ZDC/Z0C/Z0HC/SVF``, option 1's
``Z0R``, the unit-converted ``CAP*``/``AKS*``, ``DZR/DZB/DZG`` in cm) and the
street/height blocks must equal it with zero ULP.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.urban_tables import (URBAN_TABLE_SHA256, UrbanTableError,
                                     decimal_to_f32, load_urban_params,
                                     urban_category_set, urban_table_path)
from gpuwm.verify.urban_oracle import ORACLE_ROOT, load, ulp_table

INIT = ORACLE_ROOT / "infra" / "init"
CASES = [(opt, lcz) for opt in (1, 2, 3) for lcz in (0, 1)]


@pytest.fixture(scope="module")
def oracle():
    return load(INIT)


#: Names urban_param_init leaves in its module that the parser publishes.
TABLE_NAMES = (
    "ZR_TBL", "SIGMA_ZED_TBL", "Z0C_TBL", "Z0HC_TBL", "ZDC_TBL", "SVF_TBL",
    "R_TBL", "RW_TBL", "HGT_TBL", "AH_TBL", "ALH_TBL", "BETR_TBL", "BETB_TBL",
    "BETG_TBL", "FRC_URB_TBL", "COP_TBL", "BLDAC_FRC_TBL", "COOLED_FRC_TBL",
    "PWIN_TBL", "BETA_TBL", "SW_COND_TBL", "TIME_ON_TBL", "TIME_OFF_TBL",
    "TARGTEMP_TBL", "GAPTEMP_TBL", "TARGHUM_TBL", "GAPHUM_TBL", "PERFLO_TBL",
    "PV_FRAC_ROOF_TBL", "GR_FRAC_ROOF_TBL", "GR_FLAG_TBL", "GR_TYPE_TBL",
    "IRHO_TBL", "HSESF_TBL", "CAPR_TBL", "CAPB_TBL", "CAPG_TBL", "AKSR_TBL",
    "AKSB_TBL", "AKSG_TBL", "ALBR_TBL", "ALBB_TBL", "ALBG_TBL", "EPSR_TBL",
    "EPSB_TBL", "EPSG_TBL", "Z0R_TBL", "Z0B_TBL", "Z0G_TBL", "Z0HB_TBL",
    "Z0HG_TBL", "TRLEND_TBL", "TBLEND_TBL", "TGLEND_TBL",
    "AKANDA_URBAN_TBL", "NUMDIR_TBL", "STREET_DIRECTION_TBL",
    "STREET_WIDTH_TBL", "BUILDING_WIDTH_TBL", "NUMHGT_TBL", "HEIGHT_BIN_TBL",
    "HPERCENT_BIN_TBL", "BOUNDR_DATA", "BOUNDB_DATA", "BOUNDG_DATA",
    "CH_SCHEME_DATA", "TS_SCHEME_DATA", "AHOPTION", "AHDIUPRF",
    "HSEQUIP_TBL", "IMP_SCHEME", "IRI_SCHEME", "ALHOPTION", "GROPTION",
    "FGR", "OASIS", "DZGR", "ALHSEASON", "ALHDIUPRF", "PORIMP", "DENGIMP",
    "DZR", "DZB", "DZG")


@pytest.mark.parametrize("opt,lcz", CASES)
def test_the_parser_is_urban_param_init_bit_for_bit(oracle, opt, lcz):
    ref = oracle[f"opt{opt}_lcz{lcz}"]
    params = load_urban_params(opt, lcz)
    assert params.icate == int(ref["ICATE"])
    worst = {}
    for name in TABLE_NAMES:
        got = np.asarray(params.values[name])
        want = np.asarray(ref[name])
        assert got.shape == want.shape, name
        worst[name] = ulp_table(got, want)["max_ulp"]
    assert worst == {name: 0 for name in TABLE_NAMES}


def test_option_one_rederives_z0r_and_the_others_keep_the_row(oracle):
    one = load_urban_params(1, 0).Z0R_TBL
    two = load_urban_params(2, 0).Z0R_TBL
    assert not np.array_equal(one, two)
    assert np.all(two == np.float32(0.01))


def test_decimal_text_rounds_once_to_binary32():
    # 0.1 through binary64 and then binary32 is also correct; the case the
    # helper exists for is a decimal within half a binary32 ULP of a
    # midpoint, where rounding twice can land one ULP off.
    halfway = 1.0 + 2.0 ** -24               # exact binary32 midpoint
    above = repr(halfway + 2.0 ** -60)       # decimal just above it
    assert decimal_to_f32(above) == np.float32(1.0 + 2.0 ** -23)
    assert decimal_to_f32("0.1") == np.float32(0.1)
    assert decimal_to_f32("1.0E6") == np.float32(1.0e6)


def test_a_moved_table_is_refused(tmp_path):
    src = urban_table_path(0)
    changed = src.read_bytes().replace(b"FRC_URB: 0.5", b"FRC_URB: 0.6")
    (tmp_path / "URBPARM.TBL").write_bytes(changed)
    with pytest.raises(UrbanTableError, match="SHA-256"):
        load_urban_params(1, 0, tbl_dir=tmp_path)
    assert URBAN_TABLE_SHA256["URBPARM.TBL"] != ""


def test_an_untranscribed_table_arm_is_refused_by_name(tmp_path):
    src = urban_table_path(0)
    changed = src.read_bytes().replace(b"OASIS: 1.0", b"OASIS: 1.3")
    (tmp_path / "URBPARM.TBL").write_bytes(changed)
    with pytest.raises(UrbanTableError, match="OASIS"):
        load_urban_params(1, 0, tbl_dir=tmp_path, verify_sha256=False)


def test_categories_are_vegparm_rows_not_literals():
    cats = urban_category_set(isurban=13)
    assert cats.natural == 14
    assert cats.lcz == tuple(range(51, 62))
    lookup0 = cats.utype_lookup(0)
    lookup1 = cats.utype_lookup(1)
    assert lookup0[13] == 2 and lookup1[13] == 5
    assert [int(lookup0[c]) for c in cats.lcz] == list(range(1, 12))
    assert lookup0[14] == 0 and lookup0[10] == 0
