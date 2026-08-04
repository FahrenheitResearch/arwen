"""Exhaustive admission agreement with WRF v4.6.1's ported-set matrix."""

from __future__ import annotations

from collections import Counter

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.physics_compat import (
    UnsupportedPhysicsSuiteError,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_TO_RTE_RRTMGP,
)
from gpuwm.wrf461_compatibility import (
    MATRIX_CELL_COUNT,
    PBL_SURFACE_LAYER_AUTHORITY,
    WRF_COMMIT,
    WRFVerdict,
    iter_compatibility_matrix,
)


_RADIATION_SETTINGS = {
    "off": {
        "ra_lw_physics": 0,
        "ra_sw_physics": 0,
    },
    "dudhia-shortwave": {
        "ra_lw_physics": 0,
        "ra_sw_physics": 1,
    },
    "rrtmg-rte-rrtmgp": {
        "ra_lw_physics": 4,
        "ra_sw_physics": 4,
        "wrf_rrtmg_compatibility": WRF_RRTMG_TO_RTE_RRTMGP,
    },
    "rrtmg-legacy": {
        "ra_lw_physics": 4,
        "ra_sw_physics": 4,
        "wrf_rrtmg_compatibility": WRF_RRTMG_LEGACY,
        "ra_rrtmg_variant": "rrtmg_legacy",
    },
    "analytic": {
        "ra_lw_physics": 90,
        "ra_sw_physics": 90,
    },
}


def _run_config(cell) -> RunConfig:
    values = {
        "nx": 4,
        "ny": 3,
        "nz": 49,
        "dx": 1000.0,
        "dy": 1000.0,
        "ztop": 15000.0,
        "dt": 5.0,
        "run_seconds": 0.0,
        "moist": True,
        "mp_physics": cell.mp_physics,
        "sf_sfclay_physics": cell.sf_sfclay_physics,
        "bl_pbl_physics": cell.bl_pbl_physics,
        "sf_surface_physics": cell.sf_surface_physics,
        "num_soil_layers": (
            9 if cell.sf_surface_physics == 3 else 4),
        "cu_physics": cell.cu_physics,
        "cudt_minutes": 5.0 if cell.cu_physics else 0.0,
    }
    values.update(_RADIATION_SETTINGS[cell.radiation])
    return RunConfig(**values)


def test_matrix_has_every_cell_every_citation_and_pinned_counts():
    """Pinned counts, RECOMPUTED when mp_physics=28 joined the mp axis.

    The pins moved 2400 -> 2880 and 1080/360/600/360 ->
    1296/432/720/432 because MP_OPTIONS gained a sixth value.  That every
    verdict count scaled by exactly 6/5 is not an assumption used to derive
    them -- the numbers below were read off the enlarged matrix, and the
    clean ratio is the EVIDENCE that microphysics is an independent axis of
    the WRF v4.6.1 authority table: no cell's verdict depends on which
    microphysics scheme it names, which is exactly what the six per-cell
    citations say (the mp citation is informational, and only the
    PBL/surface-layer pair can be FATAL).  A count that had not scaled by
    6/5 would have meant mp=28 changed a verdict somewhere, and that would
    need a WRF citation, not a pin update.

    Re-pinned when bl_pbl_physics=11 (Shin-Hong) joined the PBL axis:
    2880 -> 3840, and the verdicts moved by exactly the 960 new pbl=11
    cells -- FATAL +480 (the (11,0) and (11,5) SHINHONGSCHEME fatals at
    phys/module_physics_init.F:3702-3704 span 240 cells each), LEGAL +288,
    reconfigured +96 (the no-LSM slice) and not-expressible +96 (the
    analytic slice).  PBL is NOT an independent axis -- the pair verdict is
    exactly where it can be FATAL -- so these did not scale uniformly, and
    each increment was read off the enlarged matrix rather than derived.
    """
    cells = tuple(iter_compatibility_matrix())
    assert len(cells) == MATRIX_CELL_COUNT == 3840
    assert len({
        (
            cell.mp_physics,
            cell.bl_pbl_physics,
            cell.sf_sfclay_physics,
            cell.sf_surface_physics,
            cell.radiation,
            cell.cu_physics,
        )
        for cell in cells
    }) == MATRIX_CELL_COUNT
    assert all(
        len(cell.citations) == 6
        and all(
            citation.path and citation.lines and citation.law
            for citation in cell.citations)
        for cell in cells
    )
    assert Counter(cell.verdict for cell in cells) == {
        WRFVerdict.LEGAL: 1584,
        WRFVerdict.LEGAL_RECONFIGURED: 528,
        WRFVerdict.FATAL: 1200,
        WRFVerdict.NOT_EXPRESSIBLE: 528,
    }
    # Every represented mp value carries its own Registry citation, so the
    # matrix cannot grow an axis value that is admitted without one.
    assert {cell.mp_physics for cell in cells} == {1, 6, 8, 10, 18, 28}
    assert WRF_COMMIT == "d66e442fccc04111067e29274c9f9eaccc3cef28"


def test_pbl_surface_layer_authority_is_the_complete_sixteen_cell_table():
    # Twelve cells until Shin-Hong (bl_pbl_physics=11) was admitted; its
    # four cells follow the SHINHONGSCHEME case, which is YSU's isfc=1 law
    # through WRF's own arm (phys/module_physics_init.F:3702-3704).
    assert len(PBL_SURFACE_LAYER_AUTHORITY) == 16
    legal = {
        pair for pair, (verdict, _citation)
        in PBL_SURFACE_LAYER_AUTHORITY.items()
        if verdict is WRFVerdict.LEGAL
    }
    assert legal == {
        (0, 0), (0, 1), (0, 5), (0, 91),
        (1, 1), (1, 91),
        (5, 1), (5, 5), (5, 91),
        (11, 1), (11, 91),
    }


def test_every_front_door_tuple_agrees_with_the_wrf_matrix():
    """Sweep the full 3,840-cell product through RunConfig admission.

    The only local refusal of a WRF-legal cell is the separately named
    ArWen structural seam: an active LSM has no exchange-field writer when
    the surface layer is off.  WRF's analytic disposition is "not
    expressible", not "fatal"; ArWen executes it and governance separately
    requires the expert acknowledgement.
    """

    observed = Counter()
    for cell in iter_compatibility_matrix():
        cfg = _run_config(cell)
        if cell.verdict is WRFVerdict.FATAL:
            with pytest.raises(UnsupportedPhysicsSuiteError) as caught:
                validate_run_config(cfg)
            blocker = caught.value.blockers[0]
            assert blocker.component == (
                "WRF v4.6.1 PBL/surface-layer compatibility")
            assert blocker.selectors == (
                ("sf_sfclay_physics", cell.sf_sfclay_physics),
                ("bl_pbl_physics", cell.bl_pbl_physics),
            )
            assert "phys/module_physics_init.F:" in str(caught.value)
            observed["wrf-fatal"] += 1
        elif cell.sf_surface_physics and not cell.sf_sfclay_physics:
            with pytest.raises(ValueError, match=(
                    "requires a surface layer .* exchange coefficients")):
                validate_run_config(cfg)
            observed["arwen-structural"] += 1
        else:
            assert validate_run_config(cfg) is cfg
            observed["admitted"] += 1

    # Recomputed with mp_physics=28 on the axis: 1650/150/600 -> 1980/180/720,
    # the same 6/5 scaling, which is the front-door half of the statement
    # the matrix test makes about the authority table.  Concretely: all 480
    # mp=28 cells reach a verdict for reasons that have nothing to do with
    # microphysics, so mp=28 adds no new refusal and removes none.
    #
    # Re-pinned when bl_pbl_physics=11 (Shin-Hong) joined the PBL axis: its
    # 960 cells split 480 wrf-fatal (sfclay 0 and 5, the SHINHONGSCHEME
    # case) and 480 admitted.  arwen-structural does not move, because the
    # structural seam is sfclay=0 with an active LSM and every (11, 0) cell
    # is already WRF-fatal before the seam is consulted.
    assert observed == {
        "admitted": 2460,
        "arwen-structural": 180,
        "wrf-fatal": 1200,
    }
