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

# Consumer expectations come from the independent generated declarations.
from gpuwm.physics_registry import implemented_selector_values as _intake_selectors
from gpuwm.physics_registry import consumer_rows_by_selector as _intake_rows
_INTAKE_MP = tuple(sorted(_intake_selectors("microphysics")))
_INTAKE_REFL = tuple(mp for mp in _INTAKE_MP if mp)
_INTAKE_PRODUCERS = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "reflectivity_input_species").items()
    if row is not None))
_INTAKE_NATIVE = tuple(sorted(set(_INTAKE_REFL) - set(_INTAKE_PRODUCERS)))
_INTAKE_SNOW = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qs" in (row or {}).get("state_fields", ())))
_INTAKE_SIX = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qg" in (row or {}).get("state_fields", ())))


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
        # KF keeps its 5-minute template cadence; GF pins cudt to 0 (it
        # runs on the model step) and cumulus-off carries no cadence.
        "cudt_minutes": 5.0 if cell.cu_physics == 1 else 0.0,
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
    assert len(cells) == MATRIX_CELL_COUNT == 38080
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
    # Re-pinned when cu_physics=3 (Grell-Freitas) joined the cumulus
    # axis: 3840 -> 5760.  Cumulus IS an independent axis in WRF v4.6.1 --
    # no other component's legality is conditioned on it -- so every
    # verdict scaled by exactly 3/2 (LEGAL 1584 -> 2376, reconfigured
    # 528 -> 792, FATAL 1200 -> 1800, not-expressible 528 -> 792), and
    # each count below was read off the enlarged matrix, not derived.
    #
    # Re-pinned when cu_physics=16 (New Tiedtke) joined the cumulus axis
    # (5760 -> 7680, every verdict scaled by 4/3) and again when mp_physics
    # 0, 9, 16 and 50 joined the microphysics axis (audit R-012; 7680 ->
    # 12800).  Both are independent axes -- compatibility_cell reads the
    # PBL/surface-layer pair, the radiation label and sf_surface_physics,
    # and takes only a citation from cu_physics and mp_physics -- so every
    # verdict scaled by exactly 4/3 and then 10/6, and each count below was
    # read off the enlarged matrix, not derived.
    #
    # Re-pinned when cu_physics=5 (Grell 3D) joined the cumulus axis on the
    # 2.9.0 line (17600 -> 21120): every verdict scaled by exactly 6/5, read
    # off the enlarged matrix, which is again the evidence that cumulus is
    # an independent axis of the WRF v4.6.1 authority table.
    # And again when cu_physics=96 (New SAS) joined (21120 -> 24640): every
    # verdict scaled by exactly 7/6, read off the enlarged matrix.
    # And again on the 2.9.0 round-2 merge, when mp_physics 2, 3, 4, 5, 24
    # and 106 (the six microphysics ports) joined the eleven-value mp axis
    # beside the cumulus ports (24640 -> 38080): every verdict scaled by
    # exactly 17/11 (LEGAL 10164 -> 15708, reconfigured 3388 -> 5236, FATAL
    # 7700 -> 11900, not-expressible 3388 -> 5236), read off the merged
    # tree's matrix -- the evidence that microphysics stays an independent
    # axis with the new schemes in it.
    assert Counter(cell.verdict for cell in cells) == {
        WRFVerdict.LEGAL: 15708,
        WRFVerdict.LEGAL_RECONFIGURED: 5236,
        WRFVerdict.FATAL: 11900,
        WRFVerdict.NOT_EXPRESSIBLE: 5236,
    }
    # Every represented cumulus value carries its own Registry citation,
    # on the same terms the mp assertion below states.
    assert {cell.cu_physics for cell in cells} == {0, 1, 3, 5, 6, 16, 96}
    # Every represented mp value carries its own Registry citation, so the
    # matrix cannot grow an axis value that is admitted without one.
    assert {cell.mp_physics for cell in cells} == {
        0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 16, 18, 24, 28, 30, 50, 106}
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
        elif cell.cu_physics == 3 and not cell.bl_pbl_physics:
            # The second named structural seam, registered on the
            # grell-freitas option (arwen_pbl_structural_requirement):
            # WRF admits GF with PBL off and reads KPBL=0; ArWen refuses
            # rather than indexing below the column base.
            with pytest.raises(ValueError, match=(
                    "cu_physics=3 .* requires a PBL scheme")):
                validate_run_config(cfg)
            observed["arwen-structural-gf-pbl"] += 1
        elif cell.mp_physics in (2, 30) and cell.radiation == "rrtmg-rte-rrtmgp":
            # A JUDGED cloud-optics exclusion, not a WRF rule (Lin, mp=2, has no
            # effective-radius producer at all): WRF pairs
            # FAST_SBM with RRTMG, but its bin effective-radius integrals
            # are not qualified for the RTE+RRTMGP substitution
            # (gpuwm.core.rrtmgp._NO_CLOUD_OPTICS_COUPLING), so the run door
            # refuses before step 0 and names the legacy-RRTMG remedy; the
            # rrtmg-legacy cells of the same tuples are admitted.
            with pytest.raises(NotImplementedError, match=(
                    "cloud-optics coupling.*rrtmg_legacy")):
                validate_run_config(cfg)
            observed["arwen-judged-cloud-optics"] += 1
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
    # Re-measured when cu_physics=3 joined the axis (3840 -> 5760 cells).
    # The 1920 new cu=3 cells split 600 wrf-fatal and 90 lsm-structural on
    # the same seams every cu value carries, plus the NEW named seam: 390
    # (pbl=0, cu=3) cells WRF admits and ArWen refuses -- 480 pbl-off
    # cells at cu=3 minus the 90 whose LSM refusal fires first in
    # validate_run_config's ordering.
    #
    # Re-measured when cu_physics=16 joined the axis (5760 -> 7680).  Its
    # 1920 cells carry NO cumulus-specific refusal at all -- New Tiedtke
    # reads no KPBL, so the seam that gives GF its 390 cells does not
    # exist for it -- and they split exactly as the cumulus-off cells do:
    # 600 wrf-fatal, 90 lsm-structural, 1230 admitted.  That the two
    # cumulus schemes split differently is the point of the count: the GF
    # seam is a scheme's own read, not a cumulus-slot rule.
    #
    # Re-measured when mp 0/9/16/50 joined the axis (7680 -> 12800, audit
    # R-012).  None of the four adds a refusal or removes one: the
    # RTE+RRTMGP adapter derives Milbrandt-Yau's radii from the scheme's
    # own moments (gpuwm/core/rrtmgp.py), so the cloud-optics seam that
    # once refused mp=9 on that radiation pair no longer exists and every
    # mp=9 cell reaches the same verdict its neighbours do.  wrf-fatal, the
    # LSM seam and the GF/PBL seam therefore all scaled by exactly 10/6,
    # and every count below was read off the enlarged walk.
    #
    # Re-measured for 2.8.9 when the run door began refusing FAST_SBM on the
    # RTE+RRTMGP pair (the registry already did): 151 mp=30 cells move from
    # admitted to the judged cloud-optics refusal -- every mp=30 RTE+RRTMGP
    # cell that no earlier WRF-fatal or structural rule refuses.  Nothing
    # else moved.
    #
    # Re-measured at the 2.8.9 integration, where FAST_SBM (mp 30) and
    # Original cu6 (cu 6) meet: 17600 cells, 12800 * 11/10 * 5/4.
    # wrf-fatal, the LSM seam and the GF/PBL seam scale by both factors
    # exactly; the cloud-optics refusal reaches 192 mp=30 RTE+RRTMGP cells
    # (the 151 measured on the four cumulus values the lane saw, plus 41 at
    # cu=6 that no earlier rule refuses), read off the merged tree's sweep.
    #
    # Re-measured when Grell 3D (cu 5) joined on the 2.9.0 line: 21120 cells.
    # wrf-fatal and the LSM seam scale by 6/5 exactly; the GF/PBL seam does
    # not move (it is a cu=3 rule); the cloud-optics refusal gains 41 cu=5
    # cells, the same 41 cu=6 contributed; the remaining 2214 new cells are
    # admitted.  Every number was read off the merged tree's sweep.
    #
    # And when New SAS (cu 96) joined: 24640 cells; wrf-fatal and the LSM
    # seam x7/6, the GF/PBL seam unchanged, cloud optics +41 (the same 41
    # per added cumulus value), the remaining 2214 new cells admitted.
    # Re-pinned on the 2.9.0 round-2 merge (24640 -> 38080 cells, the six
    # microphysics ports joining the cumulus ports' matrix): wrf-fatal,
    # arwen-structural and the GF-without-PBL refusal scale by exactly 17/11
    # (7700 -> 11900, 1155 -> 1785, 715 -> 1105); the judged cloud-optics
    # refusal doubles (274 -> 548) because Purdue Lin (mp 2) is judged under
    # RTE+RRTMGP exactly as mp 30 is, 274 cells each; the remaining 22742
    # cells are admitted.  Every number was read off the merged tree's sweep.
    assert observed == {
        "admitted": 22742,
        "arwen-judged-cloud-optics": 548,
        "arwen-structural": 1785,
        "arwen-structural-gf-pbl": 1105,
        "wrf-fatal": 11900,
    }
