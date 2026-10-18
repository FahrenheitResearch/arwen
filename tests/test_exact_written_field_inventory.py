"""Stock history records must survive the ordinary output door.

The expected records come from WRF 4.6.1 Registry.EM_COMMON and Registry.EM,
including the restart records requested in the combo recording. This test
checks presence and dimensions; real verify-exact replays grade their words.
"""
import pytest

from gpuwm.config import RunConfig
from gpuwm.io.history_layout import produced_history_shapes


GROUPS = {
    "noah": ("SMCREL", "SFROFF", "UDROFF", "ACSNOM", "NOAHRES"),
    "solar": ("COSZEN", "SR", "THM", "RTHRATEN"),
    "pressure": ("P_HYD",),
    "surface_energy": ("ACHFX", "ACLHF", "ACGRDFLX"),
    "pbl_rates": ("RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN", "RQCBLTEN"),
    "cumulus_rates": ("RUCUTEN", "RVCUTEN", "RTHCUTEN", "RQVCUTEN", "RQCCUTEN",
                      "RQRCUTEN", "RQICUTEN", "RQSCUTEN"),
    "sea": ("SST", "XICEM"),
    "scm": ("HFX_FORCE", "LH_FORCE", "TSK_FORCE", "HFX_FORCE_TEND",
            "LH_FORCE_TEND", "TSK_FORCE_TEND"),
    "inactive": ("SSTSK", "REFD_MAX"),
    "native_input": ("RESM", "PC", "O3_GFS_DU"),
    "radiation_boundary": tuple(
        prefix + side + suffix
        for prefix in ("", "AC") for side in ("SW", "LW")
        for suffix in ("UPT", "UPTC", "DNT", "DNTC", "UPB", "UPBC", "DNB", "DNBC")),
}


@pytest.mark.parametrize("microphysics,cumulus", [(1, 3), (6, 0)])
@pytest.mark.parametrize("group", GROUPS)
def test_normal_history_retains_stock_written_records(microphysics, cumulus, group):
    cfg = RunConfig(nx=8, ny=6, nz=12, dx=3000., dy=3000., ztop=12000.,
                    dt=15., run_seconds=60., moist=True, mp_physics=microphysics,
                    bl_pbl_physics=1, sf_sfclay_physics=91, sf_surface_physics=2,
                    cu_physics=cumulus, ra_physics=4,
                    ra_rrtmg_variant="rrtmg_legacy", surface_energy_diag=False)
    shapes = produced_history_shapes(cfg)
    from gpuwm.wrf_exact import ENABLED
    if group in ("pbl_rates", "cumulus_rates", "scm") and not ENABLED:
        # Strict-only records (physics_inventory.history_tendency_rows and
        # scm_forcing_fields): a default process writes none of them.
        assert not set(GROUPS[group]) & shapes.keys()
        return
    if group == "cumulus_rates" and cumulus == 0:
        assert not set(GROUPS[group]) & shapes.keys()
        return
    missing = sorted(set(GROUPS[group]) - shapes.keys())
    assert not missing, f"normal history omits stock {group} records: {missing}"
    for name in GROUPS[group]:
        expected = (6, 8)
        if name == "SMCREL":
            expected = (4, 6, 8)
        elif name in (*GROUPS["pbl_rates"], *GROUPS["cumulus_rates"], "THM", "RTHRATEN", "P_HYD"):
            expected = (12, 6, 8)
        elif name in (*GROUPS["scm"], "RESM"):
            expected = ()
        assert shapes[name] == expected, (name, shapes[name], expected)
