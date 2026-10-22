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


# What breaks without the two checks below: the forward merge of the exact
# chain onto 2.8.9 (e269e586a) kept 2.8.9's reader text in two places, and the
# strict A071 replay at that tip stopped being a PASS although every
# prognostic word still matched stock WRF 4.6.1 for the whole hour: RESM, PC
# and O3_GFS_DU were never read (so never written), and SNOALB kept the file's
# value where WRF replaces it from MAXALB (1,600 of 1,600 words differed at
# step 0).  Measured on golden/exact fc7a2c94a, G1, 2026-10-12.

def test_the_reader_does_not_ignore_native_input_records_history_writes():
    from gpuwm.ingest.wrfinput import IGNORED_WRFINPUT, MAPPED_WRFINPUT
    ignored = sorted(set(GROUPS["native_input"]) & IGNORED_WRFINPUT)
    assert not ignored, (
        f"read_wrfinput skips {ignored}, so the stock records the history "
        "schema promises are never carried from wrfinput_d01")
    assert set(GROUPS["native_input"]) <= MAPPED_WRFINPUT


def test_noah_cold_start_replaces_snoalb_from_maxalb_unless_rdmaxalb():
    # A source-level guard for a hunk a text merge has already dropped once;
    # the words themselves are graded by the strict replay, on a card.
    import inspect
    from gpuwm.ingest import wrfinput
    source = inspect.getsource(wrfinput.initialize_wrfinput_physics)
    noah = source[source.index("int(cfg.sf_surface_physics) == 2\n"):]
    noah = noah[:noah.index("elif int(cfg.sf_surface_physics) == 3")]
    assert "if not cfg.rdmaxalb:" in noah
    assert "wrf_noah_snow_albedo(" in noah and 'driver.fields["snoalb"]' in noah


def test_the_reader_carries_every_input_history_record_it_would_otherwise_ignore():
    # The third hunk the same merge dropped: 15 time-invariant stock records
    # (CFN, CFN1, CLAT, DZS, RDX, RDY, PCB, ZETATOP, ZS, MF_VX_INV, SHDAVG,
    # WATER_DEPTH and three flags) were no longer read, so the strict A071
    # hour passed on 197 compared fields with 15 excused as absent instead
    # of the 212 the exact chain compared (golden/exact 83a6116b2, G1).
    import inspect
    from gpuwm.ingest import wrfinput
    from gpuwm.io.wrf_input_history_schema import WRF_INPUT_HISTORY_ROWS
    overlap = sorted(set(WRF_INPUT_HISTORY_ROWS) & wrfinput.IGNORED_WRFINPUT)
    assert overlap, "the guard below exists for records in both tables"
    source = inspect.getsource(wrfinput.read_wrfinput)
    assert "carried |= frozenset(WRF_INPUT_HISTORY_ROWS)" in source
    assert "name in IGNORED_WRFINPUT and name not in carried" in source


def test_shin_hong_history_carries_the_registry_tke_and_length_records():
    # What breaks without this: four strict sweep rows (A121, A146, A185 on
    # iowa-convective, A146 on iowa-snow-ice) matched stock WRF 4.6.1 on
    # every field compared for all 20 steps and still failed, only because
    # TKE_PBL and EL_PBL (Registry package shinhongscheme) were never written.
    def shapes(pbl):
        return produced_history_shapes(RunConfig(
            nx=8, ny=6, nz=12, dx=3000., dy=3000., ztop=12000., dt=15.,
            run_seconds=60., moist=True, mp_physics=1, bl_pbl_physics=pbl,
            sf_sfclay_physics=91, sf_surface_physics=2, cu_physics=0,
            ra_physics=4, ra_rrtmg_variant="rrtmg_legacy",
            surface_energy_diag=False))
    shin_hong = shapes(11)
    assert shin_hong["TKE_PBL"] == (13, 6, 8) and shin_hong["EL_PBL"] == (13, 6, 8)
    assert "TKE_SHINHONG" in shin_hong  # the scheme's own diagnostic keeps its name
    ysu = shapes(1)
    assert "TKE_PBL" not in ysu and "EL_PBL" not in ysu
