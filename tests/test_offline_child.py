from dataclasses import replace
from datetime import datetime, timedelta
import json

import netCDF4
import numpy as np
import pytest

from gpuwm.offline_child import (
    OFFLINE_CHILD_MP_PHYSICS,
    PARENT_SCHEME_CONTRACT,
    _resolve_source_physics,
    bind_parent_physics_from_wrf_namelist,
    build_offline_child_domain_state,
    MomentDiagnosisRequired,
    OfflineChildContractError,
    OfflineChildPlacement,
    build_offline_lateral_boundaries,
    inspect_parent_history_frame,
    interpolate_parent_boundary_snapshot,
    interpolate_parent_initial_state,
    map_microphysics_to_nssl18,
    read_parent_microphysics,
    validate_parent_history,
)
from gpuwm import netcdf_bridge
from gpuwm.config import RunConfig
from gpuwm.offline_child_run import (
    _ChildProgress,
    _create_output_root,
    _file_receipt,
    main as offline_child_main,
    _verify_file_receipts,
)


def _variable(dataset, name, dims, value):
    variable = dataset.createVariable(name, "f4", dims)
    variable[:] = np.asarray(value, dtype=np.float32)


def test_offline_child_output_root_is_create_only(tmp_path):
    requested = tmp_path / "new" / "child-run"
    assert _create_output_root(requested) == requested.resolve()
    marker = requested / "prior-evidence.txt"
    marker.write_text("preserve", encoding="utf-8")
    # The refusal is a sentence, not a Windows error number: the reader
    # who typed the same output directory twice is told what the
    # directory holds and the two ways out, and their evidence survives.
    with pytest.raises(OfflineChildContractError) as caught:
        _create_output_root(requested)
    assert "prior-evidence.txt" in str(caught.value)
    assert "--outdir" in str(caught.value)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_offline_child_parent_receipts_detect_midrun_input_change(tmp_path):
    parent = tmp_path / "parent.nc"
    parent.write_bytes(b"first")
    receipts = [_file_receipt(parent)]
    _verify_file_receipts(receipts, label="parent history input")
    parent.write_bytes(b"later")
    with pytest.raises(OfflineChildContractError, match="changed while"):
        _verify_file_receipts(receipts, label="parent history input")


def test_offline_child_capabilities_are_warning_only_and_exact(capsys):
    assert offline_child_main(["--show-capabilities"]) == 0
    capability = json.loads(capsys.readouterr().out)
    assert capability["schema"] == "gpuwm-offline-child-capabilities-v1"
    assert capability["explicit_expert_consent_required"] is False
    # 28 (Thompson aerosol-aware) joined the same-scheme list when the lane
    # learned to read its transported nc/nwfa/nifa and its two per-domain
    # surface-emission constants.  50 (P3) joined the same way, when the
    # lane learned its qv,qc,qr,qi + ni/nr + qir/qib inventory.  Both are
    # deliberately absent from cross_scheme_transitions, which stays empty:
    # every mixed edge touching either is refused by name (see the refusal
    # tests below), matching the online nest lane's
    # UNVALIDATED_MIXED_EDGE_SELECTORS.
    # 0, 1 and 9 joined with audit R-017: the lane reads a vapour-only
    # parent, a Kessler parent and a Milbrandt-Yau parent (the last through
    # its own scheme-qualified QHAIL/QNHAIL map), and gpuwm's history
    # writer publishes QNHAIL for mp=9 at last.
    assert capability["same_scheme_mp_physics"] == [
        0, 1, 6, 8, 9, 10, 18, 28, 50]
    assert capability["cross_scheme_transitions"] == []
    # Was pinned to False while the runner refused any child whose nz
    # differed from its parent's.  A child may now carry its OWN eta ladder
    # when it declares one, through the conservative host-side remap in
    # gpuwm/vertical_remap.py, so the declaration says what it does instead
    # of denying it exists.  A child that declares no ladder still inherits
    # its parent's, bitwise.
    assert capability["vertical_remapping"] == "conservative-offline-prepare"
    assert capability["output_ownership"] == "create-only"


def _physics_binding(tmp_path, *, mp=8, morr_rimed_ice=1):
    path = tmp_path / f"namelist-mp{mp}.input"
    rimed = (f" morr_rimed_ice = {morr_rimed_ice},\n"
             if mp == 10 else "")
    path.write_text(
        f"&physics\n mp_physics = {mp},\n{rimed}/\n",
        encoding="utf-8")
    return bind_parent_physics_from_wrf_namelist(path)


def _history(path, valid_time, *, mp=8, hgt_offset=0.0, signal=0.0,
             ny=3, nx=4, omit_aerosol_surface_emission=False,
             qnrain=None, qir=None, producer="gpuwm"):
    nz = 2
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("DateStrLen", 19), ("west_east", nx),
                ("south_north", ny), ("bottom_top", nz),
                ("west_east_stag", nx + 1),
                ("south_north_stag", ny + 1),
                ("bottom_top_stag", nz + 1)):
            dataset.createDimension(name, size)
        dataset.TITLE = ("gpuwm offline-child fixture" if producer == "gpuwm"
                         else " OUTPUT FROM WRF V4.6.1 MODEL")
        dataset.DX = 1000.0
        dataset.DY = 1000.0
        dataset.MAP_PROJ = 1
        dataset.TRUELAT1 = 30.0
        dataset.TRUELAT2 = 60.0
        dataset.STAND_LON = -97.0
        dataset.CEN_LAT = 35.0
        dataset.CEN_LON = -97.0
        dataset.HYBRID_OPT = 2
        dataset.ETAC = 0.2
        if producer == "gpuwm":
            dataset.GPUWM_WRITE_COMPLETE = 1
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.frombuffer(
            valid_time.strftime("%Y-%m-%d_%H:%M:%S").encode(), dtype="S1")
        mass3 = ("Time", "bottom_top", "south_north", "west_east")
        u3 = ("Time", "bottom_top", "south_north", "west_east_stag")
        v3 = ("Time", "bottom_top", "south_north_stag", "west_east")
        w3 = ("Time", "bottom_top_stag", "south_north", "west_east")
        mass2 = ("Time", "south_north", "west_east")
        # P3 (mp=50) declares NO qs and NO qg (Registry.EM_COMMON:3038:
        # ``moist:qv,qc,qr,qi``), so a faithful P3 archive omits both, and
        # its QICE is written nonzero below so the rime pair has ice to
        # describe.  Every other fixture keeps the six-species zeros.
        # mp 0 and 1 carry the warm-rain trio and no frozen species: WRF's
        # Kessler declares moist:qv,qc,qr (Registry.EM_COMMON:3015) and
        # gpuwm's own mp=0 advects the same three
        # (gpuwm/offline_child.py::_transported_source_fields), which is
        # exactly why the inventory cannot separate them on a gpuwm tape.
        zero_masses = (("P", "QVAPOR", "QCLOUD", "QRAIN")
                       if mp in {0, 1, 50} else
                       ("P", "QVAPOR", "QCLOUD", "QRAIN",
                        "QICE", "QSNOW", "QGRAUP"))
        for name in zero_masses:
            _variable(dataset, name, mass3, np.zeros((1, nz, ny, nx)))
        pb = np.broadcast_to(
            np.asarray([65000.0, 30000.0], dtype=np.float32)[None, :, None, None],
            (1, nz, ny, nx))
        _variable(dataset, "PB", mass3, pb)
        _variable(dataset, "T", mass3,
                  np.full((1, nz, ny, nx), signal))
        _variable(dataset, "U", u3,
                  np.full((1, nz, ny, nx + 1), 5.0 + signal))
        _variable(dataset, "V", v3,
                  np.full((1, nz, ny + 1, nx), -2.0 + signal))
        for name in ("W", "PH"):
            _variable(dataset, name, w3, np.zeros((1, nz + 1, ny, nx)))
        phb = np.broadcast_to(
            np.asarray([0.0, 40000.0, 90000.0], dtype=np.float32)
            [None, :, None, None], (1, nz + 1, ny, nx))
        _variable(dataset, "PHB", w3, phb)
        _variable(dataset, "MU", mass2,
                  np.full((1, ny, nx), signal))
        _variable(dataset, "MUB", mass2,
                  np.full((1, ny, nx), 80000.0))
        _variable(dataset, "MAPFAC_M", mass2, np.ones((1, ny, nx)))
        _variable(dataset, "MAPFAC_U", ("Time", "south_north", "west_east_stag"),
                  np.ones((1, ny, nx + 1)))
        _variable(dataset, "MAPFAC_V", ("Time", "south_north_stag", "west_east"),
                  np.ones((1, ny + 1, nx)))
        _variable(dataset, "HGT", mass2,
                  np.full((1, ny, nx), hgt_offset))
        _variable(dataset, "PSFC", mass2,
                  np.full((1, ny, nx), 90000.0))
        _variable(dataset, "F", mass2,
                  np.full((1, ny, nx), 8.0e-5))
        _variable(dataset, "E", mass2,
                  np.full((1, ny, nx), 1.0e-4))
        _variable(dataset, "SINALPHA", mass2,
                  np.zeros((1, ny, nx)))
        _variable(dataset, "COSALPHA", mass2,
                  np.ones((1, ny, nx)))
        _variable(dataset, "XLAT", mass2,
                  np.full((1, ny, nx), 35.0))
        _variable(dataset, "XLONG", mass2,
                  np.full((1, ny, nx), -97.0))
        _variable(dataset, "P_TOP", ("Time",), [10000.0])
        _variable(dataset, "ZNU", ("Time", "bottom_top"), [[0.75, 0.25]])
        _variable(dataset, "ZNW", ("Time", "bottom_top_stag"),
                  [[1.0, 0.5, 0.0]])
        if mp in {8, 10, 50}:
            _variable(dataset, "QNRAIN", mass3,
                      np.full((1, nz, ny, nx),
                              0.0 if qnrain is None else qnrain))
            _variable(dataset, "QNICE", mass3,
                      np.zeros((1, nz, ny, nx)))
        if mp == 10:
            _variable(dataset, "QNSNOW", mass3,
                      np.zeros((1, nz, ny, nx)))
            _variable(dataset, "QNGRAUPEL", mass3,
                      np.zeros((1, nz, ny, nx)))
        if mp == 28:
            # Thompson aerosol-aware.  Registry.EM_COMMON:3036 declares
            # scalar:qni,qnr,qnc,qnwfa,qnifa (qnbca is wif_input_opt=2 only,
            # out of scope) plus state:qnwfa2d,qnifa2d.  Deliberately
            # DISTINCT nonzero values per field so an interpolation that
            # crossed two of them would be visible.
            _variable(dataset, "QNRAIN", mass3, np.zeros((1, nz, ny, nx)))
            _variable(dataset, "QNICE", mass3, np.zeros((1, nz, ny, nx)))
            _variable(dataset, "QNCLOUD", mass3,
                      np.full((1, nz, ny, nx), 1.0e8))
            _variable(dataset, "QNWFA", mass3,
                      np.full((1, nz, ny, nx), 1.5e8))
            _variable(dataset, "QNIFA", mass3,
                      np.full((1, nz, ny, nx), 2.5e5))
            if not omit_aerosol_surface_emission:
                _variable(dataset, "QNWFA2D", mass2,
                          np.full((1, ny, nx), 4321.0))
                _variable(dataset, "QNIFA2D", mass2,
                          np.full((1, ny, nx), 0.0))
        if mp == 16:
            # WDM6 (Registry.EM_COMMON:3031, scalar:qnn,qnc,qnr): the six
            # masses above plus a warm-rain number pair and the CCN
            # reservoir, and NO ice number, which is the discriminant.
            _variable(dataset, "QNCCN", mass3,
                      np.full((1, nz, ny, nx), 1.0e8))
            _variable(dataset, "QNCLOUD", mass3,
                      np.full((1, nz, ny, nx), 1.0e8))
            _variable(dataset, "QNRAIN", mass3, np.zeros((1, nz, ny, nx)))
        if mp == 50:
            # P3 one-category.  QNRAIN/QNICE already landed above; the rest
            # is the ice mass and its prognostic rime pair, with DISTINCT
            # values so a crossed interpolation would be visible, keeping
            # qir <= qi and rime density qir/qib = 1.0e-4 / 2.0e-7 =
            # 500 kg m-3, inside P3's admissible [50, 900] band.
            _variable(dataset, "QICE", mass3,
                      np.full((1, nz, ny, nx), 3.0e-4))
            _variable(dataset, "QIR", mass3,
                      np.full((1, nz, ny, nx),
                              1.0e-4 if qir is None else qir))
            _variable(dataset, "QIB", mass3,
                      np.full((1, nz, ny, nx), 2.0e-7))


def test_parent_history_contract_checks_geometry_cadence_and_scheme(tmp_path):
    start = datetime(1974, 4, 3, 12)
    paths = [tmp_path / f"parent-{index}.nc" for index in range(2)]
    for index, path in enumerate(paths):
        _history(path, start + timedelta(minutes=5 * index))
    contract = validate_parent_history(
        paths, max_boundary_interval_seconds=900, source_mp_physics=8)
    assert contract.interval_seconds == 300.0
    assert contract.source_kind == "gpuwm"
    assert contract.source_mp_physics == 8
    assert contract.start_time == start
    assert contract.end_time == start + timedelta(minutes=5)
    with pytest.raises(OfflineChildContractError, match="exceeds"):
        validate_parent_history(paths, max_boundary_interval_seconds=299)


def test_offline_child_refuses_feedback_modified_parent_provenance(tmp_path):
    path = tmp_path / "feedback-parent.nc"
    _history(path, datetime(1974, 4, 3, 12))
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.GPUWM_FEEDBACK = "experimental"
        dataset.GPUWM_FEEDBACK_VALUE = 1
    with pytest.raises(
            OfflineChildContractError,
            match="two-way feedback provenance.*one-way parent"):
        inspect_parent_history_frame(path)


def test_wrf_namelist_binding_is_authoritative_and_digested(tmp_path):
    binding = _physics_binding(tmp_path, mp=10, morr_rimed_ice=0)
    assert binding.mp_physics == 10
    assert binding.morr_rimed_ice == 0
    assert binding.domain_id == 1
    assert binding.evidence_kind == "wrf-namelist"
    assert len(binding.evidence_sha256) == 64


def test_parent_history_rejects_changed_static_geometry(tmp_path):
    start = datetime(1974, 4, 3, 12)
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    _history(first, start)
    _history(second, start + timedelta(minutes=5), hgt_offset=1.0)
    with pytest.raises(OfflineChildContractError, match="geometry/static"):
        validate_parent_history(
            (first, second), max_boundary_interval_seconds=900)


def test_parent_history_reader_and_moisture_inventory(tmp_path):
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10)
    info = inspect_parent_history_frame(path)
    assert info.source_mp_physics is None
    assert info.inferred_mp_physics == 10
    moisture = read_parent_microphysics(path)
    assert set(moisture) == {
        "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng"}
    assert moisture["qv"].shape == (2, 3, 4)


def _mass_fields(shape=(2, 3, 4)):
    return {name: np.zeros(shape, dtype=np.float32)
            for name in ("qv", "qc", "qr", "qi", "qs", "qg")}


def test_thompson_to_nssl_carries_available_moments_and_initializes_background():
    fields = _mass_fields()
    fields["qr"][0, 0, 0] = 1.0e-4
    fields["nr"] = np.full(fields["qr"].shape, 123.0, dtype=np.float32)
    mapped, receipt = map_microphysics_to_nssl18(
        fields, source_mp_physics=8)
    assert tuple(mapped) == (
        "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
        "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh")
    assert np.array_equal(mapped["qnr"], fields["nr"])
    assert np.all(mapped["qh"] == 0.0)
    assert np.all(mapped["qnn"] == np.float32(0.5e9 / 1.225))
    assert receipt["category_mapping"] == "graupel-to-graupel"


def test_active_mass_without_target_number_diagnosis_fails_closed():
    fields = _mass_fields()
    fields["qs"][0, 0, 0] = 2.0e-5
    with pytest.raises(MomentDiagnosisRequired, match="qns"):
        map_microphysics_to_nssl18(fields, source_mp_physics=8)

    def diagnose(mapped, names):
        return {name: np.full(mapped["qv"].shape, 7.0, dtype=np.float32)
                for name in names}

    mapped, receipt = map_microphysics_to_nssl18(
        fields, source_mp_physics=8, diagnose_missing=diagnose)
    assert np.all(mapped["qns"] == 7.0)
    assert receipt["diagnosed_target_moments"] == ("qns",)

    fields["qs"][0, 0, 0] = -1.0e-5
    with pytest.raises(OfflineChildContractError, match="negative"):
        map_microphysics_to_nssl18(fields, source_mp_physics=8)


def test_morrison_hail_receipt_is_required_and_reclassifies_rimed_category():
    fields = _mass_fields()
    fields["qg"][0, 0, 0] = 1.0e-3
    fields["ng"] = np.full(fields["qg"].shape, 11.0, dtype=np.float32)
    with pytest.raises(OfflineChildContractError, match="morr_rimed_ice"):
        map_microphysics_to_nssl18(fields, source_mp_physics=10)
    mapped, receipt = map_microphysics_to_nssl18(
        fields, source_mp_physics=10, morr_rimed_ice=1)
    assert np.all(mapped["qg"] == 0.0)
    assert np.array_equal(mapped["qh"], fields["qg"])
    assert np.array_equal(mapped["qnh"], fields["ng"])
    assert receipt["category_mapping"] == "morrison-hail-to-nssl-hail"


def test_nssl18_passthrough_is_exact_and_complete():
    names = (
        "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
        "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh")
    fields = {
        name: np.full((2, 3, 4), index + 1, dtype=np.float32)
        for index, name in enumerate(names)}
    mapped, receipt = map_microphysics_to_nssl18(
        fields, source_mp_physics=18)
    assert tuple(mapped) == names
    assert all(np.array_equal(mapped[name], fields[name]) for name in names)
    assert receipt["category_mapping"] == "nssl18-passthrough"

    fields.pop("qvolh")
    with pytest.raises(OfflineChildContractError, match="lacks transported"):
        map_microphysics_to_nssl18(fields, source_mp_physics=18)


def test_inventory_inference_is_advisory_when_companion_binds_physics(tmp_path):
    path = tmp_path / "dormant-inventory.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10)
    info = inspect_parent_history_frame(path, source_mp_physics=18)
    assert info.source_mp_physics == 18
    assert info.inferred_mp_physics == 10


@pytest.mark.parametrize("parent_mp", [16])
def test_offline_child_parent_scheme_refusal_names_the_switch(
        tmp_path, parent_mp):
    """A genuine unreadable-parent refusal, watched at all four gates.

    This route's parent contract is NOT a profile whitelist: it names the
    parents whose transported state the lane can actually read.  mp 0
    (vapour plus the warm-rain pair) and mp 1 (Kessler) LEFT this test
    with audit R-017 -- their inventories are the prefix the lane already
    built, and refusing them was the admission set contradicting the
    lane's own helper -- and mp=9 joined the admitted set with them once
    the lane learned its scheme-qualified QHAIL/QNHAIL rows.  What still
    stands here is mp=16 (WDM6), and it stands on a named breakage: nn and
    NSSL's qnn both publish under QNCCN, so the field map has no
    unambiguous row and a WDM6 child would start with a zero-filled CCN
    reservoir.  What the 2026-07-31 suite ruling demands is that the
    refusal name the exact switch and its value, and be pinned rather than
    believed.
    """

    namelist = tmp_path / f"namelist-mp{parent_mp}.input"
    namelist.write_text(
        f"&physics\n mp_physics = {parent_mp},\n/\n", encoding="utf-8")
    with pytest.raises(
            OfflineChildContractError,
            match=f"mp_physics={parent_mp}"):
        bind_parent_physics_from_wrf_namelist(namelist)

    frame = tmp_path / "parent.nc"
    _history(frame, datetime(1974, 4, 3, 12))
    with pytest.raises(
            OfflineChildContractError,
            match=f"mp_physics={parent_mp}"):
        inspect_parent_history_frame(frame, source_mp_physics=parent_mp)

    with pytest.raises(
            OfflineChildContractError,
            match=f"mp_physics={parent_mp}"):
        map_microphysics_to_nssl18({}, source_mp_physics=parent_mp)

    with pytest.raises(
            OfflineChildContractError,
            match=f"mp_physics={parent_mp}"):
        _resolve_source_physics(parent_mp, None, None)

    # The four enforcement points read ONE named contract, so a scheme
    # can never be carried by one of them and refused by another.
    # Re-measured with audit R-017: 0, 1 and 9 are READ by this lane now,
    # and each is held out of the CROSS-scheme contract by its own named
    # reason rather than by being unreadable
    # (_OFFLINE_CROSS_LEG_UNBUILT_REASONS).
    # mp=28 joined when its online mixed edge was ratified and the derived
    # closure mirror emptied: its masses are classic Thompson's and the
    # NSSL conversion is the one mp=8 already runs.
    assert PARENT_SCHEME_CONTRACT == frozenset({6, 8, 10, 18, 28})
    assert {0, 1, 9} <= OFFLINE_CHILD_MP_PHYSICS
    for mp in (0, 1, 9):
        with pytest.raises(OfflineChildContractError,
                           match="offline cross-physics conversion"):
            map_microphysics_to_nssl18({}, source_mp_physics=mp)


def test_conservative_parent_snapshot_and_streamed_lateral_intervals(tmp_path):
    start = datetime(1974, 4, 3, 12)
    paths = (tmp_path / "parent-0.nc", tmp_path / "parent-1.nc")
    _history(paths[0], start, ny=18, nx=20, signal=0.0)
    _history(paths[1], start + timedelta(minutes=5),
             ny=18, nx=20, signal=1.0)
    placement = OfflineChildPlacement(
        parent_nx=20, parent_ny=18, child_nx=12, child_ny=10,
        parent_grid_ratio=1, i_parent_start=4, j_parent_start=4)
    snapshot = interpolate_parent_boundary_snapshot(
        paths[0], placement, source_mp_physics=8, backend="cpu")
    assert snapshot.fields["u"].shape == (2, 10, 13)
    assert snapshot.fields["v"].shape == (2, 11, 12)
    assert snapshot.fields["w"].shape == (3, 10, 12)
    assert snapshot.fields["mu"].shape == (1, 10, 12)
    assert set(snapshot.fields) == {
        "u", "v", "w", "theta", "phi", "mu",
        "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni"}

    binding = _physics_binding(tmp_path, mp=8)
    contract = validate_parent_history(
        paths, max_boundary_interval_seconds=900, physics_binding=binding)
    result = build_offline_lateral_boundaries(
        contract, placement, backend="cpu")
    assert len(result.boundaries.intervals) == 1
    interval = result.boundaries.intervals[0]
    assert interval.start_seconds == 0.0
    assert interval.end_seconds == 300.0
    assert set(interval.fields) == set(snapshot.fields)
    assert np.isfinite(interval.fields["theta"].west.tendency).all()
    assert np.any(interval.fields["theta"].west.tendency != 0.0)


def test_thompson_parent_boundary_can_change_to_nssl_inventory(tmp_path):
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), ny=18, nx=20)
    placement = OfflineChildPlacement(
        parent_nx=20, parent_ny=18, child_nx=12, child_ny=10,
        parent_grid_ratio=1, i_parent_start=4, j_parent_start=4)
    snapshot = interpolate_parent_boundary_snapshot(
        path, placement, source_mp_physics=8, target_mp_physics=18)
    assert set(snapshot.fields) == {
        "u", "v", "w", "theta", "phi", "mu", "qv", "qc", "qr",
        "qi", "qs", "qg", "qh", "qndrop", "qnr", "qni", "qns",
        "qng", "qnh", "qnn", "qvolg", "qvolh"}
    assert np.all(snapshot.fields["qh"] == 0.0)
    assert snapshot.receipt["conversion"]["target_mp_physics"] == 18


def test_parent_initial_state_builds_standalone_numpy_domain(tmp_path):
    path = tmp_path / "parent.nc"
    valid_time = datetime(1974, 4, 3, 12)
    _history(path, valid_time, ny=18, nx=20, signal=0.5)
    placement = OfflineChildPlacement(
        parent_nx=20, parent_ny=18, child_nx=12, child_ny=10,
        parent_grid_ratio=1, i_parent_start=4, j_parent_start=4)
    initial = interpolate_parent_initial_state(
        path, placement, physics_binding=_physics_binding(tmp_path, mp=8),
        backend="cpu")
    cfg = RunConfig(
        nx=12, ny=10, nz=2, dx=1000.0, dy=1000.0,
        ztop=9000.0, dt=5.0, run_seconds=300.0,
        hybrid_opt=2, etac=0.2, moist=True, mp_physics=8,
        specified=True, nested=False, terrain_opt=1, map_proj=1,
        hypsometric_opt=2)
    state = build_offline_child_domain_state(
        initial, cfg, array_module=np)

    assert initial.valid_time == valid_time
    assert initial.receipt["terrain_policy"] == "sint-parent-inherited"
    assert "held physics tendencies" in initial.receipt["spinup_policy"]
    assert initial.receipt["source_physics_binding"]["evidence_kind"] == \
        "wrf-namelist"
    assert state.u.shape == (2, 10, 13)
    assert state.v.shape == (2, 11, 12)
    assert state.thb.shape == (2, 10, 12)
    assert np.all(state.u == np.float32(5.5))
    assert np.all(state.qv == 0.0)
    assert np.array_equal(state.u0, state.u)
    assert np.array_equal(state.nr0, state.nr)
    assert state.rotational
    for field in (state.p, state.al, state.alt, state.thp, state.phb):
        assert np.isfinite(field).all()

    # WRF history stores ETAC in FP32; the corresponding round-trip must not
    # reject the decimal RunConfig value as a different vertical coordinate.
    fp32_receipt = dict(initial.receipt)
    fp32_receipt["etac"] = float(np.float32(0.2))
    fp32_initial = replace(initial, receipt=fp32_receipt)
    build_offline_child_domain_state(fp32_initial, cfg, array_module=np)

    incompatible = RunConfig(
        nx=12, ny=10, nz=2, dx=1000.0, dy=1000.0,
        ztop=9000.0, dt=5.0, run_seconds=300.0,
        hybrid_opt=1, etac=0.2, moist=True, mp_physics=8,
        specified=True, nested=False, terrain_opt=1, map_proj=1,
        hypsometric_opt=2)
    with pytest.raises(OfflineChildContractError, match="hybrid_opt"):
        build_offline_child_domain_state(initial, incompatible, array_module=np)


# ---------------------------------------------------------------------------
# mp_physics=28 (Thompson aerosol-aware) -- the offline-child lane decision.
#
# The lane admits 28 for the SAME-SCHEME case (a 28 parent forcing a 28
# child) and refuses every CROSS-scheme edge touching it by name.  Both
# halves are asserted here, because "half-supported" is exactly the state
# this work package existed to eliminate.
# ---------------------------------------------------------------------------

def _mp28_placement():
    return OfflineChildPlacement(
        parent_nx=20, parent_ny=18, child_nx=12, child_ny=10,
        parent_grid_ratio=1, i_parent_start=4, j_parent_start=4)


def _mp28_child_config():
    return RunConfig(
        nx=12, ny=10, nz=2, dx=1000.0, dy=1000.0,
        ztop=9000.0, dt=5.0, run_seconds=300.0,
        hybrid_opt=2, etac=0.2, moist=True, mp_physics=28,
        specified=True, nested=False, terrain_opt=1, map_proj=1,
        hypsometric_opt=2)


def test_mp28_parent_history_is_read_and_inferred_ahead_of_classic_thompson(
        tmp_path):
    """An mp=28 stream must not be advertised as classic Thompson.

    mp=28's wrfout inventory is a strict SUPERSET of mp=8's -- it carries
    QNRAIN/QNICE too -- so the advisory inference has to test the aerosol
    pair first or every aerosol-aware archive reports 8 in its receipt.
    """
    path = tmp_path / "parent-mp28.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20)
    info = inspect_parent_history_frame(path, source_mp_physics=28)
    assert info.source_mp_physics == 28
    assert info.inferred_mp_physics == 28

    classic = tmp_path / "parent-mp8.nc"
    _history(classic, datetime(1974, 4, 3, 12), mp=8, ny=18, nx=20)
    assert inspect_parent_history_frame(classic).inferred_mp_physics == 8


def test_mp28_transported_inventory_reaches_the_child_state(tmp_path):
    """The five mp=28 scalars plus the two surface constants land on state."""
    path = tmp_path / "parent-mp28.nc"
    valid_time = datetime(1974, 4, 3, 12)
    _history(path, valid_time, mp=28, ny=18, nx=20, signal=0.5)
    initial = interpolate_parent_initial_state(
        path, _mp28_placement(),
        physics_binding=_physics_binding(tmp_path, mp=28), backend="cpu")

    # Registry.EM_COMMON:3036's scalar list, minus qnbca (wif_input_opt=2).
    assert set(initial.microphysics) == {
        "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni",
        "nc", "nwfa", "nifa"}

    state = build_offline_child_domain_state(
        initial, _mp28_child_config(), array_module=np)
    assert np.all(state.nc == np.float32(1.0e8))
    assert np.all(state.nwfa == np.float32(1.5e8))
    assert np.all(state.nifa == np.float32(2.5e5))
    # The RK time-t copies mp=28 alone owns.
    assert np.array_equal(state.nc0, state.nc)
    assert np.array_equal(state.nwfa0, state.nwfa)
    assert np.array_equal(state.nifa0, state.nifa)
    # And the per-domain surface emission constants, which nothing in the
    # child can re-derive: thompson_init's nwfa2d fill
    # (module_mp_thompson.F:510) runs only on the "no initial CCN" branch,
    # and this child HAS initial CCN.
    assert np.all(state.nwfa2d == np.float32(4321.0))
    assert np.all(state.nifa2d == np.float32(0.0))
    assert state.nwfa2d.shape == (10, 12)


def test_mp28_child_refuses_a_parent_without_surface_aerosol_emission(
        tmp_path):
    """A parent history missing QNWFA2D must fail loud, not default to zero.

    Zero surface emission is a legal, finite, bounded mp=28 forecast that
    nothing anywhere would flag -- WRF's terminal clamps
    (module_mp_thompson.F:3976-3982) hold the aerosol at its floors.  That
    is precisely why the absence has to raise here.
    """
    path = tmp_path / "parent-mp28-noemit.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20,
             omit_aerosol_surface_emission=True)
    with pytest.raises(OfflineChildContractError, match="QNWFA2D"):
        interpolate_parent_initial_state(
            path, _mp28_placement(),
            physics_binding=_physics_binding(tmp_path, mp=28),
            backend="cpu")


def test_mp28_boundary_snapshot_carries_the_aerosol_scalars(tmp_path):
    path = tmp_path / "parent-mp28.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20)
    snapshot = interpolate_parent_boundary_snapshot(
        path, _mp28_placement(), source_mp_physics=28, backend="cpu")
    assert set(snapshot.fields) == {
        "u", "v", "w", "theta", "phi", "mu",
        "qv", "qc", "qr", "qi", "qs", "qg",
        "nr", "ni", "nc", "nwfa", "nifa"}


def test_mp28_streamed_lateral_intervals_include_the_aerosol_tracers(
        tmp_path):
    start = datetime(1974, 4, 3, 12)
    paths = (tmp_path / "p0.nc", tmp_path / "p1.nc")
    _history(paths[0], start, mp=28, ny=18, nx=20, signal=0.0)
    _history(paths[1], start + timedelta(minutes=5), mp=28,
             ny=18, nx=20, signal=1.0)
    binding = _physics_binding(tmp_path, mp=28)
    contract = validate_parent_history(
        paths, max_boundary_interval_seconds=900, physics_binding=binding)
    assert contract.source_mp_physics == 28
    result = build_offline_lateral_boundaries(
        contract, _mp28_placement(), backend="cpu")
    interval = result.boundaries.intervals[0]
    assert {"nc", "nwfa", "nifa"} <= set(interval.fields)
    assert np.isfinite(interval.fields["nwfa"].west.value).all()


#: Reading a parent history file goes through the Rust NetCDF decoder
#: (gpuwm.netcdf_bridge.NetcdfBridgeMissing otherwise).  Per test rather than
#: module-wide: the rest of this deck writes its fixtures with netCDF4 and
#: asks gpuwm to decode none of them.
needs_netcdf_bridge = pytest.mark.skipif(
    netcdf_bridge.find_netcdf_bin() is None,
    reason="rw_netcdf is not built; build tools/rustwx to run this")


@needs_netcdf_bridge
def test_mp28_offline_converts_to_nssl_and_drops_only_its_aerosols(tmp_path):
    """The closure-mirror refusal retired with the online one.

    mp=28 carries classic Thompson's six masses and adds nc/nwfa/nifa.
    The NSSL conversion is the one mp=8 already runs -- nc rides the
    qndrop alias, the two aerosol tracers have no NSSL counterpart and
    are dropped -- so there was nothing left for the mirror to refuse.
    """
    aero = tmp_path / "parent-mp28.nc"
    _history(aero, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20)
    placement = _mp28_placement()

    assert 28 in PARENT_SCHEME_CONTRACT
    state = interpolate_parent_initial_state(
        aero, placement, source_mp_physics=28, target_mp_physics=18,
        backend="cpu")
    assert state is not None
    snapshot = interpolate_parent_boundary_snapshot(
        aero, placement, source_mp_physics=28, target_mp_physics=18,
        backend="cpu")
    assert snapshot is not None


def test_every_admitted_parent_scheme_has_a_transport_mapping():
    # The contract set and the transport function are two spellings of the
    # same promise.  WSM6 (mp=6) was admitted by OFFLINE_CHILD_MP_PHYSICS
    # and then refused by _transported_source_fields, which never
    # terminated its chain for a single-moment parent -- a stock WSM6
    # archive hit "unsupported" from a path whose contract said supported
    # (reported by a user against 2.5.2).
    from gpuwm.offline_child import (
        OFFLINE_CHILD_MP_PHYSICS,
        _transported_source_fields,
    )

    for mp in sorted(OFFLINE_CHILD_MP_PHYSICS):
        fields = _transported_source_fields(mp)
        assert fields, f"mp_physics={mp} transports no fields"


def test_wsm6_transports_the_single_moment_set():
    from gpuwm.offline_child import _transported_source_fields

    assert _transported_source_fields(6) == (
        "qv", "qc", "qr", "qi", "qs", "qg")


# The concrete breakage, measured 2026-08-29: a ratio-3 downscale of an
# mp=10 parent aborts at its first radiation call with
#   nr must be finite and non-negative: first_index=(161115, 18),
#   first_value=-3.7252903e-09, negative_count=626, nonfinite_count=0
# -- 626 cells of float32 SINT rounding out of 22.5 million, in an interior
# column.  The fix-up for exactly that artefact had existed and been
# measured since the online nest lane needed it, but reached only children
# built through a DomainState; the offline downscale initializer holds its
# SINT output in a dict and silently went without.  These two tests pin
# both halves: the artefact is cleared, and a real defect still is not.
_SINT_ROUNDING_UNDERSHOOT = -3.7252903e-09


def _mp10_placement():
    return OfflineChildPlacement(
        parent_nx=20, parent_ny=18, child_nx=12, child_ny=10,
        parent_grid_ratio=1, i_parent_start=4, j_parent_start=4)


def test_offline_initial_state_clamps_sint_rounding_undershoot(tmp_path):
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10, ny=18, nx=20,
             qnrain=_SINT_ROUNDING_UNDERSHOOT)
    initial = interpolate_parent_initial_state(
        path, _mp10_placement(),
        physics_binding=_physics_binding(tmp_path, mp=10), backend="cpu")

    nr = initial.microphysics["nr"]
    assert nr.min() == 0.0
    account = initial.receipt["positive_definite_clamp"]["nr"]
    assert account["cells"] == nr.size
    assert account["most_negative"] == pytest.approx(
        _SINT_ROUNDING_UNDERSHOOT, rel=1e-6)


def test_offline_initial_state_leaves_a_real_negative_for_the_gate(tmp_path):
    # A thousand per kilogram is not rounding.  The clamp has a ceiling so
    # that the engine's own refusal -- the one that caught the real case --
    # still has something to catch.
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10, ny=18, nx=20,
             qnrain=-1.0e3)
    initial = interpolate_parent_initial_state(
        path, _mp10_placement(),
        physics_binding=_physics_binding(tmp_path, mp=10), backend="cpu")

    assert initial.microphysics["nr"].min() < 0.0
    assert initial.receipt["positive_definite_clamp"] == {}


def test_offline_boundary_snapshot_clamps_coupled_undershoot(tmp_path):
    # The boundary lane SINTs COUPLED moments, so the absolute floor has to
    # travel with the coupling or it clamps nothing.  Same parent field as
    # the initial-state case; here it arrives multiplied by chm ~ 8e4.
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10, ny=18, nx=20,
             qnrain=_SINT_ROUNDING_UNDERSHOOT)
    snapshot = interpolate_parent_boundary_snapshot(
        path, _mp10_placement(),
        physics_binding=_physics_binding(tmp_path, mp=10), backend="cpu")

    assert snapshot.fields["nr"].min() == 0.0
    assert snapshot.receipt["positive_definite_clamp"]["nr"]["cells"] > 0


def test_offline_boundary_snapshot_leaves_a_real_negative_for_the_gate(tmp_path):
    path = tmp_path / "parent.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=10, ny=18, nx=20,
             qnrain=-1.0e3)
    snapshot = interpolate_parent_boundary_snapshot(
        path, _mp10_placement(),
        physics_binding=_physics_binding(tmp_path, mp=10), backend="cpu")

    assert snapshot.fields["nr"].min() < 0.0
    assert snapshot.receipt["positive_definite_clamp"] == {}


# ---------------------------------------------------------------------------
# mp_physics=50 (P3, one-category ice with prognostic riming) -- the
# offline-child lane decision, on mp=28's exact terms: admitted for the
# SAME-SCHEME case (a 50 parent forcing a 50 child), every cross-scheme
# edge refused by name through the DERIVED mirror of the online lane's
# UNVALIDATED_MIXED_EDGE_SELECTORS.  Transported set per
# Registry.EM_COMMON:3038: moist qv,qc,qr,qi (no qs, no qg) plus scalar
# qni,qnr,qir,qib.
# ---------------------------------------------------------------------------

def _p3_placement():
    return OfflineChildPlacement(
        parent_nx=20, parent_ny=18, child_nx=12, child_ny=10,
        parent_grid_ratio=1, i_parent_start=4, j_parent_start=4)


def _p3_child_config():
    return RunConfig(
        nx=12, ny=10, nz=2, dx=1000.0, dy=1000.0,
        ztop=9000.0, dt=5.0, run_seconds=300.0,
        hybrid_opt=2, etac=0.2, moist=True, mp_physics=50,
        specified=True, nested=False, terrain_opt=1, map_proj=1,
        hypsometric_opt=2)


def test_p3_parent_history_is_inferred_ahead_of_classic_thompson(tmp_path):
    """A P3 stream must not be advertised as classic Thompson.

    P3's wrfout inventory carries QNRAIN/QNICE beside its rime pair, so
    the advisory inference has to test QIR/QIB first -- the mp=28-vs-mp=8
    superset problem again, with a different discriminant: QIR/QIB are
    declared by no other scheme.
    """
    path = tmp_path / "parent-mp50.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=50, ny=18, nx=20)
    info = inspect_parent_history_frame(path, source_mp_physics=50)
    assert info.source_mp_physics == 50
    assert info.inferred_mp_physics == 50


def test_p3_transported_inventory_reaches_the_child_state(tmp_path):
    """qv,qc,qr,qi + ni/nr + the rime pair land on state, and NOTHING else.

    A P3 state allocates no qs/qg at all (one ice category), so the
    right assertion is two-sided: the eight transported fields arrive,
    and the six-species leftovers do not exist to be zero-filled.
    """
    path = tmp_path / "parent-mp50.nc"
    valid_time = datetime(1974, 4, 3, 12)
    _history(path, valid_time, mp=50, ny=18, nx=20, signal=0.5)
    initial = interpolate_parent_initial_state(
        path, _p3_placement(),
        physics_binding=_physics_binding(tmp_path, mp=50), backend="cpu")

    assert set(initial.microphysics) == {
        "qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"}

    state = build_offline_child_domain_state(
        initial, _p3_child_config(), array_module=np)
    assert np.all(state.qi == np.float32(3.0e-4))
    assert np.all(state.qir == np.float32(1.0e-4))
    assert np.all(state.qib == np.float32(2.0e-7))
    # One ice category: P3 allocates neither qs nor qg
    # (gpuwm/core/state.py mp==50 branch), and this route must not have
    # fabricated them.
    assert getattr(state, "qs", None) is None
    assert getattr(state, "qg", None) is None
    # The RK time-t copies, seeded through the online lane's own table
    # (gpuwm/ingest/nest_init.py::RK_TIME_T_SEED_PAIRS) so the rime pair's
    # qir0/qib0 cannot go missing on this birth path the way they once did
    # on the online one.
    assert np.array_equal(state.qir0, state.qir)
    assert np.array_equal(state.qib0, state.qib)
    assert np.array_equal(state.ni0, state.ni)
    assert np.array_equal(state.nr0, state.nr)


def test_p3_parent_reader_is_scheme_aware(tmp_path):
    """The bound form reads P3's own inventory; the blind form still
    holds the six-species closed world it has always promised."""
    path = tmp_path / "parent-mp50.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=50)
    moisture = read_parent_microphysics(path, source_mp_physics=50)
    assert set(moisture) == {
        "qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"}
    assert np.all(moisture["qir"] == np.float32(1.0e-4))
    # No scheme evidence means no smaller inventory is honestly complete:
    # a P3 archive has no QSNOW/QGRAUP, and the blind contract says so.
    with pytest.raises(OfflineChildContractError, match="mass fields"):
        read_parent_microphysics(path)


def test_p3_boundary_snapshot_carries_the_rime_pair_and_receipts_it(tmp_path):
    path = tmp_path / "parent-mp50.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=50, ny=18, nx=20)
    snapshot = interpolate_parent_boundary_snapshot(
        path, _p3_placement(), source_mp_physics=50, backend="cpu")
    assert set(snapshot.fields) == {
        "u", "v", "w", "theta", "phi", "mu",
        "qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"}
    # The receipt names the scheme's field set the way it names every
    # other scheme's: the full sorted inventory, rime pair included.
    assert snapshot.receipt["field_inventory"] == tuple(
        sorted(snapshot.fields))
    assert {"qib", "qir"} <= set(snapshot.receipt["field_inventory"])


def test_p3_streamed_lateral_intervals_include_the_rime_pair(tmp_path):
    start = datetime(1974, 4, 3, 12)
    paths = (tmp_path / "p0.nc", tmp_path / "p1.nc")
    _history(paths[0], start, mp=50, ny=18, nx=20, signal=0.0)
    _history(paths[1], start + timedelta(minutes=5), mp=50,
             ny=18, nx=20, signal=1.0)
    binding = _physics_binding(tmp_path, mp=50)
    contract = validate_parent_history(
        paths, max_boundary_interval_seconds=900, physics_binding=binding)
    assert contract.source_mp_physics == 50
    result = build_offline_lateral_boundaries(
        contract, _p3_placement(), backend="cpu")
    interval = result.boundaries.intervals[0]
    assert {"qi", "ni", "nr", "qir", "qib"} <= set(interval.fields)
    assert np.isfinite(interval.fields["qir"].west.value).all()
    assert np.all(interval.fields["qir"].west.value != 0.0)


def test_p3_cross_scheme_offline_edges_are_refused_by_name(tmp_path):
    """Both directions, both the initial-state and forcing paths.

    Same shape as mp=28's refusal test, DIFFERENT reason now: the online
    nest lane RATIFIED 50's rime-pair closure (it left
    UNVALIDATED_MIXED_EDGE_SELECTORS), so what refuses these edges is no
    longer the derived closure mirror but this module's own named gate
    (``_P3_OFFLINE_EDGE_UNBUILT_MP_PHYSICS``): the offline lane has no leg
    that runs the ratified merge/split maps, and converting without them
    would zero-fill or invent the rime pair.  Landing the leg retires the
    row.
    """
    p3 = tmp_path / "parent-mp50.nc"
    classic = tmp_path / "parent-mp8.nc"
    _history(p3, datetime(1974, 4, 3, 12), mp=50, ny=18, nx=20)
    _history(classic, datetime(1974, 4, 3, 12), mp=8, ny=18, nx=20)
    placement = _p3_placement()

    # 50 -> 18 (the one cross-scheme target the lane otherwise supports).
    with pytest.raises(OfflineChildContractError, match="REFUSED"):
        interpolate_parent_initial_state(
            p3, placement, source_mp_physics=50, target_mp_physics=18,
            backend="cpu")
    with pytest.raises(OfflineChildContractError, match="REFUSED"):
        interpolate_parent_boundary_snapshot(
            p3, placement, source_mp_physics=50, target_mp_physics=18,
            backend="cpu")

    # 8 -> 50 (a child the lane could not seed).
    with pytest.raises(OfflineChildContractError, match="REFUSED"):
        interpolate_parent_initial_state(
            classic, placement, source_mp_physics=8, target_mp_physics=50,
            backend="cpu")
    with pytest.raises(OfflineChildContractError, match="REFUSED"):
        interpolate_parent_boundary_snapshot(
            classic, placement, source_mp_physics=8, target_mp_physics=50,
            backend="cpu")

    # And the direct converter names P3's own closure, nobody else's.
    with pytest.raises(OfflineChildContractError) as caught:
        map_microphysics_to_nssl18({}, source_mp_physics=50)
    assert "REFUSED" in str(caught.value)
    assert "mp_physics=50" in str(caught.value)
    assert "qir/qib" in str(caught.value)
    assert "CCN reservoir" not in str(caught.value)


def test_p3_clamp_membership_matches_the_online_lane(tmp_path):
    """ni/nr are clamped members; the rime pair is DELIBERATELY not.

    The membership is the online nest lane's
    (gpuwm/ingest/nest_init.py::POSITIVE_DEFINITE_MOMENTS, imported, not
    re-spelled): number moments carry 1e3..1e9 per kilogram, so a float32
    SINT can round one across zero, while qir/qib are O(1e-4) mixing
    ratios whose absolute rounding error sits decades lower and has never
    been observed to cross -- clamping them would be a trajectory change
    with no evidence behind it.  This test pins that the two lanes keep
    agreeing: the same fabricated undershoot is cleaned off nr and left
    exactly where it is on qir.
    """
    path = tmp_path / "parent-mp50.nc"
    _history(path, datetime(1974, 4, 3, 12), mp=50, ny=18, nx=20,
             qnrain=_SINT_ROUNDING_UNDERSHOOT,
             qir=_SINT_ROUNDING_UNDERSHOOT)
    initial = interpolate_parent_initial_state(
        path, _p3_placement(),
        physics_binding=_physics_binding(tmp_path, mp=50), backend="cpu")

    assert initial.microphysics["nr"].min() == 0.0
    assert "nr" in initial.receipt["positive_definite_clamp"]
    assert initial.microphysics["qir"].min() == pytest.approx(
        _SINT_ROUNDING_UNDERSHOOT, rel=1e-6)
    assert "qir" not in initial.receipt["positive_definite_clamp"]
    assert "qib" not in initial.receipt["positive_definite_clamp"]


def test_offline_transport_inventory_derives_from_the_online_forcing_table():
    """Every admitted scheme's offline inventory IS the online one.

    ``_transported_source_fields`` and ``nest_field_kinds`` are two
    spellings of the same forcing promise; a species present in one and
    absent from the other is exactly how a scheme becomes half-supported.
    Pinned for the WHOLE admitted set so the next scheme's landing cannot
    drift the two lanes apart.
    """
    from types import SimpleNamespace

    from gpuwm.core.preflight import nest_field_kinds
    from gpuwm.offline_child import (
        OFFLINE_CHILD_MP_PHYSICS,
        _transported_source_fields,
    )

    dynamics = {"u", "v", "w", "t", "ph", "mu"}
    for mp in sorted(OFFLINE_CHILD_MP_PHYSICS):
        online = set(nest_field_kinds(
            SimpleNamespace(moist=True, mp_physics=mp))) - dynamics
        assert set(_transported_source_fields(mp)) == online, (
            f"mp_physics={mp}: offline transport inventory drifted from "
            "the online forcing table")


# ---------------------------------------------------------------------
# The child's own progress receipts.  A downscale used to publish
# report.json at the end and nothing a reader could bind to the process
# writing it, so no run browser could show one in flight.
# ---------------------------------------------------------------------

_CHILD_TOML = """[grid]
nx = 12
ny = 10
nz = 3

[run]
run_seconds = 900.0
output_interval_s = 900.0
grid_id = 1
"""


def test_a_child_is_named_after_its_parent_run_not_a_layout_folder():
    """"Downscale of wrfout" names nobody's forecast.

    The prepared routes write ``<stamp>/run/wrfout/wrfout_d01_*``, so the
    frame's own folder is a layout name and so is the one above it; the
    stamped run folder is the first name that says which run this was.
    A parent whose frames sit in its own directory keeps that directory.
    """
    from pathlib import PurePosixPath

    from gpuwm.offline_child_run import _parent_label

    frame = PurePosixPath(
        "/out/chain/run-20260910-214921Z_i202609091200Z/run/wrfout"
        "/wrfout_d01_2026-09-09_12_00_00")
    assert _parent_label(frame) == "run-20260910-214921Z_i202609091200Z"
    assert _parent_label(PurePosixPath(
        "/cases/parent-run/wrfout_d01_2026-09-09_12_00_00")) == "parent-run"


def _start_child_progress(progress, tmp_path):
    """Publish one child's manifest and stream, as ``run`` does."""
    outdir = tmp_path / "child-run"
    outdir.mkdir(exist_ok=True)
    config = tmp_path / "child.toml"
    config.write_text(_CHILD_TOML, encoding="utf-8")
    progress.start(
        outdir=outdir, child_config=config, ratio=3,
        start_time=datetime(1974, 4, 3, 12),
        parent={"run_dir": str(tmp_path), "restart": None, "frames": 3,
                "cadence_seconds": 900.0},
        name="Downscale of parent-run")
    return outdir, config


def test_child_publishes_the_manifest_every_other_run_publishes(tmp_path):
    """One manifest shape for every route, so every reader works unchanged."""
    import os

    from gpuwm import runplan
    from gpuwm.offline_child_run import _sha256

    progress = _ChildProgress()
    outdir, config = _start_child_progress(progress, tmp_path)
    try:
        manifest = json.loads(
            (outdir / runplan.MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["schema"] == runplan.MANIFEST_SCHEMA
        assert manifest["route"] == "downscale"
        assert manifest["pid"] == os.getpid()
        assert manifest["run_id"]
        # The binding a reader checks: run_dir == outputs_dir == the run's
        # own directory, and a plan source naming the exact config it ran
        # with that file's hash beside it.
        assert manifest["run_dir"] == manifest["outputs_dir"] == str(outdir)
        assert manifest["plan_source"] == f"gpuwm downscale {config.resolve()}"
        assert manifest["plan_sha256"] == _sha256(config.resolve())
        assert manifest["events_path"] == str(outdir / "events.jsonl")
        # No supervisor heartbeat on this route: null, never a path to a
        # file nothing writes.
        assert manifest["progress_path"] is None
        assert manifest["start_time"] == "1974-04-03T12:00:00Z"
        assert manifest["parent"]["frames"] == 3
        assert manifest["name"] == "Downscale of parent-run"

        progress.emit("stage_started", stage="forecast", phase="integrate")
        progress.emit("output_committed", domain=1,
                      valid_time="1974-04-03T12:00:00Z",
                      path=str(outdir / "wrfout_d01_1974-04-03_12_00_00"),
                      bytes=64)
        progress.emit("model_progress", domain=1, model_seconds=900.0,
                      run_seconds=900.0, outer_step=300, total_steps=300,
                      wall_seconds=1.5)
        progress.emit("completed", stage="forecast", result="PASS",
                      outputs=1)
    finally:
        progress.close()
    events = runplan.read_events(outdir / "events.jsonl")
    assert [event["event"] for event in events] == [
        "resolved_plan", "stage_started", "output_committed",
        "model_progress", "completed"]
    assert [event["sequence"] for event in events] == [1, 2, 3, 4, 5]
    assert all(event["schema_version"] == runplan.EVENT_SCHEMA
               for event in events)
    assert events[0]["config_source"] == str(config.resolve())
    assert events[0]["config_sha256"] == events[0]["config_sha256"]


def test_a_child_that_dies_says_so_in_its_own_stream(tmp_path, monkeypatch):
    """The last event a reader sees is always terminal.

    A run that raised used to leave a stream whose final record was an
    ordinary progress line, so a reader tailing it could not tell a dead
    child from a slow one.
    """
    from gpuwm import runplan
    import gpuwm.offline_child_run as child_run

    def explode(args, progress):
        _start_child_progress(progress, tmp_path)
        raise RuntimeError("offline child became non-finite at step 7")

    monkeypatch.setattr(child_run, "_run", explode)
    with pytest.raises(RuntimeError):
        child_run.run(object())
    events = runplan.read_events(tmp_path / "child-run" / "events.jsonl")
    assert events[-1]["event"] == "failed"
    assert "non-finite at step 7" in events[-1]["message"]
    assert events[-1]["message"].startswith("RuntimeError:")
def test_a_gpuwm_warm_rain_frame_is_not_claimed_as_kessler(tmp_path):
    """The R-018 residual: the writer, not the scheme, sets this inventory.

    Stock WRF's passiveqv (mp=0) transports qv alone, so a stock frame
    carrying QVAPOR/QCLOUD/QRAIN is Kessler and nothing else.  gpuwm's own
    mp=0 allocates and advects the warm-rain pair beside qv
    (_transported_source_fields says so in the same module) and
    gpuwm/io/wrfout.py writes all three whenever the state is moist, so on
    a gpuwm tape the same three names are mp=0 OR mp=1.  Naming one of them
    is the mislabel class this ladder was rewritten to end.
    """
    ours = tmp_path / "gpuwm-warm.nc"
    theirs = tmp_path / "wrf-warm.nc"
    _history(ours, datetime(1974, 4, 3, 12), mp=1)
    _history(theirs, datetime(1974, 4, 3, 12), mp=1, producer="wrf")

    mine = inspect_parent_history_frame(ours)
    assert mine.source_kind == "gpuwm"
    assert mine.inferred_mp_physics is None

    stock = inspect_parent_history_frame(theirs)
    assert stock.source_kind == "wrf"
    assert stock.inferred_mp_physics == 1

    # Both are still READABLE: the ambiguity is advisory, and declaring the
    # parent's scheme is the evidence-bearing route that always worked.
    for path in (ours, theirs):
        frame = inspect_parent_history_frame(path, source_mp_physics=1)
        assert frame.source_mp_physics == 1


def test_an_undeclared_parent_inferring_an_unsupported_scheme_is_refused(
        tmp_path):
    """A WDM6 archive satisfies the blind six-species contract.

    With nothing declared, the inventory is the only scheme evidence there
    is, and these arms are single-package discriminants rather than subset
    matches.  Left to fall through, the child was prepared with the
    parent's warm-rain numbers and its CCN reservoir dropped in silence --
    the same cross-scheme entry-closure breakage the online mixed nest edge
    refuses by name for mp=16.
    """
    frame = tmp_path / "wdm6.nc"
    _history(frame, datetime(1974, 4, 3, 12), mp=16)

    with pytest.raises(OfflineChildContractError) as excinfo:
        inspect_parent_history_frame(frame)
    message = str(excinfo.value)
    assert "mp_physics=16" in message
    assert "no parent scheme was declared" in message
    # It names the way out and the row an editor would change, not a bare
    # "unsupported".
    assert "consumers.offline_child" in message

    # Declaring 16 is refused too, on the same registry row -- the point is
    # that the two routes agree rather than one being a hole.
    with pytest.raises(OfflineChildContractError, match="mp_physics=16"):
        inspect_parent_history_frame(frame, source_mp_physics=16)


def test_an_admitted_scheme_inferred_from_a_frame_still_needs_no_declaration(
        tmp_path):
    """The refusal above must not have closed the undeclared route itself."""
    frame = tmp_path / "morrison.nc"
    _history(frame, datetime(1974, 4, 3, 12), mp=10)
    info = inspect_parent_history_frame(frame)
    assert info.inferred_mp_physics == 10
    assert info.source_mp_physics is None
