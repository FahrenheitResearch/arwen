from dataclasses import replace
from datetime import datetime, timedelta
import json

import netCDF4
import numpy as np
import pytest

from gpuwm.offline_child import (
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
from gpuwm.config import RunConfig
from gpuwm.offline_child_run import (
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
    with pytest.raises(FileExistsError):
        _create_output_root(requested)
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
    # surface-emission constants.  It is deliberately absent from
    # cross_scheme_transitions, which stays empty: every mixed edge touching
    # 28 is refused by name (see the two refusal tests below), matching the
    # online nest lane's UNVALIDATED_MIXED_EDGE_SELECTORS.
    assert capability["same_scheme_mp_physics"] == [6, 8, 10, 18, 28]
    assert capability["cross_scheme_transitions"] == []
    assert capability["vertical_remapping"] is False
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
             ny=3, nx=4, omit_aerosol_surface_emission=False):
    nz = 2
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("DateStrLen", 19), ("west_east", nx),
                ("south_north", ny), ("bottom_top", nz),
                ("west_east_stag", nx + 1),
                ("south_north_stag", ny + 1),
                ("bottom_top_stag", nz + 1)):
            dataset.createDimension(name, size)
        dataset.TITLE = "gpuwm offline-child fixture"
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
        dataset.GPUWM_WRITE_COMPLETE = 1
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.frombuffer(
            valid_time.strftime("%Y-%m-%d_%H:%M:%S").encode(), dtype="S1")
        mass3 = ("Time", "bottom_top", "south_north", "west_east")
        u3 = ("Time", "bottom_top", "south_north", "west_east_stag")
        v3 = ("Time", "bottom_top", "south_north_stag", "west_east")
        w3 = ("Time", "bottom_top_stag", "south_north", "west_east")
        mass2 = ("Time", "south_north", "west_east")
        for name in ("P", "QVAPOR", "QCLOUD", "QRAIN",
                     "QICE", "QSNOW", "QGRAUP"):
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
        if mp in {8, 10}:
            _variable(dataset, "QNRAIN", mass3,
                      np.zeros((1, nz, ny, nx)))
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


@pytest.mark.parametrize("parent_mp", [0, 1])
def test_offline_child_parent_scheme_refusal_names_the_switch(
        tmp_path, parent_mp):
    """A genuine unimplemented-selector refusal, watched at all four gates.

    This route's parent contract is NOT a profile whitelist: the child's
    hydrometeor mapping is written against the transported species of
    WSM6/Thompson/Morrison/NSSL, and mp 0 (no microphysics) and mp 1
    (Kessler, qc+qr only) have no mapping here -- so the 2026-07-31 suite
    ruling leaves these refusals standing.  What the ruling does demand
    is that each one name the exact switch and its value, and that the
    refusal be pinned rather than merely believed; before this test the
    four enforcement points had no coverage at all.
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
            match=f"source mp 6/8/10/18, got {parent_mp}"):
        map_microphysics_to_nssl18({}, source_mp_physics=parent_mp)

    with pytest.raises(
            OfflineChildContractError,
            match=f"mp_physics={parent_mp}"):
        _resolve_source_physics(parent_mp, None, None)

    # The four enforcement points read ONE named contract, so a scheme
    # can never be carried by one of them and refused by another.
    assert PARENT_SCHEME_CONTRACT == frozenset({6, 8, 10, 18})


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


def test_mp28_cross_scheme_offline_edges_are_refused_by_name(tmp_path):
    """Both directions, and both the initial-state and forcing paths.

    This must stay a REFUSAL rather than becoming the obvious
    nc = Nt_c/rho, nwfa = 11.1E6/rho closure: the online nest lane refuses
    the identical edge (gpuwm/core/microphysics_transition.py::
    UNVALIDATED_MIXED_EDGE_SELECTORS), and an offline path that quietly
    performed the closure would defeat that refusal instead of respecting
    it.
    """
    aero = tmp_path / "parent-mp28.nc"
    classic = tmp_path / "parent-mp8.nc"
    _history(aero, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20)
    _history(classic, datetime(1974, 4, 3, 12), mp=8, ny=18, nx=20)
    placement = _mp28_placement()

    # 28 -> 18 (the one cross-scheme target the lane otherwise supports).
    with pytest.raises(OfflineChildContractError, match="REFUSED"):
        interpolate_parent_initial_state(
            aero, placement, source_mp_physics=28, target_mp_physics=18,
            backend="cpu")
    with pytest.raises(OfflineChildContractError, match="REFUSED"):
        interpolate_parent_boundary_snapshot(
            aero, placement, source_mp_physics=28, target_mp_physics=18,
            backend="cpu")

    # 8 -> 28 (a child the lane could not seed).
    with pytest.raises(OfflineChildContractError, match="REFUSED"):
        interpolate_parent_initial_state(
            classic, placement, source_mp_physics=8, target_mp_physics=28,
            backend="cpu")
    with pytest.raises(OfflineChildContractError, match="REFUSED"):
        interpolate_parent_boundary_snapshot(
            classic, placement, source_mp_physics=8, target_mp_physics=28,
            backend="cpu")

    # And the direct converter, so the refusal cannot be bypassed by
    # calling it rather than the two wrappers.
    with pytest.raises(OfflineChildContractError, match="REFUSED"):
        map_microphysics_to_nssl18({}, source_mp_physics=28)
