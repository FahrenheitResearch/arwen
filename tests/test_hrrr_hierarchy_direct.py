"""Fail-closed contracts for the public HRRR hierarchy orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import pytest

from gpuwm.hrrr_hierarchy_direct import (
    _atomic_staging_sibling,
    _compare_stock_experiment,
    _require_raw_stock_delta,
    _require_raw_wps_contract,
    _supported_hierarchy_slice,
    _validated_root_preparation_binding,
)
from gpuwm.ingest.hrrr_target import HrrrTargetDomain
from gpuwm.native_wrf_contract import CERTIFIED_ETA_LEVELS


def test_atomic_staging_sibling_keeps_deep_windows_publication_short(tmp_path):
    """The d06 transaction must not repeat a long public output basename."""

    parent = tmp_path
    while len(str(parent)) < 125:
        parent /= "deep-path-budget-segment"
    parent.mkdir(parents=True)
    output = parent / ("hrrr-six-domain-z80-" + "x" * 72)

    staging = _atomic_staging_sibling(output)

    assert staging.parent == output.parent
    assert staging.name.startswith(".d-")
    assert len(staging.name) == len(".d-") + 10
    assert output.name not in staging.name
    # Reproduce the nested hierarchy/domain/prepared-cache publication depth
    # that raised WinError 206 in the genuine six-domain run.
    payload = (
        staging / ".d-0123456789" / "domains" / ".d-0123456789"
        / ".p-012345abcd" / "header.json"
    )
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"path-budget-pass")
    assert payload.read_bytes() == b"path-budget-pass"


@dataclass(frozen=True)
class _Run:
    grid_id: int = 1
    mp_physics: int = 6
    bl_pbl_physics: int = 1
    sf_sfclay_physics: int = 91
    sf_surface_physics: int = 2
    ra_physics: int = 0
    ra_lw_physics: int = 0
    ra_sw_physics: int = 1
    cu_physics: int = 0
    radt: float = 1.0
    radt_minutes: float = 1.0
    cudt_minutes: float = 0.0
    nx: int = 199
    ny: int = 199
    nz: int = 49
    dx: float = 2999.4213047435587
    dy: float = 2999.4213047435587
    dt: float = 15.0
    specified: bool = True
    nested: bool = False
    spec_zone: int = 1
    relax_zone: int = 4
    moist: bool = True
    moist_cq: bool = False
    nest_microphysics_transition: str = "same-scheme-only"


@dataclass(frozen=True)
class _Domain:
    grid_id: int
    run: _Run = _Run()
    parent_id: int = 0
    i_parent_start: int = 1
    j_parent_start: int = 1
    parent_grid_ratio: int = 1
    parent_time_step_ratio: int = 1
    history_interval_s: float = 300.0


@dataclass(frozen=True)
class _Projection:
    map_proj: str = "lambert"
    ref_lat: float = 35.5028506728143
    ref_lon: float = -98.0021669285660
    truelat1: float = 38.5
    truelat2: float = 38.5
    stand_lon: float = -97.5


@dataclass(frozen=True)
class _Vertical:
    eta_levels: tuple[float, ...] = CERTIFIED_ETA_LEVELS
    p_top: float = 10_000.0
    hybrid_opt: int = 2
    etac: float = 0.2


@dataclass(frozen=True)
class _Experiment:
    name: str
    domains: tuple[_Domain, ...]
    feedback: int = 0
    smooth_option: int = 0
    run_seconds: float = 43_200.0
    projection: _Projection = _Projection()
    vertical: _Vertical = _Vertical()
    spec_bdy_width: int = 5


def _native() -> _Experiment:
    child_run = replace(
        _Run(), grid_id=2, nx=300, ny=300, dx=999.8071015811862,
        dy=999.8071015811862, dt=5.0, specified=False, nested=True,
    )
    return _Experiment(
        name="native",
        domains=(
            _Domain(1),
            _Domain(
                2, child_run, parent_id=1, i_parent_start=50,
                j_parent_start=50, parent_grid_ratio=3,
                parent_time_step_ratio=3),
        ),
    )


def _target(**changes) -> HrrrTargetDomain:
    values = {
        "name": "hrrr_test",
        "map_proj": "lambert",
        "nx": 199,
        "ny": 199,
        "nz": 49,
        "dx_m": 2999.4213047435587,
        "dy_m": 2999.4213047435587,
        "ref_lat": 35.5028506728143,
        "ref_lon": -98.0021669285660,
        "truelat1": 38.5,
        "truelat2": 38.5,
        "stand_lon": -97.5,
        "time_step_seconds": 15,
        "spec_bdy_width": 5,
        "spec_zone": 1,
        "relax_zone": 4,
    }
    values.update(changes)
    return HrrrTargetDomain(**values)


def _raw_runtime_namelist(
        max_dom, *, longwave, theta_m, ghg_input=None, run_hours=12):
    def repeated(value):
        return ", ".join(str(value) for _ in range(max_dom))
    ghg = "" if ghg_input is None else f" ghg_input = {ghg_input},\n"
    return (
        "&time_control\n"
        f" run_hours = {run_hours},\n"
        " interval_seconds = 3600,\n"
        f" frames_per_outfile = {repeated(1)},\n"
        " restart = .false.,\n"
        " io_form_history = 2,\n io_form_restart = 2,\n"
        " io_form_input = 2,\n io_form_boundary = 2,\n/\n"
        "&domains\n"
        f" max_dom = {max_dom},\n"
        " num_metgrid_levels = 51,\n"
        " num_metgrid_soil_levels = 9,\n"
        " sfcp_to_sfcp = .true.,\n/\n"
        "&physics\n"
        f" ra_lw_physics = {repeated(longwave)},\n"
        f" ra_sw_physics = {repeated(1)},\n"
        " isfflx = 1,\n ifsnow = 1,\n icloud = 1,\n"
        " surface_input_source = 1,\n num_soil_layers = 4,\n"
        f" sf_urban_physics = {repeated(0)},\n"
        " sst_update = 0,\n"
        f"{ghg}/\n"
        f"&dynamics\n use_theta_m = {theta_m},\n/\n"
    )


def test_public_gate_accepts_generic_parent_ordered_easy_physics_slice():
    native = _native()
    target = _target()
    _supported_hierarchy_slice(native, target)
    _supported_hierarchy_slice(replace(native, domains=(_Domain(1),)), target)

    child3 = replace(
        native.domains[1], grid_id=3, parent_id=2,
        i_parent_start=100, j_parent_start=100,
        run=replace(
            native.domains[1].run, grid_id=3, nx=240, ny=240,
            dx=333.2690338603954, dy=333.2690338603954,
            dt=5.0 / 3.0))
    sibling4 = replace(
        native.domains[1], grid_id=4, parent_id=1,
        i_parent_start=70, j_parent_start=60,
        run=replace(
            native.domains[1].run, grid_id=4, nx=180, ny=180))
    generic = replace(
        native, domains=(*native.domains, child3, sibling4))
    _supported_hierarchy_slice(generic, target)

    with pytest.raises(ValueError, match="parent-before-child"):
        _supported_hierarchy_slice(replace(
            generic, domains=(generic.domains[0], generic.domains[2],
                              generic.domains[1], generic.domains[3])), target)
    with pytest.raises(ValueError, match="one-way"):
        _supported_hierarchy_slice(replace(_native(), feedback=1), target)
    # A mixed WSM6 -> Morrison edge that names no transition policy.
    #
    # This used to refuse as "unsupported mixed": before v1.3.1 the only
    # admitted mixed edge was Thompson -> NSSL.  The nest-edge lane
    # admits all 20 ordered mixed edges now, and the refusal moved from
    # "the pair is unsupported" to "the pair needs its policy named" --
    # which is the stronger statement, because a silent default is what
    # a cross-scheme translation must never have.  The gate is not
    # weakened: the same hierarchy still fails closed.
    drifted = replace(
        _native(), domains=(
            _native().domains[0],
            replace(_native().domains[1], run=replace(
                _native().domains[1].run, mp_physics=10))))
    with pytest.raises(ValueError,
                       match="requires explicit nest_microphysics_transition"):
        _supported_hierarchy_slice(drifted, target)

    # ...and naming it, with the moist/CQ contract the transition also
    # requires on both domains, admits the same tree.  That is the other
    # half of the contract, and it is what makes the refusal above a
    # named gate rather than a blanket one.
    admitted = replace(
        _native(), domains=(
            replace(_native().domains[0], run=replace(
                _native().domains[0].run, moist=True, moist_cq=True)),
            replace(_native().domains[1], run=replace(
                _native().domains[1].run, mp_physics=10,
                moist=True, moist_cq=True,
                nest_microphysics_transition="mp-edge-mass-diagnosed-v1"))))
    _supported_hierarchy_slice(admitted, target)


def test_public_gate_admits_explicit_thompson_to_nssl_tree_without_consent():
    base = _native()
    root_run = replace(base.domains[0].run, mp_physics=8, moist_cq=True)
    d02_run = replace(
        base.domains[1].run, mp_physics=8, moist_cq=True)
    d03_run = replace(
        d02_run, grid_id=3, nx=180, ny=180,
        dx=d02_run.dx / 3.0, dy=d02_run.dy / 3.0,
        dt=d02_run.dt / 3.0, mp_physics=18,
        nest_microphysics_transition=(
            "mp8-to-mp18-mass-diagnosed-v1"))
    d04_run = replace(
        d03_run, grid_id=4, nx=120, ny=120,
        dx=d03_run.dx / 2.0, dy=d03_run.dy / 2.0,
        dt=d03_run.dt / 2.0,
        nest_microphysics_transition="same-scheme-only")
    mixed = replace(base, domains=(
        replace(base.domains[0], run=root_run),
        replace(base.domains[1], run=d02_run),
        _Domain(
            3, d03_run, parent_id=2, i_parent_start=20,
            j_parent_start=20, parent_grid_ratio=3,
            parent_time_step_ratio=3),
        _Domain(
            4, d04_run, parent_id=3, i_parent_start=20,
            j_parent_start=20, parent_grid_ratio=2,
            parent_time_step_ratio=2),
    ))

    _supported_hierarchy_slice(mixed, _target())

    missing_policy = replace(
        mixed, domains=(*mixed.domains[:2], replace(
            mixed.domains[2], run=replace(
                mixed.domains[2].run,
                nest_microphysics_transition="same-scheme-only")),
            mixed.domains[3]))
    with pytest.raises(ValueError, match="requires explicit"):
        _supported_hierarchy_slice(missing_policy, _target())


def test_public_gate_accepts_short_ohio_z80_sealed_root():
    target = _target(
        name="hrrr_ohio_z80", nx=192, ny=160, nz=80,
        dx_m=3000.0, dy_m=3000.0, ref_lat=40.0, ref_lon=-83.0,
        truelat1=30.0, truelat2=60.0, stand_lon=-84.0)
    root_run = replace(
        _Run(), nx=192, ny=160, nz=80, dx=3000.0, dy=3000.0)
    child_run = replace(
        root_run, grid_id=2, nx=90, ny=90, dx=1000.0, dy=1000.0,
        dt=5.0, specified=False, nested=True)
    vertical = _Vertical(
        eta_levels=tuple(1.0 - i / 80.0 for i in range(81)))
    short = _Experiment(
        name="ohio-z80", run_seconds=3600.0,
        projection=_Projection(
            ref_lat=40.0, ref_lon=-83.0, truelat1=30.0,
            truelat2=60.0, stand_lon=-84.0),
        vertical=vertical,
        domains=(
            _Domain(1, root_run),
            _Domain(
                2, child_run, parent_id=1, i_parent_start=30,
                j_parent_start=30, parent_grid_ratio=3,
                parent_time_step_ratio=3),
        ),
    )
    _supported_hierarchy_slice(short, target)


def test_root_cache_reuse_binds_effective_d01_not_child_topology():
    native = _native()
    identity = {
        "domain_config": asdict(native.domains[0]),
        "namelist_sha256": "a" * 64,
    }
    digest, prepared = _validated_root_preparation_binding(
        identity, native.domains[0])
    assert digest == "a" * 64
    assert prepared == identity["domain_config"]

    deeper = replace(native, domains=(*native.domains, replace(
        native.domains[1], grid_id=3, parent_id=2,
        run=replace(native.domains[1].run, grid_id=3))))
    assert _validated_root_preparation_binding(
        identity, deeper.domains[0])[0] == "a" * 64

    drifted_root = replace(
        native.domains[0], run=replace(native.domains[0].run, mp_physics=10))
    with pytest.raises(ValueError, match="d01 trajectory controls differ"):
        _validated_root_preparation_binding(identity, drifted_root)
    with pytest.raises(ValueError, match="invalid namelist_sha256"):
        _validated_root_preparation_binding(
            {**identity, "namelist_sha256": "editable"}, native.domains[0])


def test_stock_comparison_allows_only_explicit_longwave_runtime_change():
    native = _native()
    stock = replace(native, name="stock", domains=tuple(
        replace(domain, run=replace(domain.run, ra_lw_physics=1))
        for domain in native.domains))
    _compare_stock_experiment(native, stock)

    physics_drift = replace(stock, domains=(
        stock.domains[0],
        replace(stock.domains[1], run=replace(
            stock.domains[1].run, sf_surface_physics=0)),
    ))
    with pytest.raises(ValueError, match="beyond the allowed"):
        _compare_stock_experiment(native, physics_drift)

    with pytest.raises(ValueError, match="must select ra_lw_physics=1"):
        _compare_stock_experiment(native, replace(stock, domains=(
            stock.domains[0], native.domains[1])))


def test_stock_comparison_rejects_forecast_duration_drift():
    native = _native()
    stock = replace(native, name="stock", run_seconds=15.0, domains=tuple(
        replace(domain, run=replace(domain.run, ra_lw_physics=1))
        for domain in native.domains))
    with pytest.raises(ValueError, match="beyond the allowed"):
        _compare_stock_experiment(native, stock)


@pytest.mark.parametrize("max_dom", range(1, 22))
def test_raw_namelist_gate_allows_only_explicit_runtime_deltas(
        tmp_path, max_dom):
    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native.write_text(
        _raw_runtime_namelist(max_dom, longwave=0, theta_m=0),
        encoding="ascii")
    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=1, ghg_input=0),
        encoding="ascii")
    receipt = _require_raw_stock_delta(native, stock)
    assert set(receipt["allowed_deltas"]) == {
        "physics.ra_lw_physics", "dynamics.use_theta_m",
        "physics.ghg_input"}
    assert receipt["max_dom"] == max_dom

    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=1, ghg_input=0, run_hours=6),
        encoding="ascii")
    with pytest.raises(ValueError, match="time_control/run_hours"):
        _require_raw_stock_delta(native, stock)

    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=1, ghg_input=1),
        encoding="ascii")
    with pytest.raises(ValueError, match="stock-only ghg_input=0"):
        _require_raw_stock_delta(native, stock)


@pytest.mark.parametrize(
    "old, new, key",
    [
        ("interval_seconds = 3600", "interval_seconds = 1800",
         "interval_seconds"),
        ("sfcp_to_sfcp = .true.", "sfcp_to_sfcp = .false.",
         "sfcp_to_sfcp"),
        ("io_form_input = 2", "io_form_input = 3", "io_form_input"),
        ("isfflx = 1", "isfflx = 0", "isfflx"),
        ("num_soil_layers = 4", "num_soil_layers = 9",
         "num_soil_layers"),
        ("interval_seconds = 3600", "interval_seconds = 3600.0",
         "interval_seconds"),
        ("restart = .false.", "restart = 0", "restart"),
        ("sfcp_to_sfcp = .true.", "sfcp_to_sfcp = 1",
         "sfcp_to_sfcp"),
        ("io_form_input = 2", "io_form_input = 2.0", "io_form_input"),
        ("isfflx = 1", "isfflx = 1.0", "isfflx"),
    ],
)
def test_raw_namelist_gate_rejects_dropped_runtime_drift(
        tmp_path, old, new, key):
    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native_text = _raw_runtime_namelist(4, longwave=0, theta_m=0)
    stock_text = _raw_runtime_namelist(
        4, longwave=1, theta_m=1, ghg_input=0)
    assert old in native_text and old in stock_text
    native.write_text(native_text.replace(old, new), encoding="ascii")
    stock.write_text(stock_text.replace(old, new), encoding="ascii")
    with pytest.raises(ValueError, match=key):
        _require_raw_stock_delta(native, stock)


def test_raw_wps_gate_binds_hourly_hierarchy_contract(tmp_path):
    wps = tmp_path / "namelist.wps"
    wps.write_text(
        "&share\n max_dom = 4,\n interval_seconds = 3600,\n/\n",
        encoding="ascii")
    assert _require_raw_wps_contract(wps, 4)["status"] == "PASS"

    wps.write_text(
        "&share\n max_dom = 4,\n interval_seconds = 1800,\n/\n",
        encoding="ascii")
    with pytest.raises(ValueError, match="interval_seconds"):
        _require_raw_wps_contract(wps, 4)

    wps.write_text(
        "&share\n max_dom = 4,\n interval_seconds = 3600.0,\n/\n",
        encoding="ascii")
    with pytest.raises(ValueError, match="interval_seconds"):
        _require_raw_wps_contract(wps, 4)
